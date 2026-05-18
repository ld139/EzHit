#!/usr/bin/env python3
"""
predict.py

Local EZHit inference for enzyme-reaction compatibility prediction.

This script supports:
1. Single-pair prediction from command-line inputs.
2. Batch prediction from a CSV file.
3. Local checkpoints or automatic checkpoint download from HuggingFace Hub.
4. Ensemble prediction across multiple seed checkpoints.
5. Optional Mahalanobis-distance inference if train_distribution_stat.pt is provided.

Example
-------
python predict.py \
    --input_csv examples/predict_demo.csv \
    --seq_col protein_sequence \
    --rxn_col CANO_RXN_SMILES \
    --model_group general \
    --seeds 40 41 42 43 44 \
    --standardize_smiles \
    --uncharge \
    --filter_charged_single_atom \
    --output_csv results/predict_demo_output.csv
"""

import os
import re
import glob
import argparse
from hashlib import blake2b
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from drfp import DrfpEncoder
from transformers import T5Tokenizer, T5EncoderModel

from kan import KAN

RDLogger.DisableLog("rdApp.*")


# -----------------------
# DRFP compatibility patch
# -----------------------
def patch_drfp_numpy2_overflow():
    """
    DRFP 0.3.6 may create np.array(..., dtype=np.int32) from unsigned 32-bit hash
    values. NumPy 2.x raises OverflowError for values above int32 max. Returning
    int64 keeps the hash value and avoids the overflow.
    """
    def safe_hash(shingling):
        hash_values = []
        for t in shingling:
            if isinstance(t, str):
                t = t.encode("utf-8")
            hash_values.append(int(blake2b(t, digest_size=4).hexdigest(), 16))
        return np.asarray(hash_values, dtype=np.int64)

    DrfpEncoder.hash = staticmethod(safe_hash)


patch_drfp_numpy2_overflow()


# -----------------------
# Model
# -----------------------
class HybridPairKAN(nn.Module):
    def __init__(self, esm_dim: int, drfp_dim: int, react_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.esm_ln = nn.LayerNorm(esm_dim)
        self.drfp_ln = nn.LayerNorm(drfp_dim)
        self.react_ln = nn.LayerNorm(react_dim)

        in_dim = esm_dim + drfp_dim + react_dim
        self.bottleneck_dim = hidden

        self.compressor = nn.Sequential(
            nn.Linear(in_dim, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.net = KAN([self.bottleneck_dim, hidden, 1])

    def forward(
        self,
        esm: torch.Tensor,
        drfp: torch.Tensor,
        react: torch.Tensor,
        update_grid: bool = False,
        return_latent: bool = False,
    ):
        e_norm = self.esm_ln(esm)
        d_norm = self.drfp_ln(drfp)
        r_norm = self.react_ln(react)

        x = torch.cat([e_norm, d_norm, r_norm], dim=-1)
        latent = self.compressor(x)
        logit = self.net(latent, update_grid=update_grid).squeeze(-1)

        if return_latent:
            return logit, latent
        return logit


# -----------------------
# Reaction preprocessing and fingerprints
# -----------------------
def standardize_reaction_smiles(
    rxn_smiles: str,
    preserve_direction: bool = False,
    uncharge: bool = False,
    filter_charged_single_atom: bool = False,
) -> Optional[str]:
    if rxn_smiles is None or not isinstance(rxn_smiles, str):
        return None
    if ">>" not in rxn_smiles:
        return None

    uncharger = rdMolStandardize.Uncharger() if uncharge else None
    sides = rxn_smiles.split(">>")
    new_parts = []

    for side in sides:
        mols = []
        for s in side.split("."):
            s = s.strip()
            if not s:
                continue
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    continue

                if filter_charged_single_atom and mol.GetNumAtoms() == 1:
                    atom = mol.GetAtomWithIdx(0)
                    if atom.GetFormalCharge() != 0:
                        continue

                if uncharger is not None:
                    try:
                        mol = uncharger.uncharge(mol)
                    except Exception:
                        pass

                smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                if smi:
                    mols.append(smi)
            except Exception:
                continue

        mols.sort()
        new_parts.append(".".join(mols))

    if len(new_parts) < 2 or not new_parts[0] or not new_parts[1]:
        return None

    if not preserve_direction:
        new_parts.sort()

    return ">>".join(new_parts)


def l2norm_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def make_drfp_features(
    rxn_smiles: List[str],
    fp_dim: int = 2048,
    standardize_smiles: bool = False,
    preserve_direction: bool = False,
    uncharge: bool = False,
    filter_charged_single_atom: bool = False,
    failure_mode: str = "strict",
) -> torch.Tensor:
    used = []

    for s in rxn_smiles:
        if standardize_smiles:
            ss = standardize_reaction_smiles(
                s,
                preserve_direction=preserve_direction,
                uncharge=uncharge,
                filter_charged_single_atom=filter_charged_single_atom,
            )
            used.append("" if ss is None else ss)
        else:
            used.append(str(s))

    uniq = []
    smi2uid = {}
    uids = np.empty(len(used), dtype=np.int64)

    for i, s in enumerate(used):
        uid = smi2uid.get(s)
        if uid is None:
            uid = len(uniq)
            smi2uid[s] = uid
            uniq.append(s)
        uids[i] = uid

    print(f"[DRFP] rows={len(used):,} unique_rxn={len(uniq):,}")

    fps_u = np.zeros((len(uniq), fp_dim), dtype=np.float32)
    failed = []

    for i, s in enumerate(uniq):
        try:
            fps_u[i] = np.asarray(DrfpEncoder.encode([s], n_folded_length=fp_dim)[0], dtype=np.float32)
        except Exception as e:
            failed.append((i, s, str(e)))
            if failure_mode == "zero_vector":
                fps_u[i] = np.zeros((fp_dim,), dtype=np.float32)

    if failed and failure_mode == "strict":
        msg = "\n".join([f"[uid={i}] {s} | {err[:300]}" for i, s, err in failed[:10]])
        raise RuntimeError(
            f"DRFP failed for {len(failed)} unique reaction(s). First failures:\n{msg}\n"
            "Fix these reaction SMILES or use --drfp_failure_mode zero_vector for quick testing."
        )

    fps_u = l2norm_np(fps_u)
    return torch.tensor(fps_u[uids], dtype=torch.float32)


def get_reactant_morgan_fp(rxn_smiles: str, radius: int = 2, nBits: int = 2048) -> np.ndarray:
    if not isinstance(rxn_smiles, str) or ">>" not in rxn_smiles:
        return np.zeros((nBits,), dtype=np.float32)

    reactants_smi = rxn_smiles.split(">>")[0]
    mol = Chem.MolFromSmiles(reactants_smi)

    if mol is None:
        return np.zeros((nBits,), dtype=np.float32)

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def make_reactant_morgan_features(rxn_smiles: List[str], fp_dim: int = 2048) -> torch.Tensor:
    uniq = []
    smi2uid = {}
    uids = np.empty(len(rxn_smiles), dtype=np.int64)

    for i, s in enumerate(rxn_smiles):
        s = str(s)
        uid = smi2uid.get(s)
        if uid is None:
            uid = len(uniq)
            smi2uid[s] = uid
            uniq.append(s)
        uids[i] = uid

    print(f"[Reactant FP] rows={len(rxn_smiles):,} unique_rxn={len(uniq):,}")

    fps_u = np.zeros((len(uniq), fp_dim), dtype=np.float32)
    for i, s in enumerate(uniq):
        fps_u[i] = get_reactant_morgan_fp(s, radius=2, nBits=fp_dim)

    fps_u = l2norm_np(fps_u)
    return torch.tensor(fps_u[uids], dtype=torch.float32)


# -----------------------
# Protein encoder
# -----------------------
def preprocess_seq_for_prott5(seq: str, max_len: int = 1022) -> str:
    seq = str(seq).strip()[:max_len]
    seq = re.sub(r"[UZOBuzob]", "X", seq.upper())
    return " ".join(list(seq))


def mean_pool_reps(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask_expanded).sum(dim=1)
    denom = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def make_dynamic_batches(lengths: List[int], max_tokens: int) -> List[List[int]]:
    order = np.argsort(-np.asarray(lengths))
    batches = []
    cur = []
    cur_tok = 0

    for idx in order.tolist():
        L = int(lengths[idx])
        if not cur:
            cur = [idx]
            cur_tok = L
            continue

        if cur_tok + L <= max_tokens:
            cur.append(idx)
            cur_tok += L
        else:
            batches.append(cur)
            cur = [idx]
            cur_tok = L

    if cur:
        batches.append(cur)

    return batches


@torch.no_grad()
def encode_proteins_dynamic(
    sequences: List[str],
    model_name: str,
    device: torch.device,
    max_len: int = 1022,
    max_tokens: int = 4000,
) -> torch.Tensor:
    uniq = []
    seq2uid = {}
    uids = np.empty(len(sequences), dtype=np.int64)

    for i, seq in enumerate(sequences):
        seq = str(seq)
        uid = seq2uid.get(seq)
        if uid is None:
            uid = len(uniq)
            seq2uid[seq] = uid
            uniq.append(seq)
        uids[i] = uid

    print(f"[ProtT5] rows={len(sequences):,} unique={len(uniq):,}")

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)

    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        model = T5EncoderModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
    else:
        dtype = torch.float32
        model = T5EncoderModel.from_pretrained(model_name).to(device)

    model.eval()

    processed = [preprocess_seq_for_prott5(s, max_len=max_len) for s in uniq]
    lengths = [min(len(str(s)), max_len) + 1 for s in uniq]
    batches = make_dynamic_batches(lengths, max_tokens=max_tokens)

    print(f"[ProtT5] dynamic batches={len(batches):,}, max_tokens={max_tokens}")

    out = None

    for batch_ids in batches:
        batch_seqs = [processed[i] for i in batch_ids]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len + 1)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if torch.cuda.is_available():
            with torch.autocast("cuda", dtype=dtype):
                outputs = model(**inputs)
                pooled = mean_pool_reps(outputs.last_hidden_state, inputs["attention_mask"])
        else:
            outputs = model(**inputs)
            pooled = mean_pool_reps(outputs.last_hidden_state, inputs["attention_mask"])

        if out is None:
            out = torch.zeros((len(uniq), pooled.shape[1]), dtype=torch.float32)

        out[batch_ids] = pooled.detach().cpu().float()

    if out is None:
        raise RuntimeError("No protein sequence was encoded.")

    return out[uids].contiguous()


# -----------------------
# Checkpoints and inference
# -----------------------
def hf_checkpoint_filenames(model_group: str, seeds: List[int]) -> List[str]:
    patterns = {
        "general": "checkpoints/binarycls_best_val_seed{seed}.pt",
        "p450": "checkpoints/ft_p450_best_seed{seed}.pt",
        "phosphatase": "checkpoints/ft_phosphatase_best_seed{seed}.pt",
        "terpene": "checkpoints/ft_terpene_best_seed{seed}.pt",
    }

    if model_group not in patterns:
        raise ValueError(f"Unknown model_group: {model_group}")

    return [patterns[model_group].format(seed=s) for s in seeds]


def resolve_checkpoint_paths(args) -> List[str]:
    if args.ckpt_paths:
        paths = []
        for item in args.ckpt_paths:
            paths.extend(glob.glob(item))
        paths = sorted(paths)
        if not paths:
            raise FileNotFoundError("No checkpoint files matched --ckpt_paths.")
        return paths

    from huggingface_hub import hf_hub_download

    paths = []
    for filename in hf_checkpoint_filenames(args.model_group, args.seeds):
        print(f"[HF] downloading {args.hf_repo}/{filename}")
        paths.append(hf_hub_download(repo_id=args.hf_repo, filename=filename))

    return paths


def load_models(ckpt_paths: List[str], device: torch.device, dropout: float = 0.0) -> List[nn.Module]:
    models = []

    for path in ckpt_paths:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

        hidden = int(ckpt_args.get("hidden", 512))

        model = HybridPairKAN(
            esm_dim=1024,
            drfp_dim=2048,
            react_dim=2048,
            hidden=hidden,
            dropout=dropout,
        )
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        models.append(model)
        print(f"[Model] loaded {path} | hidden={hidden}")

    return models


def load_maha_stats(path: str, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mahalanobis stat file not found: {path}")

    stat = torch.load(path, map_location=device)
    if "mean" not in stat or "inv_cov" not in stat:
        raise KeyError("Mahalanobis stat file must contain keys: 'mean' and 'inv_cov'.")

    return {
        "mean": stat["mean"].float().to(device).view(-1),
        "inv_cov": stat["inv_cov"].float().to(device),
    }


@torch.no_grad()
def predict_ensemble(
    models: List[nn.Module],
    esm: torch.Tensor,
    drfp: torch.Tensor,
    react: torch.Tensor,
    device: torch.device,
    maha_stats: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: int = 512,
) -> pd.DataFrame:
    all_mean = []
    all_std = []
    all_mi = []
    all_maha = []

    eps = 1e-10

    for start in range(0, esm.shape[0], batch_size):
        end = min(start + batch_size, esm.shape[0])

        e = esm[start:end].to(device)
        f = drfp[start:end].to(device)
        r = react[start:end].to(device)

        probs = []
        latents = []

        for model in models:
            logit, latent = model(e, f, r, return_latent=True)
            prob = torch.sigmoid(logit)
            probs.append(prob)
            latents.append(latent)

        prob_stack = torch.stack(probs, dim=0)
        mean_prob = prob_stack.mean(dim=0)
        std_prob = prob_stack.std(dim=0) if len(models) > 1 else torch.zeros_like(mean_prob)

        entropy_mean = -(
            mean_prob * torch.log(mean_prob + eps)
            + (1.0 - mean_prob) * torch.log(1.0 - mean_prob + eps)
        )
        entropy_each = -(
            prob_stack * torch.log(prob_stack + eps)
            + (1.0 - prob_stack) * torch.log(1.0 - prob_stack + eps)
        )
        mi = entropy_mean - entropy_each.mean(dim=0)

        all_mean.append(mean_prob.detach().cpu())
        all_std.append(std_prob.detach().cpu())
        all_mi.append(mi.detach().cpu())

        if maha_stats is not None:
            latent_mean = torch.stack(latents, dim=0).mean(dim=0)
            mu = maha_stats["mean"]
            inv_cov = maha_stats["inv_cov"]

            if latent_mean.shape[-1] != mu.shape[0]:
                raise ValueError(f"Latent dimension mismatch: latent={latent_mean.shape[-1]}, stat={mu.shape[0]}")

            delta = latent_mean.float() - mu.view(1, -1)
            maha_sq = torch.sum(delta * torch.matmul(delta, inv_cov), dim=-1)
            maha = torch.sqrt(torch.clamp(maha_sq, min=1e-9))
            all_maha.append(maha.detach().cpu())

    out = {
        "EZHit_probability": torch.cat(all_mean).numpy(),
        "EZHit_probability_std": torch.cat(all_std).numpy(),
        "EZHit_ensemble_MI": torch.cat(all_mi).numpy(),
    }

    if maha_stats is not None:
        out["Mahalanobis_Dist"] = torch.cat(all_maha).numpy()

    return pd.DataFrame(out)


def build_input_dataframe(args) -> pd.DataFrame:
    if args.input_csv:
        df = pd.read_csv(args.input_csv, low_memory=False)
        if args.seq_col not in df.columns:
            raise ValueError(f"Missing sequence column: {args.seq_col}")
        if args.rxn_col not in df.columns:
            raise ValueError(f"Missing reaction column: {args.rxn_col}")
        return df

    if not args.protein_sequence or not args.rxn_smiles:
        raise ValueError("Provide either --input_csv or both --protein_sequence and --rxn_smiles.")

    return pd.DataFrame({
        args.seq_col: [args.protein_sequence],
        args.rxn_col: [args.rxn_smiles],
    })


def main():
    ap = argparse.ArgumentParser(description="Local EZHit inference")

    ap.add_argument("--input_csv", type=str, default="", help="CSV containing enzyme sequences and reaction SMILES")
    ap.add_argument("--protein_sequence", type=str, default="", help="Single enzyme sequence")
    ap.add_argument("--rxn_smiles", type=str, default="", help="Single reaction SMILES")
    ap.add_argument("--seq_col", type=str, default="protein_sequence")
    ap.add_argument("--rxn_col", type=str, default="CANO_RXN_SMILES")
    ap.add_argument("--output_csv", type=str, default="ezhit_predictions.csv")

    ap.add_argument("--ckpt_paths", nargs="*", default=[], help="Local checkpoint paths or glob patterns")
    ap.add_argument("--hf_repo", type=str, default="deanluo/EzHit")
    ap.add_argument("--model_group", type=str, default="general", choices=["general", "p450", "phosphatase", "terpene"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])

    ap.add_argument("--maha_stat_path", type=str, default="", help="Path to train_distribution_stat.pt")

    ap.add_argument("--prott5_model", type=str, default="Rostlab/prot_t5_xl_half_uniref50-enc")
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--max_tokens", type=int, default=4000)
    ap.add_argument("--fp_dim", type=int, default=2048)
    ap.add_argument("--reactant_fp_dim", type=int, default=2048)

    ap.add_argument("--standardize_smiles", action="store_true")
    ap.add_argument("--preserve_direction", action="store_true")
    ap.add_argument("--uncharge", action="store_true")
    ap.add_argument("--filter_charged_single_atom", action="store_true")
    ap.add_argument("--drfp_failure_mode", type=str, default="strict", choices=["strict", "zero_vector"])

    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    df = build_input_dataframe(args)
    seqs = df[args.seq_col].fillna("").astype(str).tolist()
    rxns = df[args.rxn_col].fillna("").astype(str).tolist()

    print(f"[Input] rows={len(df):,}")

    print("[Feature] encoding proteins...")
    esm = encode_proteins_dynamic(
        seqs,
        model_name=args.prott5_model,
        device=device,
        max_len=args.max_len,
        max_tokens=args.max_tokens,
    )

    print("[Feature] encoding DRFP...")
    drfp = make_drfp_features(
        rxns,
        fp_dim=args.fp_dim,
        standardize_smiles=args.standardize_smiles,
        preserve_direction=args.preserve_direction,
        uncharge=args.uncharge,
        filter_charged_single_atom=args.filter_charged_single_atom,
        failure_mode=args.drfp_failure_mode,
    )

    print("[Feature] encoding reactant Morgan FP...")
    react = make_reactant_morgan_features(rxns, fp_dim=args.reactant_fp_dim)

    print("[Model] resolving checkpoints...")
    ckpt_paths = resolve_checkpoint_paths(args)
    models = load_models(ckpt_paths, device=device, dropout=0.0)

    maha_stats = load_maha_stats(args.maha_stat_path, device=device) if args.maha_stat_path else None

    print("[Predict] running ensemble inference...")
    pred_df = predict_ensemble(
        models=models,
        esm=esm,
        drfp=drfp,
        react=react,
        device=device,
        maha_stats=maha_stats,
        batch_size=args.batch_size,
    )

    out_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    out_df.to_csv(args.output_csv, index=False)

    print(f"[Done] saved predictions to: {args.output_csv}")
    print(out_df.head())


if __name__ == "__main__":
    main()
