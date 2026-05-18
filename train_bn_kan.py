"""
Binary classification baseline (sigmoid) for enzyme-reaction pairs.
Revised to include Global Enrichment Factor (EF) calculation matching EnzymeCAGE paper methods.

Metrics:
  - Top-k Success Rate & DCG@10: Calculated per reaction group (Local).
  - Enrichment Factor (EF): Calculated across the entire dataset (Global).
"""

import os
import math
import argparse
from typing import Any, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score
from kan import KAN
# ---------- DRFP ----------
try:
    from drfp import DrfpEncoder
    HAS_DRFP = True
except ImportError:
    HAS_DRFP = False

# ---------- RDKit ----------
try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import rdMolDescriptors  # [NEW] 用于提取 Morgan 指纹
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False


# ----------------------- ESM cache helpers (row-aligned only) -----------------------
def _unwrap_esm_cache_to_tensor(obj: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)) and len(obj) > 0 and all(torch.is_tensor(x) for x in obj):
        return torch.cat(list(obj), dim=0)
    if isinstance(obj, dict):
        for k in ["embeddings", "embedding", "esm", "repr", "features", "mean"]:
            if k in obj:
                v = obj[k]
                if torch.is_tensor(v):
                    return v
                if isinstance(v, (list, tuple)) and len(v) > 0 and all(torch.is_tensor(x) for x in v):
                    return torch.cat(list(v), dim=0)
    return None


def load_row_aligned_cache(path: str, expected_rows: int) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    t = _unwrap_esm_cache_to_tensor(obj)
    if t is None or t.dim() != 2 or t.shape[0] != expected_rows:
        raise ValueError(
            f"Bad row-aligned cache: {path}, got {None if t is None else tuple(t.shape)}, expected ({expected_rows}, D)"
        )
    return t.float()


# ----------------------- RDKit reaction standardization -----------------------
def standardize_reaction_smiles(
    rxn_smiles: str,
    preserve_direction: bool,
    uncharge: bool,
    filter_charged_single_atom: bool,
) -> Optional[str]:
    if rxn_smiles is None or not isinstance(rxn_smiles, str):
        return None
    if ">>" not in rxn_smiles:
        return None

    if not HAS_RDKIT:
        parts = [p.strip() for p in rxn_smiles.split(">>")]
        if len(parts) < 2 or parts[0] == "" or parts[1] == "":
            return None
        parts = [".".join(sorted([m.strip() for m in side.split(".") if m.strip()])) for side in parts]
        if not preserve_direction:
            parts.sort()
        return ">>".join(parts)

    uncharger = rdMolStandardize.Uncharger() if uncharge else None
    sides = rxn_smiles.split(">>")
    new_parts: List[str] = []

    for side in sides:
        mols: List[str] = []
        for s in side.split("."):
            s = s.strip()
            if not s:
                continue
            try:
                m = Chem.MolFromSmiles(s)
                if m is None:
                    continue

                if filter_charged_single_atom and m.GetNumAtoms() == 1:
                    atom = m.GetAtomWithIdx(0)
                    if atom.GetFormalCharge() != 0:
                        continue

                if uncharger is not None:
                    try:
                        m = uncharger.uncharge(m)
                    except Exception:
                        pass

                smi = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
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


# ----------------------- DRFP (dedup + cache) -----------------------
def drfp_encode(smiles_list: List[str], n_folded_length: int = 2048) -> np.ndarray:
    fps = DrfpEncoder.encode(smiles_list, n_folded_length=n_folded_length)
    return np.asarray(fps, dtype=np.float32)


def l2norm_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def make_drfp_rowaligned_dedup(
    df: pd.DataFrame,
    rxn_col: str,
    fp_dim: int,
    standardize_flag: bool,
    preserve_direction: bool,
    uncharge: bool,
    filter_charged_single_atom: bool,
) -> torch.Tensor:
    smis = df[rxn_col].fillna("").astype(str).tolist()

    if standardize_flag:
        out = []
        fail = 0
        for s in tqdm(smis, desc="[RDKit] Standardize"):
            ss = standardize_reaction_smiles(
                s,
                preserve_direction=preserve_direction,
                uncharge=uncharge,
                filter_charged_single_atom=filter_charged_single_atom,
            )
            if ss is None:
                out.append("")
                fail += 1
            else:
                out.append(ss)
        print(f"[Diag] std_fail: {fail}/{len(smis)} ({fail/max(1,len(smis)):.2%})")
        smis = out

    uniq: List[str] = []
    smi2uid: Dict[str, int] = {}
    uids = np.empty((len(smis),), dtype=np.int64)
    for i, s in enumerate(smis):
        uid = smi2uid.get(s)
        if uid is None:
            uid = len(uniq)
            smi2uid[s] = uid
            uniq.append(s)
        uids[i] = uid

    print(f"[DRFP] rows={len(smis)} unique_rxn={len(uniq)} (ratio={len(uniq)/max(1,len(smis)):.4f})")

    fps_u = np.zeros((len(uniq), fp_dim), dtype=np.float32)
    chunk = 4096
    for i in tqdm(range(0, len(uniq), chunk), desc="[DRFP] Encode unique"):
        part = uniq[i:i + chunk]
        fps_u[i:i + len(part)] = drfp_encode(part, n_folded_length=fp_dim)

    fps_u = l2norm_np(fps_u)
    fps = fps_u[uids]
    return torch.tensor(fps, dtype=torch.float32)


def load_or_build_drfp_cache(
    cache_path: str,
    df: pd.DataFrame,
    rxn_col: str,
    fp_dim: int,
    standardize_flag: bool,
    preserve_direction: bool,
    uncharge: bool,
    filter_charged_single_atom: bool,
) -> torch.Tensor:
    if cache_path and os.path.exists(cache_path):
        obj = torch.load(cache_path, map_location="cpu")
        if not torch.is_tensor(obj) or obj.dim() != 2 or obj.shape[0] != len(df) or obj.shape[1] != fp_dim:
            raise ValueError(
                f"Bad DRFP cache: {cache_path}, got "
                f"{None if not torch.is_tensor(obj) else tuple(obj.shape)}, expected ({len(df)}, {fp_dim})"
            )
        return obj.float()

    fp = make_drfp_rowaligned_dedup(
        df=df,
        rxn_col=rxn_col,
        fp_dim=fp_dim,
        standardize_flag=standardize_flag,
        preserve_direction=preserve_direction,
        uncharge=uncharge,
        filter_charged_single_atom=filter_charged_single_atom,
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(fp, cache_path)
        print(f"[Cache] saved DRFP row-aligned -> {cache_path} shape={tuple(fp.shape)} dtype={fp.dtype}")

    return fp

# ----------------------- Reactant Morgan FP (dedup + cache) [NEW] -----------------------
def get_reactant_morgan_fp(rxn_smiles: str, radius: int = 2, nBits: int = 2048) -> np.ndarray:
    """提取反应方程式左侧（所有反应物）的 Morgan 指纹"""
    if not isinstance(rxn_smiles, str) or ">>" not in rxn_smiles:
        return np.zeros((nBits,), dtype=np.float32)
        
    # 切分反应式，只取左侧的反应物
    reactants_smi = rxn_smiles.split(">>")[0]
    
    mol = Chem.MolFromSmiles(reactants_smi)
    if mol is None:
        return np.zeros((nBits,), dtype=np.float32)
        
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def make_reactant_morgan_rowaligned_dedup(
    df: pd.DataFrame,
    rxn_col: str,
    fp_dim: int
) -> torch.Tensor:
    smis = df[rxn_col].fillna("").astype(str).tolist()

    uniq: List[str] = []
    smi2uid: Dict[str, int] = {}
    uids = np.empty((len(smis),), dtype=np.int64)
    for i, s in enumerate(smis):
        uid = smi2uid.get(s)
        if uid is None:
            uid = len(uniq)
            smi2uid[s] = uid
            uniq.append(s)
        uids[i] = uid

    print(f"[Reactant FP] rows={len(smis)} unique_rxn={len(uniq)} (ratio={len(uniq)/max(1,len(smis)):.4f})")

    fps_u = np.zeros((len(uniq), fp_dim), dtype=np.float32)
    # 逐个提取 unique 反应物的指纹
    for i, s in enumerate(tqdm(uniq, desc="[Reactant FP] Encode unique")):
        fps_u[i] = get_reactant_morgan_fp(s, radius=2, nBits=fp_dim)

    # 归一化以对齐 DRFP 的尺度
    fps_u = l2norm_np(fps_u)
    fps = fps_u[uids]
    return torch.tensor(fps, dtype=torch.float32)

def load_or_build_reactant_morgan_cache(
    cache_path: str,
    df: pd.DataFrame,
    rxn_col: str,
    fp_dim: int
) -> torch.Tensor:
    if cache_path and os.path.exists(cache_path):
        obj = torch.load(cache_path, map_location="cpu")
        if not torch.is_tensor(obj) or obj.dim() != 2 or obj.shape[0] != len(df) or obj.shape[1] != fp_dim:
            raise ValueError(
                f"Bad Reactant cache: {cache_path}, got "
                f"{None if not torch.is_tensor(obj) else tuple(obj.shape)}, expected ({len(df)}, {fp_dim})"
            )
        return obj.float()

    fp = make_reactant_morgan_rowaligned_dedup(df=df, rxn_col=rxn_col, fp_dim=fp_dim)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(fp, cache_path)
        print(f"[Cache] saved Reactant FP row-aligned -> {cache_path} shape={tuple(fp.shape)} dtype={fp.dtype}")

    return fp

# ----------------------- Metric Functions -----------------------

# 1. Local (Per-Group) Metrics
def dcg_at_k_binary(labels_sorted: np.ndarray, k: int = 10) -> float:
    k = min(k, len(labels_sorted))
    if k <= 0:
        return 0.0
    rel = labels_sorted[:k].astype(np.float32)
    denom = np.log2(np.arange(2, k + 2, dtype=np.float32))
    return float((rel / denom).sum())

def topk_success(labels_sorted: np.ndarray, k: int) -> float:
    k = min(k, len(labels_sorted))
    if k <= 0:
        return 0.0
    return 1.0 if labels_sorted[:k].sum() > 0 else 0.0

@torch.no_grad()
def evaluate_groups_ranking(
    df: pd.DataFrame,
    scores: np.ndarray,
    group_key: str,
    label_col: str,
) -> Dict[str, float]:
    labels_all = df[label_col].to_numpy(dtype=np.int32)

    if labels_all.min() != labels_all.max():
        auc_micro = float(roc_auc_score(labels_all, scores))
        auprc = float(average_precision_score(labels_all, scores))
    else:
        auc_micro = float("nan")
        auprc = float("nan")

    groups = df.groupby(group_key, sort=False).indices
    keys = list(groups.keys())

    topKs = [1, 3, 5, 10]
    topk_hits = {k: 0.0 for k in topKs}
    dcg10_sum = 0.0
    valid_groups = 0

    for k in keys:
        idxs = np.array(groups[k], dtype=np.int64)
        lab = labels_all[idxs]
        sc = scores[idxs]

        if lab.sum() <= 0:
            continue
        valid_groups += 1

        order = np.argsort(-sc)
        lab_sorted = lab[order]

        for kk in topKs:
            topk_hits[kk] += topk_success(lab_sorted, kk)
        dcg10_sum += dcg_at_k_binary(lab_sorted, 10)

    if valid_groups == 0:
        return {
            "top1": 0.0, "top10": 0.0, "dcg@10": 0.0, 
            "auc_micro": auc_micro, "auprc_micro": auprc,
            "valid_groups": 0, "total_groups": len(keys)
        }

    out = {}
    for kk in topKs:
        out[f"top{kk}"] = float(topk_hits[kk] / valid_groups)
    out["dcg@10"] = float(dcg10_sum / valid_groups)
    out["auc_micro"] = auc_micro
    out["auprc_micro"] = auprc
    out["valid_groups"] = float(valid_groups)
    out["total_groups"] = float(len(keys))
    return out


# 2. Global (Dataset-wide) Metrics
def calculate_local_ef_enzymecage(
    df: pd.DataFrame, 
    scores: np.ndarray, 
    group_key: str, 
    label_col: str, 
    fractions: List[float] = [0.01, 0.02, 0.05, 0.1]
) -> Dict[str, float]:
    df_temp = df.copy()
    df_temp["_score"] = scores
    df_temp["_label"] = df[label_col].astype(int)
    
    ef_lists = {f: [] for f in fractions}
    groups = df_temp.groupby(group_key)
    valid_groups = 0
    
    for grp_name, grp in groups:
        y = grp["_label"].values
        s = grp["_score"].values
        
        N = len(y)
        n_pos = y.sum()
        
        if n_pos == 0:
            continue
        
        sorted_indices = np.argsort(-s) 
        sorted_labels = y[sorted_indices]
        
        for frac in fractions:
            k_raw = int(N * frac)
            k = max(k_raw, 5) 
            k = min(k, N) 
            
            hits = sorted_labels[:k].sum()
            random_expectation = n_pos * frac
            
            if random_expectation == 0:
                ef = 0.0
            else:
                ef = hits / random_expectation
            ef_lists[frac].append(ef)
            
        valid_groups += 1

    metrics = {}
    for frac, values in ef_lists.items():
        metrics[f"local_ef@{frac}"] = float(np.mean(values)) if values else 0.0
        
    metrics["ef_valid_groups"] = float(valid_groups)
    return metrics

# ----------------------- Model [MODIFIED for 3 features] -----------------------
class HybridPairKAN(nn.Module):
    """先用 Linear 稳定降维，后用 KAN 进行复杂推理"""
    def __init__(self, esm_dim: int, drfp_dim: int, react_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.esm_ln = nn.LayerNorm(esm_dim)
        self.drfp_ln = nn.LayerNorm(drfp_dim)
        self.react_ln = nn.LayerNorm(react_dim)

        in_dim = esm_dim + drfp_dim + react_dim
        
        # 1. 瓶颈层：高效、稳定地压缩特征 (5376 -> 256)
        self.bottleneck_dim = hidden
        self.compressor = nn.Sequential(
            nn.Linear(in_dim, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.SiLU(), 
            nn.Dropout(dropout) 
        )
        
        # 2. KAN 层：输入维度降到了 256，极其稳定且不会产生 NaN
        # 这里的 hidden 建议使用 128 或 256
        self.net = KAN([self.bottleneck_dim, hidden, 1])

    def forward(self, esm: torch.Tensor, drfp: torch.Tensor, react: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        e_norm = self.esm_ln(esm)
        d_norm = self.drfp_ln(drfp)
        r_norm = self.react_ln(react)
        
        x = torch.cat([e_norm, d_norm, r_norm], dim=-1)
        x_compressed = self.compressor(x)
        logits = self.net(x_compressed, update_grid=update_grid).squeeze(-1)
        if not self.training:
            return logits, x_compressed # 评估时返回压缩特征以供分析
        
        return logits


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    esm: torch.Tensor,
    fp: torch.Tensor,
    react: torch.Tensor, # [NEW]
    device: str,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    scores = np.zeros((esm.shape[0],), dtype=np.float32)
    for i in range(0, esm.shape[0], batch_size):
        e = esm[i:i + batch_size].to(device)
        f = fp[i:i + batch_size].to(device)
        r = react[i:i + batch_size].to(device) # [NEW]
        
        logit = model(e, f, r)
        prob = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)
        scores[i:i + len(prob)] = prob
    return scores


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--test_csv", type=str, default="")

    ap.add_argument("--train_esm_cache", type=str, required=True)
    ap.add_argument("--val_esm_cache", type=str, required=True)
    ap.add_argument("--test_esm_cache", type=str, default="")

    ap.add_argument("--group_key", type=str, default="CANO_RXN_SMILES")
    ap.add_argument("--rxn_col", type=str, default="CANO_RXN_SMILES")
    ap.add_argument("--label_col", type=str, default="Label")

    ap.add_argument("--fp_dim", type=int, default=2048)
    ap.add_argument("--standardize_smiles", action="store_true")
    ap.add_argument("--preserve_direction", action="store_true")
    ap.add_argument("--uncharge", action="store_true")
    ap.add_argument("--filter_charged_single_atom", action="store_true")

    ap.add_argument("--train_drfp_cache", type=str, default="")
    ap.add_argument("--val_drfp_cache", type=str, default="")
    ap.add_argument("--test_drfp_cache", type=str, default="")

    # [NEW] 反应物指纹参数
    ap.add_argument("--train_reactant_cache", type=str, default="")
    ap.add_argument("--val_reactant_cache", type=str, default="")
    ap.add_argument("--test_reactant_cache", type=str, default="")
    ap.add_argument("--reactant_fp_dim", type=int, default=2048)

    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--pos_weight", type=str, default="auto", help="'auto' or a float like 10.0")

    # early stopping
    ap.add_argument("--select_metric", type=str, default="top1", choices=["dcg@10", "top1","top10", "auc_micro", "auprc_micro"])
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=0.003)
    ap.add_argument("--warmup_epochs", type=int, default=2)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_best_path", type=str, default="checkpoints/binarycls_best_val.pt")
    ap.add_argument("--kan_reg_weight", type=float, default=1e-4, help="Weight for KAN regularization loss")
    args = ap.parse_args()


    import os
    base_name, ext = os.path.splitext(args.save_best_path)
    args.save_best_path = f"{base_name}_seed{args.seed}{ext}"
    
    if not HAS_DRFP:
        raise SystemExit("ERROR: drfp not installed. `pip install drfp`")
    if args.standardize_smiles and not HAS_RDKIT:
        raise SystemExit("ERROR: RDKit not available but --standardize_smiles was set.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"seed: {args.seed}")

    # load dfs - added low_memory=False to suppress warnings
    df_tr = pd.read_csv(args.train_csv, low_memory=False)
    df_va = pd.read_csv(args.val_csv, low_memory=False)
    df_te = pd.read_csv(args.test_csv, low_memory=False) if args.test_csv else None

    for df, name in [(df_tr, "train"), (df_va, "val")] + ([(df_te, "test")] if df_te is not None else []):
        if df is None:
            continue
        for c in [args.rxn_col, "sequence", args.label_col, args.group_key]:
            if c not in df.columns:
                raise ValueError(f"[{name}] missing column {c}, have: {list(df.columns)[:40]}")
        df[args.label_col] = df[args.label_col].astype(int)

    # load esm
    esm_tr = load_row_aligned_cache(args.train_esm_cache, expected_rows=len(df_tr))
    esm_va = load_row_aligned_cache(args.val_esm_cache, expected_rows=len(df_va))
    esm_te = None
    if df_te is not None:
        if not args.test_esm_cache:
            raise ValueError("Provided --test_csv but missing --test_esm_cache")
        esm_te = load_row_aligned_cache(args.test_esm_cache, expected_rows=len(df_te))

    # drfp
    fp_tr = load_or_build_drfp_cache(
        cache_path=args.train_drfp_cache,
        df=df_tr,
        rxn_col=args.rxn_col,
        fp_dim=args.fp_dim,
        standardize_flag=args.standardize_smiles,
        preserve_direction=args.preserve_direction,
        uncharge=args.uncharge,
        filter_charged_single_atom=args.filter_charged_single_atom,
    )
    fp_va = load_or_build_drfp_cache(
        cache_path=args.val_drfp_cache,
        df=df_va,
        rxn_col=args.rxn_col,
        fp_dim=args.fp_dim,
        standardize_flag=args.standardize_smiles,
        preserve_direction=args.preserve_direction,
        uncharge=args.uncharge,
        filter_charged_single_atom=args.filter_charged_single_atom,
    )
    fp_te = None
    if df_te is not None:
        fp_te = load_or_build_drfp_cache(
            cache_path=args.test_drfp_cache,
            df=df_te,
            rxn_col=args.rxn_col,
            fp_dim=args.fp_dim,
            standardize_flag=args.standardize_smiles,
            preserve_direction=args.preserve_direction,
            uncharge=args.uncharge,
            filter_charged_single_atom=args.filter_charged_single_atom,
        )

    # load Reactant Morgan FP [NEW]
    react_tr = load_or_build_reactant_morgan_cache(
        args.train_reactant_cache, df_tr, args.rxn_col, args.reactant_fp_dim
    )
    react_va = load_or_build_reactant_morgan_cache(
        args.val_reactant_cache, df_va, args.rxn_col, args.reactant_fp_dim
    )
    react_te = None
    if df_te is not None:
        react_te = load_or_build_reactant_morgan_cache(
            args.test_reactant_cache, df_te, args.rxn_col, args.reactant_fp_dim
        )

    # pos_weight
    y_tr = df_tr[args.label_col].to_numpy(dtype=np.int64)
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    if args.pos_weight == "auto":
        pos_weight_val = (n_neg / max(1, n_pos))
    else:
        pos_weight_val = float(args.pos_weight)
    print(f"[Diag] train pos={n_pos} neg={n_neg} pos_weight={pos_weight_val:.4f}")

    device = args.device
    
    # 实例化更新后的三特征主模型
    model = HybridPairKAN(
        esm_dim=esm_tr.shape[1], 
        drfp_dim=fp_tr.shape[1], 
        react_dim=react_tr.shape[1], 
        hidden=args.hidden, 
        dropout=args.dropout
    ).to(device)

    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(os.path.dirname(args.save_best_path) or ".", exist_ok=True)

    best_val = -1e9
    best_epoch = -1
    bad_epochs = 0

    # pre-make tensors for labels
    ytr_t = torch.tensor(df_tr[args.label_col].to_numpy(np.float32))

    for ep in range(args.epochs):
        model.train()
        # shuffle indices
        perm = np.random.permutation(len(df_tr))
        losses = []

        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            e_b = esm_tr[b].to(device)
            f_b = fp_tr[b].to(device)
            r_b = react_tr[b].to(device) # 获取反应物特征
            y_b = ytr_t[b].to(device)

            # update_grid_flag = (i == 0)
            logit = model(e_b, f_b, r_b)
            # logit = model(e_b, f_b, r_b) # 传入模型
            loss = crit(logit, y_b)
            loss = loss + args.kan_reg_weight * model.net.regularization_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        # val eval
        with torch.no_grad():
            val_scores = predict_scores(
                model, esm_va, fp_va, react_va, device=device, batch_size=8192
            )
            val_metrics = evaluate_groups_ranking(df_va, val_scores, group_key=args.group_key, label_col=args.label_col)

        sel = float(val_metrics[args.select_metric])
        improved = (not np.isnan(sel)) and (sel > (best_val + args.min_delta))

        print(
            f"\n[Ep {ep+1:03d}] loss={float(np.mean(losses)):.4f} | "
            f"val top1={val_metrics['top1']:.4f} top10={val_metrics['top10']:.4f} "
            f"dcg@10={val_metrics['dcg@10']:.4f} | "
            f"sel({args.select_metric})={sel:.4f} best={best_val:.4f} improved={improved}"
        )

        if improved:
            best_val = sel
            best_epoch = ep + 1
            bad_epochs = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "model": model.state_dict(),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                },
                args.save_best_path,
            )
        else:
            if (ep + 1) >= args.warmup_epochs:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"[EarlyStop] triggered at epoch {ep+1}")
                    break

    print(f"\n[Best] epoch={best_epoch} by val {args.select_metric}={best_val:.4f}")

    # TEST Evaluation
    if df_te is not None:
        ckpt = torch.load(args.save_best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()

        with torch.no_grad():
            test_scores = predict_scores(
                model, esm_te, fp_te, react_te, device=device, batch_size=8192
            )
            
            # 1. Local (Per-Reaction) Metrics -> for DCG and Top-k
            group_metrics = evaluate_groups_ranking(df_te, test_scores, group_key=args.group_key, label_col=args.label_col)
            
            # 2. Local Metrics matching EnzymeCAGE logic
            enzymecage_ef_metrics = calculate_local_ef_enzymecage(
                df_te, 
                test_scores, 
                group_key=args.group_key, 
                label_col=args.label_col,
                fractions=[0.01, 0.02, 0.05, 0.1]
            )

        print("\n" + "="*40)
        print("          TEST EVALUATION")
        print("="*40)
        print(f"Top-1 Success Rate (Local): {group_metrics['top1']:.4f}")
        print(f"Top-10 Success Rate (Local):{group_metrics['top10']:.4f}")
        print(f"DCG@10 (Local):             {group_metrics['dcg@10']:.4f}")
        print(f"AUC Micro:                  {group_metrics['auc_micro']:.4f}")
        print("-" * 40)
        # 打印新的 EF
        print(f"EF@1% (EnzymeCAGE Logic):   {enzymecage_ef_metrics['local_ef@0.01']:.4f}")
        print(f"EF@2% (EnzymeCAGE Logic):   {enzymecage_ef_metrics['local_ef@0.02']:.4f}")
        print(f"EF@5% (EnzymeCAGE Logic):   {enzymecage_ef_metrics['local_ef@0.05']:.4f}")
        print(f"Valid Groups for EF:        {int(enzymecage_ef_metrics['ef_valid_groups'])}")
        print("="*40)

if __name__ == "__main__":
    main()