# EzHit

**EzHit** is a lightweight enzyme-reaction retrieval framework for predicting potential catalytic compatibility between enzyme sequences and biochemical reactions.

Given an enzyme amino-acid sequence and a reaction SMILES, EzHit estimates whether the enzyme is likely to catalyze the reaction. EzHit supports online prediction, custom fine-tuning, local training, local inference, and uncertainty-aware inference using ensemble prediction and Mahalanobis-distance-based distribution assessment.

---

## Online demo

EzHit can be used directly through the HuggingFace Space:

[Try EzHit on HuggingFace Space](https://huggingface.co/spaces/deanluo/Enzyme-Catalysis-Predictor)

The web interface allows users to submit an enzyme sequence and a reaction SMILES and obtain prediction results through an interactive interface.

---

## Colab fine-tuning

Users can fine-tune EZHit on their own enzyme-reaction datasets using the provided Google Colab notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ld139/EzHit/blob/main/colab/EZHit_FineTune_Colab.ipynb)

The Colab notebook provides a ready-to-run workflow for:

- uploading a custom enzyme-reaction dataset
- uploading a pretrained EZHit checkpoint
- generating required feature caches
- fine-tuning EZHit on user-provided data
- evaluating the fine-tuned model
- exporting the fine-tuned checkpoint
- exporting `train_distribution_stat.pt` for Mahalanobis-distance inference

After fine-tuning, the exported checkpoint and `train_distribution_stat.pt` can be used for customized prediction in the HuggingFace Space or local inference scripts.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/ld139/EzHit.git
cd EzHit
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Checkpoints and data

| Resource | Link |
|---|---|
| HuggingFace Space | [Enzyme-Catalysis-Predictor](https://huggingface.co/spaces/deanluo/Enzyme-Catalysis-Predictor) |
| Colab notebook | [EZHit_FineTune_Colab.ipynb](https://colab.research.google.com/github/ld139/EzHit/blob/main/colab/EZHit_FineTune_Colab.ipynb) |
| Pretrained EZHit checkpoints | [deanluo/EzHit](https://huggingface.co/deanluo/EzHit) |
| Large-scale screening results | [deanluo/EzHit-large-scale-screening](https://huggingface.co/datasets/deanluo/EzHit-large-scale-screening) |
| Full training data | To be added |

To download released checkpoints:

```bash
pip install -U huggingface_hub

hf download deanluo/EzHit \
    --include "checkpoints/*.pt" \
    --local-dir .
```

The checkpoint files will be placed under:

```text
checkpoints/
```

---

## Local training from scratch

EZHit can be trained locally using `train_bn_kan.py`.

Example command:

```bash
python train_bn_kan.py \
    --train_csv data/train.csv \
    --val_csv data/valid.csv \
    --test_csv data/Enzyme-405.csv \
    --train_esm_cache caches/train_prott5_bf16.pt \
    --val_esm_cache caches/valid_prott5_bf16.pt \
    --test_esm_cache caches/Enzyme-405_bf16.pt \
    --train_drfp_cache caches/train_drfp.pt \
    --val_drfp_cache caches/valid_drfp.pt \
    --test_drfp_cache caches/Enzyme-405_drfp.pt \
    --train_reactant_cache caches/train_reactant_2048.pt \
    --val_reactant_cache caches/valid_reactant_2048.pt \
    --test_reactant_cache caches/Enzyme-405_reactant_2048.pt \
    --group_key CANO_RXN_SMILES \
    --rxn_col CANO_RXN_SMILES \
    --label_col Label \
    --standardize_smiles \
    --uncharge \
    --filter_charged_single_atom \
    --fp_dim 2048 \
    --reactant_fp_dim 2048 \
    --hidden 512 \
    --dropout 0.4 \
    --epochs 30 \
    --batch_size 2048 \
    --lr 1e-4 \
    --patience 5 \
    --select_metric top10 \
    --pos_weight auto \
    --seed 44
```

---

## Local fine-tuning

EZHit can also be fine-tuned locally using `finetune_bn.py`.

Example command:

```bash
python finetune_bn.py \
    --pretrained_ckpt checkpoints/binarycls_best_val_seed40.pt \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --test_csv data/test.csv \
    --train_esm_cache caches/train_protein.pt \
    --val_esm_cache caches/val_protein.pt \
    --test_esm_cache caches/test_protein.pt \
    --train_drfp_cache caches/train_drfp.pt \
    --val_drfp_cache caches/val_drfp.pt \
    --test_drfp_cache caches/test_drfp.pt \
    --train_reactant_cache caches/train_reactant_2048.pt \
    --val_reactant_cache caches/val_reactant_2048.pt \
    --test_reactant_cache caches/test_reactant_2048.pt \
    --rxn_col CANO_RXN_SMILES \
    --group_key CANO_RXN_SMILES \
    --label_col Label \
    --epochs 15 \
    --batch_size 512 \
    --lr 1e-4 \
    --pos_weight auto \
    --save_best_path checkpoints/ezhit_finetuned.pt
```

---

## Local inference

Local inference is provided through `predict.py`.

The script can run either:

1. single-pair prediction from command-line inputs, or
2. batch prediction from a CSV file.

It automatically computes:

- ProtT5 protein embeddings
- DRFP reaction fingerprints
- reactant Morgan fingerprints
- ensemble prediction probability
- ensemble uncertainty
- optional Mahalanobis distance if `train_distribution_stat.pt` is provided

### Single-pair prediction

```bash
python predict.py \
    --protein_sequence "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG" \
    --rxn_smiles "CCO>>CC=O" \
    --model_group general \
    --seeds 40 41 42 43 44 \
    --standardize_smiles \
    --uncharge \
    --filter_charged_single_atom \
    --output_csv results/single_prediction.csv
```

### Batch prediction from CSV

Prepare a CSV file:

```csv
protein_sequence,CANO_RXN_SMILES
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG,CCO>>CC=O
MKKLLPTAAAGLLLLAAQPAMA,CCO>>CC=O
```

Run prediction:

```bash
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
```

### Use local checkpoints

By default, `predict.py` downloads checkpoints from [deanluo/EzHit](https://huggingface.co/deanluo/EzHit). To use local checkpoints:

```bash
python predict.py \
    --input_csv examples/predict_demo.csv \
    --ckpt_paths checkpoints/binarycls_best_val_seed40.pt checkpoints/binarycls_best_val_seed41.pt checkpoints/binarycls_best_val_seed42.pt checkpoints/binarycls_best_val_seed43.pt checkpoints/binarycls_best_val_seed44.pt \
    --standardize_smiles \
    --uncharge \
    --filter_charged_single_atom \
    --output_csv results/predict_demo_output.csv
```

### Mahalanobis-distance inference

If a matching `train_distribution_stat.pt` file is available, pass it through `--maha_stat_path`:

```bash
python predict.py \
    --input_csv examples/predict_demo.csv \
    --ckpt_paths checkpoints/binarycls_best_val_seed40.pt checkpoints/binarycls_best_val_seed41.pt checkpoints/binarycls_best_val_seed42.pt checkpoints/binarycls_best_val_seed43.pt checkpoints/binarycls_best_val_seed44.pt \
    --maha_stat_path train_distribution_stat.pt \
    --standardize_smiles \
    --uncharge \
    --filter_charged_single_atom \
    --output_csv results/predict_demo_with_maha.csv
```

The output CSV contains:

| Column | Description |
|---|---|
| `EZHit_probability` | Mean ensemble match probability |
| `EZHit_probability_std` | Standard deviation across ensemble checkpoints |
| `EZHit_ensemble_MI` | Mutual-information-based ensemble uncertainty |
| `Mahalanobis_Dist` | Optional latent-space Mahalanobis distance |

---

## Output files from fine-tuning

The Colab fine-tuning workflow exports the following key files:

| File | Description |
|---|---|
| `ezhit_finetuned_seed42.pt` | Fine-tuned EZHit model checkpoint |
| `train_distribution_stat.pt` | Training-distribution statistics for Mahalanobis-distance inference |
| `val_predictions.csv` | Prediction results on the validation set |
| `test_predictions.csv` | Prediction results on the test set |

The two most important files for customized inference are:

```text
ezhit_finetuned_seed42.pt
train_distribution_stat.pt
```

The checkpoint stores the fine-tuned model weights. The `train_distribution_stat.pt` file stores latent-space statistics used for Mahalanobis-distance calculation.

---

## Prediction outputs

EZHit reports prediction results that may include:

| Output | Description |
|---|---|
| Match probability | Predicted enzyme-reaction compatibility score |
| Ensemble uncertainty | Model-disagreement-based uncertainty estimate |
| Mahalanobis distance | Latent-space distance from the training distribution |

A typical interpretation is:

| Probability | Mahalanobis distance | Interpretation |
|---|---|---|
| High | Low | High-priority candidate |
| High | High | Potentially useful but less reliable or out-of-distribution |
| Low | Low | In-distribution but predicted as incompatible |
| Low | High | Low-priority candidate |

The thresholds should be adjusted based on the dataset, model version, and validation results.

---

## Citation

If you use EZHit in your research, please cite:

```bibtex
@article{ezhit,
  title   = {Accurate and large-scale enzyme-reaction retrieval with sequence information},
  author  = {Ding Luo, Binju Wang},
  journal = {Under Review},
  year    = {2026},
}
```

---

## License

This project is released under the MIT License.

---

## Contact

For questions or issues, please contact the developers through GitHub Issues.
