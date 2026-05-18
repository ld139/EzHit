# EZHit

**EZHit** is a lightweight enzyme-reaction retrieval framework for predicting potential catalytic compatibility between enzyme sequences and biochemical reactions.

Given an enzyme amino-acid sequence and a reaction SMILES, EZHit estimates whether the enzyme is likely to catalyze the reaction. EZHit supports online prediction, custom fine-tuning, and uncertainty-aware inference using ensemble prediction and Mahalanobis-distance-based distribution assessment.

---

## Online demo

EZHit can be used directly through the HuggingFace Space:

[Try EZHit on HuggingFace Space](https://huggingface.co/spaces/deanluo/Enzyme-Catalysis-Predictor)

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

## Input data format

The fine-tuning dataset should be provided as a CSV file.

Required columns:

| Column | Description |
|---|---|
| `protein_sequence` | Enzyme amino-acid sequence |
| `CANO_RXN_SMILES` | Reaction SMILES in `reactants>>products` format |
| `Label` | Binary label. `1` for compatible enzyme-reaction pairs and `0` for negative pairs |

Example:

```csv
protein_sequence,CANO_RXN_SMILES,Label
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG,CCO>>CC=O,1
MKKLLPTAAAGLLLLAAQPAMA,CCO>>CC=O,0
```

An optional `split` column can also be provided:

| Column | Description |
|---|---|
| `split` | Dataset split. Allowed values: `train`, `val`, and `test` |

Example:

```csv
protein_sequence,CANO_RXN_SMILES,Label,split
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG,CCO>>CC=O,1,train
MKKLLPTAAAGLLLLAAQPAMA,CCO>>CC=O,0,val
```

If no `split` column is provided, the Colab notebook will automatically generate train, validation, and test splits.

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
| Pretrained EZHit checkpoints | [EZHit_Checkpoints](https://huggingface.co/deanluo/EzHit) |
| Example fine-tuning dataset | To be added |
| Full training data | To be added |

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
    --train_esm_cache cache/train_protein.pt \
    --val_esm_cache cache/val_protein.pt \
    --test_esm_cache cache/test_protein.pt \
    --rxn_col CANO_RXN_SMILES \
    --group_key CANO_RXN_SMILES \
    --label_col Label \
    --epochs 15 \
    --batch_size 512 \
    --lr 1e-4 \
    --save_best_path checkpoints/ezhit_finetuned.pt
```

The script loads a pretrained checkpoint and continues training on the user-provided dataset.

---

## Required files for local fine-tuning

For local fine-tuning, users need to prepare:

| File | Description |
|---|---|
| Training CSV | Training enzyme-reaction pairs |
| Validation CSV | Validation enzyme-reaction pairs |
| Test CSV | Optional test enzyme-reaction pairs |
| Protein embedding cache | Row-aligned protein representation cache |
| Pretrained checkpoint | EZHit pretrained model checkpoint |

The Colab notebook provides a simpler workflow because it can generate the required feature files automatically.

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

## Mahalanobis-distance inference

Mahalanobis-distance inference requires a `train_distribution_stat.pt` file generated from the same model architecture as the checkpoint used for prediction.

The expected file contains:

```python
{
    "mean": positive_class_latent_mean,
    "inv_cov": inverse_covariance_matrix
}
```

The latent dimension of `train_distribution_stat.pt` must match the hidden dimension of the model checkpoint. If the model is fine-tuned with a different hidden dimension, the training-distribution statistics must be regenerated.

For very small fine-tuning datasets, Mahalanobis-distance estimates may be unstable and should be interpreted with caution.

---

## Reaction SMILES format

Reaction SMILES should follow the format:

```text
reactants>>products
```

Example:

```text
CCO>>CC=O
```

Rows with invalid reaction SMILES may fail during feature generation.

---


## Citation

If you use EZHit in your research, please cite:

```bibtex
@article{ezhit,
  title   = {Accurate and large-scale enzyme-reaction retrieval with sequence information},
  author  = {To be added},
  journal = {To be added},
  year    = {To be added},
  doi     = {To be added}
}
```

---

## License

This project is released under the MIT License.

---

## Contact

For questions or issues, please contact the developers through GitHub Issues.
