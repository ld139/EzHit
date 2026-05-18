# EZHit

**EZHit** is a lightweight enzyme–reaction retrieval framework for predicting potential catalytic compatibility between enzyme sequences and biochemical reactions. Given an enzyme amino-acid sequence and a reaction SMILES, EZHit estimates whether the enzyme is likely to catalyze the reaction and provides uncertainty-aware outputs for candidate prioritization.

EZHit is designed for three common use cases:

1. **Single-pair prediction** through a HuggingFace Space interface.
2. **Batch enzyme–reaction screening** for large candidate libraries.
3. **User-side fine-tuning** on custom enzyme–reaction datasets through Google Colab.

---

## Overview

EZHit predicts enzyme–reaction compatibility using a neural retrieval model trained on enzyme–reaction pairs. The model combines enzyme and reaction representations and outputs a compatibility probability. For more reliable screening, EZHit can also report uncertainty-related scores, including ensemble disagreement and Mahalanobis distance in the learned latent space.

The Mahalanobis distance is used as a distribution-aware reliability estimate. A lower distance suggests that the enzyme–reaction pair is closer to the learned training distribution, while a higher distance may indicate a potentially out-of-distribution candidate.

---

## Online demo

A web demo is available through HuggingFace Space:

**HuggingFace Space:**  
`TODO: add HuggingFace Space URL here`

The Space supports:

- enzyme–reaction pair prediction
- ensemble probability output
- ensemble uncertainty estimation
- Mahalanobis-distance-based reliability assessment
- reaction visualization

Users need to provide:

| Input | Description |
|---|---|
| Enzyme protein sequence | Amino-acid sequence of the enzyme |
| Reaction SMILES | Reaction in `reactants>>products` format |
| Prediction model | General or enzyme-family-specific model |
| Probability threshold | Threshold for compatibility prediction |
| Mahalanobis threshold | Threshold for distribution-aware filtering |

---

## Google Colab fine-tuning

Users can fine-tune EZHit on their own enzyme–reaction datasets using the provided Colab notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](TODO_ADD_COLAB_LINK_HERE)

The Colab notebook allows users to:

- upload a custom enzyme–reaction dataset
- upload or download a pretrained EZHit checkpoint
- automatically build required feature caches
- fine-tune the model on user-provided data
- evaluate performance on validation and test sets
- export a fine-tuned checkpoint
- export `train_distribution_stat.pt` for Mahalanobis-distance inference

Recommended repository location:

```text
colab/
└── EZHit_FineTune_Colab.ipynb
```

After uploading the notebook to GitHub, replace the Colab badge link with:

```text
https://colab.research.google.com/github/<GitHub-username>/<repo-name>/blob/main/colab/EZHit_FineTune_Colab.ipynb
```

---

## Required input format for fine-tuning

The input dataset should be a CSV file with the following columns:

| Column | Description |
|---|---|
| `protein_sequence` | Enzyme amino-acid sequence |
| `CANO_RXN_SMILES` | Reaction SMILES in `reactants>>products` format |
| `Label` | Binary label. `1` for compatible enzyme–reaction pairs and `0` for negative pairs |

Example:

```csv
protein_sequence,CANO_RXN_SMILES,Label
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG,CCO>>CC=O,1
MKKLLPTAAAGLLLLAAQPAMA,CCO>>CC=O,0
```

An optional `split` column can be provided:

| Column | Description |
|---|---|
| `split` | Dataset split. Allowed values: `train`, `val`, `test` |

Example:

```csv
protein_sequence,CANO_RXN_SMILES,Label,split
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG,CCO>>CC=O,1,train
MKKLLPTAAAGLLLLAAQPAMA,CCO>>CC=O,0,val
```

If no `split` column is provided, the Colab notebook will automatically generate train, validation, and test splits.

---

## Fine-tuning outputs

The Colab notebook produces the following files:

| Output file | Description |
|---|---|
| `ezhit_finetuned_seed42.pt` | Fine-tuned EZHit checkpoint |
| `train_distribution_stat.pt` | Latent-space training distribution statistics for Mahalanobis-distance inference |
| `val_predictions.csv` | Validation-set prediction results |
| `test_predictions.csv` | Test-set prediction results |
| `training_log.csv` | Training and validation log |

The two most important files for deployment are:

```text
ezhit_finetuned_seed42.pt
train_distribution_stat.pt
```

Upload these files to the HuggingFace Space directory if you want to run customized inference with the fine-tuned model and Mahalanobis-distance estimation.

---

## Installation

### Option 1. Use the Colab notebook

For most users, we recommend using the Colab notebook because it provides a ready-to-run environment.

### Option 2. Local installation

Clone this repository:

```bash
git clone <TODO_ADD_GITHUB_REPO_URL>
cd EZHit
```

Create a conda environment:

```bash
conda create -n ezhit python=3.10 -y
conda activate ezhit
```

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn tqdm gradio transformers sentencepiece
pip install rdkit-pypi drfp pyarrow
pip install pykan
```

Depending on your system, RDKit installation through conda may be more stable:

```bash
conda install -c conda-forge rdkit -y
```

---

## Pretrained checkpoints and data

Pretrained checkpoints and example datasets will be provided at:

| Resource | Link |
|---|---|
| Pretrained EZHit checkpoints | `TODO: add checkpoint link` |
| Example fine-tuning dataset | `TODO: add example dataset link` |
| Full training data | `TODO: add data link` |
| HuggingFace Space | `TODO: add Space link` |
| Colab notebook | `TODO: add Colab link` |

Recommended checkpoint directory:

```text
checkpoints/
├── binarycls_best_val_seed40.pt
├── binarycls_best_val_seed41.pt
├── binarycls_best_val_seed42.pt
├── binarycls_best_val_seed43.pt
└── binarycls_best_val_seed44.pt
```

---

## Repository structure

A suggested repository structure is:

```text
EZHit/
├── README.md
├── app.py
├── requirements.txt
├── train_finetune.py
├── inference.py
├── massive_screening.py
├── colab/
│   └── EZHit_FineTune_Colab.ipynb
├── examples/
│   └── example_dataset.csv
├── checkpoints/
│   └── README.md
└── scripts/
    └── prepare_features.py
```

Large checkpoint files are not recommended to be stored directly in the GitHub repository. We recommend hosting them on HuggingFace, Zenodo, or GitHub Releases, and providing download links in this README.

---

## Quick start: single-pair prediction

The easiest way to use EZHit is through the HuggingFace Space:

```text
TODO: add HuggingFace Space URL here
```

Users can input:

```text
Protein sequence:
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG

Reaction SMILES:
CCO>>CC=O
```

EZHit will return:

- mean match probability
- ensemble uncertainty
- Mahalanobis distance
- compatibility diagnosis
- reaction visualization

---

## Quick start: fine-tuning in Colab

1. Open the Colab notebook:

```text
TODO: add Colab URL here
```

2. Select GPU runtime:

```text
Runtime → Change runtime type → T4 GPU
```

3. Upload your files:

```text
dataset.csv
pretrained_checkpoint.pt
```

4. Run all cells.

5. Download the output files:

```text
ezhit_finetuned_seed42.pt
train_distribution_stat.pt
test_predictions.csv
```

6. Optionally upload the fine-tuned checkpoint and `train_distribution_stat.pt` to the HuggingFace Space for customized inference.

---

## Command-line fine-tuning

For local fine-tuning, prepare train, validation, and optionally test CSV files.

Example command:

```bash
python train_finetune.py \
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

The script will save the best checkpoint according to the selected validation metric.

---

## Batch screening

EZHit can also be used for large-scale enzyme–reaction screening. The batch screening workflow takes candidate enzyme embeddings and reaction databases as input, performs matrix-style enzyme–reaction prediction, and writes confident hits to disk.

Example command:

```bash
python massive_screening.py \
    --enzyme_parquet data/candidate_enzymes.parquet \
    --esm_cache cache/candidate_protein_embeddings.pt \
    --rhea_csv data/rhea_database_unique.csv \
    --maha_stats train_distribution_stat.pt \
    --ckpt_dir checkpoints/ \
    --ckpt_pattern "binarycls_best_val_seed*.pt" \
    --prob_threshold 0.50 \
    --maha_threshold 20.0 \
    --protein_chunk 256 \
    --rhea_chunk 1024 \
    --out_csv results/massive_hits_mahalanobis.csv
```

The output file contains:

| Column | Description |
|---|---|
| `protein_id` | Candidate enzyme identifier |
| `rhea_id` | Reaction identifier |
| `Predictive_Prob` | Predicted enzyme–reaction compatibility probability |
| `Mahalanobis_Dist` | Latent-space Mahalanobis distance |

---

## Interpretation of prediction outputs

### Probability

The predicted probability estimates the compatibility between the input enzyme and reaction. A higher value indicates stronger predicted enzyme–reaction matching.

### Ensemble uncertainty

Ensemble uncertainty reflects disagreement among models. Higher uncertainty suggests that different models provide less consistent predictions.

### Mahalanobis distance

Mahalanobis distance measures how far the input pair is from the learned training distribution in the model latent space. A lower value generally suggests that the prediction is closer to the domain learned during training.

A typical interpretation is:

| Probability | Mahalanobis distance | Interpretation |
|---|---|---|
| High | Low | High-priority candidate |
| High | High | Potentially interesting but less reliable or out-of-distribution |
| Low | Low | In-distribution but predicted as incompatible |
| Low | High | Low-priority candidate |

The exact thresholds should be adjusted based on the dataset, model version, and validation results.

---

## Notes on Mahalanobis-distance statistics

The file `train_distribution_stat.pt` should be generated from the same model architecture and latent dimension as the checkpoint used for inference.

The expected file contains:

```python
{
    "mean": positive_class_latent_mean,
    "inv_cov": inverse_covariance_matrix
}
```

If the fine-tuned model uses a different hidden dimension, the corresponding `train_distribution_stat.pt` must be regenerated. Otherwise, the Mahalanobis-distance calculation will not be valid.

For very small fine-tuning datasets, the covariance estimate may be unstable. In such cases, probability and ensemble uncertainty should be interpreted together with caution.

---

## Requirements

Core dependencies:

```text
python>=3.10
torch
numpy
pandas
scikit-learn
tqdm
transformers
sentencepiece
rdkit
drfp
gradio
pyarrow
pykan
```

A minimal `requirements.txt` can be:

```text
torch
numpy
pandas
scikit-learn
tqdm
transformers
sentencepiece
rdkit-pypi
drfp
gradio
pyarrow
pykan
```

For Colab, the notebook installs the required packages automatically.

---

## Common issues

### 1. RDKit installation fails

Try installing RDKit through conda:

```bash
conda install -c conda-forge rdkit -y
```

### 2. CUDA out-of-memory error

Reduce the batch size:

```bash
--batch_size 128
```

For large-scale screening, reduce chunk sizes:

```bash
--protein_chunk 64
--rhea_chunk 256
```

### 3. Mahalanobis dimension mismatch

Make sure the checkpoint and `train_distribution_stat.pt` were generated using the same model hidden dimension.

For example, if the model hidden dimension is 512, then:

```text
mean shape: [512]
inv_cov shape: [512, 512]
```

### 4. Invalid reaction SMILES

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
@article{TODO_EZHit,
  title   = {TODO: add paper title},
  author  = {TODO: add authors},
  journal = {TODO: add journal or preprint server},
  year    = {TODO: add year},
  doi     = {TODO: add DOI}
}
```

---

## License

This project is released under the MIT License.

See `LICENSE` for details.

---

## Contact

For questions, please contact:

```text
TODO: add contact email
```