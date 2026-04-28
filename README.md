# BERT Symptom NER

This repository contains a symptom-focused biomedical NER workflow built around BERT-style token classification models. The project generates synthetic symptom data, converts it to WordPiece-aligned labels, trains BioBERT variants, evaluates them on standard and behavioural sets, and tracks error categories to guide the next iteration.

## Current Status

The repository now has multiple versioned tracks:

- `v01` is the latest validated baseline and the recommended starting point.
- `v02` is the active experimental branch. It builds on `v01`, adds new training/debugging work, and is still under investigation because some reported metrics look suspiciously perfect.
- `v00` is legacy history that documents the original label-design failure.

If you are starting fresh, begin with `v01`. Use `v02` only if you are specifically working on the current experiments or debugging pass.

## What The Project Does

The end-to-end workflow is:

1. Build a symptom dictionary from BioPortal-derived ontology data.
2. Generate synthetic symptom sentences with positive and negative polarity labels.
3. Convert word-level annotations into model-specific WordPiece token labels.
4. Split the dataset and publish it to the Hugging Face Hub.
5. Train token classification models, mainly BioBERT and DistilBERT.
6. Evaluate on validation/test data plus a behavioural challenge set.
7. Categorize failures to decide whether the next improvement should target data, training, or inference.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```bash
# Required for Hub access
HF_TOKEN=your_huggingface_token
HF_USERNAME=your_huggingface_username
VERSION=v01

# Optional explicit overrides
HUGGINGFACE_REPO_ID=your_username/symptoms_ner_v01
HUGGINGFACE_REPO_ID_BIOBERT=your_username/symptoms_ner_v01_biobert

# Optional tracking
USE_WANDB=true
WANDB_API_KEY=your_wandb_key

# Optional ontology work
BIO_PORTAL_API_KEY=your_bioportal_key

# Optional GCP / GCS
SAVE_TO_GCS=true
BUCKET_NAME=ner_training_data_results
```

`config.py` derives the default Hugging Face dataset repo IDs from `HF_USERNAME` and `VERSION`:

- `{HF_USERNAME}/symptoms_ner_{VERSION}`
- `{HF_USERNAME}/symptoms_ner_{VERSION}_biobert`

It also defines a shared model repo name via `HUGGINGFACE_MODEL_REPO_ID`.

## Recommended Entry Points

For the stable baseline workflow, start here:

- `v01/7_trainer_gcp.py`: main validated training script.
- `hyperparam_sets.py`: selects model, dataset repo, and training sweep values.
- `v01/inference_test.ipynb`: baseline inference notebook.
- `v01/9_categorize_errors_in_behavioural_set.ipynb`: behavioural-set review for the validated workflow.
- `inference/v01/inference_utils.py`: reusable token-to-word-to-span inference helpers.

For experimental `v02` work:

- `v02/7_trainer_gcp.py`: experimental trainer for the current branch.
- `v02/inference_test.ipynb`: checkpoint inspection and inference checks.
- `v02/9_categorize_errors_in_behavoural_set.ipynb`: updated behavioural analysis pass.
- `v02/download_model_from_gcs.py`: downloads a trained run locally for notebook-based inspection.

Legacy:

- `7_trainer.py`: older local script with a hardcoded `v00` dataset reference.

## Version Guide

### `v00`

Early dataset generation and training experiments. This version used symptom-specific labels, created an unmanageably sparse label space, and is now mainly useful as historical context for why the project changed direction.

### `v01`

The first stable collapsed-label pipeline and the recommended default:

- synthetic data generation notebooks
- WordPiece tokenization notebooks for DistilBERT and BioBERT
- train/validation/test split generation
- behavioural evaluation set
- comparison and inference notebooks
- `v01/7_trainer_gcp.py`, which freezes the encoder and trains only the classification head

### `v02`

The active experimental branch focused on improving behaviour on the challenge set:

- publishes a `v02` BioBERT dataset to the Hub
- reuses the collapsed label mappings
- trains with `v02/7_trainer_gcp.py`
- supports pushing each run to a dedicated Hugging Face branch
- uploads run artifacts to GCS
- includes notebooks for inference validation, error categorization, and debugging suspiciously high metrics

One important detail: `v02` is still partly layered on top of `v01` assets. Some notebooks reuse `v01` label files, the behavioural set, and the inference helpers under `inference/v01`. `v02` should be treated as experimental until the perfect-metrics debugging work is resolved.

## Training

### Stable baseline path

Use the validated `v01` trainer by default:

```bash
python v01/7_trainer_gcp.py
```

### Experimental path

Use the `v02` trainer only for the active experiments:

```bash
python v02/7_trainer_gcp.py
```

Both versioned trainers:

1. loads the dataset from the Hugging Face Hub using the repo in `hyperparam_sets.py`
2. downloads `id2label.json` and `label2id.json`
3. initializes the tokenizer and token-classification model
4. trains with the selected hyperparameter slot
5. evaluates on validation and test splits
6. saves metrics, plots, and a summary under the versioned run directory
7. optionally pushes model artifacts to the Hub and uploads results to GCS

### Hyperparameter selection

Edit `hyperparam_sets.py` and choose the run index inside the trainer:

```python
from hyperparam_sets import distilbert_hyperparams, biobert_hyperparams

idx = 2
hyperparameters = biobert_hyperparams[idx]
```

Each sweep entry includes:

- `model_name`
- `dataset_repo`
- `epoch`
- `lr`
- `batch_size`
- `weight_decay`
- `warmup_ratio`
- `push_to_hub`

### Training behavior by version

- `7_trainer.py` and `v01/7_trainer_gcp.py` freeze the backbone and train only the classification head.
- `v02/7_trainer_gcp.py` is set up for newer experiments and no longer freezes the full encoder stack by default.
- `7_trainer.py` is legacy and still hardcodes the `v00` dataset, so it should not be treated as the default training entrypoint.

### Output locations

Versioned trainers save artifacts under:

```text
{VERSION}/runs/{MODEL_NAME}/run_{idx}/
```

Typical contents:

```text
checkpoint-*/
val_metrics.json
test_metrics.json
val_f1_bins_plot.png
test_f1_bins_plot.png
summary.json
```

## Inference

For notebook-based inference and inspection, use `v01/inference_test.ipynb` for the stable baseline or `v02/inference_test.ipynb` for experimental checkpoints. The `v02` notebook can:

- load a checkpoint from a local `downloaded_models/` directory
- download a run from GCS if it is not present locally
- reuse the inference helpers from `inference/v01/inference_utils.py`
- convert token predictions into word-level labels and final character spans

Programmatically, the recommended helpers are still:

```python
from inference.v01.inference_utils import predict_word_level, word_labels_to_spans
```

Those helpers handle:

- WordPiece token prediction
- token-to-word aggregation via `word_ids()`
- BIO conflict resolution at word level
- conversion from word labels to character spans

## Data And Labeling

The active label scheme is the collapsed 5-class BIO setup:

- `B-SYMPTOM_POS`
- `I-SYMPTOM_POS`
- `B-SYMPTOM_NEG`
- `I-SYMPTOM_NEG`
- `O`

This replaced the earlier symptom-specific label space from `v00`, which was too sparse to train reliably.

A typical example contains:

```json
{
  "text": "Patient notes icteric eyes.",
  "word_tokens": ["Patient", "notes", "icteric", "eyes", "."],
  "word_labels": ["O", "O", "B-SYMPTOM_POS", "I-SYMPTOM_POS", "O"],
  "tokens": ["[CLS]", "patient", "notes", "ict", "##eric", "eyes", ".", "[SEP]"],
  "input_ids": [101, 5776, 3964, 25891, 22420, 2159, 1012, 102],
  "token_labels": ["None", "O", "O", "B-SYMPTOM_POS", "I-SYMPTOM_POS", "I-SYMPTOM_POS", "O", "None"],
  "token_label_ids": [-100, 3498, 3498, 1393, 3139, 3139, 3498, -100]
}
```

Key fields:

- `word_labels`: word-level BIO tags
- `token_labels`: WordPiece-aligned tags
- `token_label_ids`: integer IDs used for training
- `-100`: ignored positions for special tokens and padding

## Evaluation And Error Analysis

`metrics.py` provides the standard evaluation utilities:

- `compute_metrics()`: overall seqeval metrics
- `compute_metrics_complete()`: overall plus per-entity metrics
- `plot_metrics()`: histogram of entity-level F1 scores

The behavioural/error-analysis workflow now spans both a package and notebooks:

- `error_analysis/error_taxonomy.py`: dataclasses and taxonomy labels
- `ERROR_ANALYSIS/error_categorization.py`: categorization logic
- `ERROR_ANALYSIS/ERROR_LOG_TEMPLATE.md`: template for manual or batch failure reviews
- `v01/9_categorize_errors_in_behavioural_set.ipynb`: original behavioural categorization pass
- `v02/9_categorize_errors_in_behavoural_set.ipynb`: updated categorization pass for the current workstream

The current taxonomy covers:

- irrelevant span mislabeling
- missing expected entities
- tokenization artifacts
- incorrect polarity assignment
- BIO sequencing errors
- boundary overreach
- boundary undereach
- model overgeneralization

`PROGRESS_NOTES/v01.md` documents the validated collapsed-label baseline, while `PROGRESS_NOTES/v02.md` explains the precision-focused follow-up experiments and the motivation for the current debugging work.

## Debugging Artifacts

`v02/debugging_accuracy_1.ipynb` is an investigation notebook, not part of the canonical training pipeline. It is used to check whether unexpectedly perfect metrics are caused by leakage, overlap, or other evaluation artifacts.

`v02/test.py` currently appears to be a scratch file rather than a documented project entrypoint.

## Project Layout

```text
bert_symptom_ner/
├── 1_build_symptom_dict.ipynb
├── 7_trainer.py
├── config.py
├── gcp_utils.py
├── hf_utils.py
├── hyperparam_sets.py
├── metrics.py
├── bio_portal_symptoms/
├── ERROR_ANALYSIS/
├── error_analysis/
├── inference/
│   └── v01/
├── v00/
├── v01/
└── v02/
```

## Notes

- BioBERT remains the main model family of interest.
- Device selection is automatic: CUDA, then MPS, then CPU.
- `v02/download_model_from_gcs.py` expects GCP credentials or Application Default Credentials.
- Some naming is still inconsistent across the repo, especially around `ERROR_ANALYSIS` vs `error_analysis` and `behaviour` vs `behavior`.
- The recommended default is `v01`; `v02` is intentionally documented as experimental rather than as the canonical baseline.

## License

[TODO]

## Citation

[TODO]
