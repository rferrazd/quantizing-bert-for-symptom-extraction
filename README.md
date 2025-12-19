# BERT Symptom NER

A Named Entity Recognition (NER) system for extracting medical symptoms from text using BERT-based models. This project trains token classification models to identify and classify symptom entities in medical narratives with support for both positive and negative symptom detection.

## Overview

This repository contains a complete pipeline for:

- Building a symptom dictionary from BioPortal ontology
- Generating synthetic training data
- Training BERT-based NER models (DistilBERT and BioBERT)
- Evaluating model performance with detailed metrics and visualizations
- Model quantization for deployment (post-training)

## Features

- **Models**: Supports DistilBERT-base-uncased and BioBERT variants
- **Task**: Multi-label NER with BIO tagging scheme
- **Labels**: Supports positive/negative symptom detection (e.g., `B-SYMPTOM_s0104_POS`, `I-SYMPTOM_s0104_NEG`)
- **Evaluation**: Overall F1, precision, recall, accuracy, and per-entity metrics using seqeval
- **Integration**: HuggingFace Hub integration for dataset and model sharing
- **Experiment Tracking**: Weights & Biases (wandb) integration
- **Hyperparameter Management**: Configurable hyperparameter sets for different models

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:

- `transformers` - HuggingFace Transformers library
- `datasets` - Dataset handling
- `seqeval` - Sequence evaluation metrics
- `wandb` - Experiment tracking
- `torch` - PyTorch
- `accelerate>=0.26.0` - Training acceleration
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities

### Environment Variables

Create a `.env` file with:

```
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key  # Optional
```

## Pipeline

The workflow is organized into sequential notebooks:

1. **`1_build_symptom_dict.ipynb`** - Extracts symptom ontology from BioPortal
2. **`2_generate_tokenized_synthetic_data.ipynb`** - Generates synthetic training examples
3. **`3_word_piece_tokenization.ipynb`** - Tokenization setup and exploration
4. **`4_generate_splits.ipynb`** - Creates train/validation/test splits (80/10/10)
5. **`5_save_dataset_to_hf_hub.ipynb`** - Uploads dataset to HuggingFace Hub
6. **`6_trainer.py`** or **`6_trainer.ipynb`** - Model training and evaluation

## Training

The training script (`6_trainer.py`) supports multiple model architectures and hyperparameter configurations:

### Supported Models

- **DistilBERT**: `distilbert-base-uncased` - Lightweight, fast training
- **BioBERT**: `dmis-lab/biobert-base-cased-v1.1` - Domain-specific for biomedical text

### Hyperparameter Configuration

Hyperparameters are managed in `hyperparam_sets.py` with separate configurations for each model:

- **DistilBERT configs**: Multiple configurations with varying learning rates (2e-5 to 5e-5), batch sizes (16-32), and epochs (2-15)
- **BioBERT configs**: Configurations optimized for biomedical NER with learning rates (1e-5 to 5e-5), batch sizes (16-32), and epochs (20-30)

### Training Features

- **Dataset**: Loaded from HuggingFace Hub (`Rogarcia18/symptoms_ner_v00`)
- **Device Support**: Automatic detection of CUDA, MPS (Apple Silicon), or CPU
- **Evaluation Strategy**: `epoch` - evaluates after each epoch
- **Best Model Selection**: Based on overall F1 score (saves best checkpoint)
- **Checkpointing**: Saves only the best model (`save_total_limit=1`)
- **Experiment Tracking**: Weights & Biases integration with detailed run names
- **Output**: Results saved to `runs/{MODEL_NAME}/run_{idx}/`

### Running Training

```bash
python 6_trainer.py
```

The script automatically:
1. Loads the dataset from HuggingFace Hub
2. Selects hyperparameters from `hyperparam_sets.py` (configurable via `idx` variable)
3. Initializes the appropriate model and tokenizer based on the model name
4. Trains with the specified hyperparameters
5. Evaluates on both validation and test sets
6. Saves metrics, plots, and summary to the output directory

### Customizing Training

Edit `6_trainer.py` to:
- Change the hyperparameter set: `from hyperparam_sets import distilbert_hyperparams, biobert_hyperparams`
- Select a different configuration: `idx = 0` (change to 1, 2, etc.)
- Enable model pushing to Hub: Set `push_to_hub: True` in hyperparameter config

## Evaluation

The `metrics.py` module provides comprehensive evaluation using the `seqeval` library:

- **`compute_metrics()`** - Overall metrics (accuracy, precision, recall, F1) and per-entity metrics
- **`plot_metrics()`** - Visualization of per-entity F1 score distribution with histogram

### Metrics

The evaluation uses seqeval which provides:

- **Overall F1**: Sequence-level F1 score (harmonic mean of precision and recall) 
    NOTE: still have to check default behavior of the evaluate library to certify if f1 equates to the macro_f1 or micro_f1
- **Overall Precision**: Sequence-level precision
- **Overall Recall**: Sequence-level recall
- **Overall Accuracy**: Token-level accuracy
- **Per-Entity Metrics**: For each symptom entity (e.g., `SYMPTOM_s0001_POS`), provides precision, recall, and F1

### Evaluation Output

After training, the script generates:

- **`val_metrics.json`** / **`test_metrics.json`**: Complete metrics including overall and per-entity scores
- **`val_f1_bins_plot.png`** / **`test_f1_bins_plot.png`**: Histogram showing distribution of per-entity F1 scores with:
  - Color-coded bars (green ≥0.8, orange ≥0.5, red <0.5)
  - Overall F1 reference line
  - Statistics (mean, median, percentage of perfect/zero F1 scores)
- **`summary.json`**: Training summary with model name, training time, hyperparameters, and key metrics

## Dataset Structure

Each example contains:

- `text`: Original sentence
- `word_tokens`: Word-level tokens
- `word_labels`: Word-level BIO labels
- `tokens`: Subword tokens (WordPiece)
- `input_ids`: Token IDs
- `token_labels`: Token-level labels
- `token_label_ids`: Numeric label IDs

## Project Structure

```
bert_symptom_ner/
├── 1_build_symptom_dict.ipynb          # BioPortal ontology extraction
├── 2_generate_tokenized_synthetic_data.ipynb  # Synthetic data generation
├── 3_word_piece_tokenization.ipynb     # Tokenization exploration
├── 4_generate_splits.ipynb              # Train/val/test splits
├── 5_save_dataset_to_hf_hub.ipynb      # Dataset upload
├── 6_trainer.py                         # Main training script
├── 6_trainer.ipynb                      # Training notebook (alternative)
├── metrics.py                           # Evaluation functions
├── hyperparam_sets.py                   # Hyperparameter configurations
├── id2label.json                        # Label ID to label mapping
├── label2id.json                        # Label to ID mapping
├── base_symptom_dict.csv                # Symptom dictionary
├── requirements.txt                     # Python dependencies
├── train.jsonl / val.jsonl / test.jsonl # Dataset splits (local)
└── runs/                                # Training outputs (gitignored)
    └── {MODEL_NAME}/
        └── run_{idx}/
            ├── val_metrics.json
            ├── test_metrics.json
            ├── val_f1_bins_plot.png
            ├── test_f1_bins_plot.png
            └── checkpoint-*/
```

## Results

After training, the model is evaluated on both validation and test sets with:

- **Overall metrics**: F1, precision, recall, and accuracy
- **Per-entity performance**: Detailed metrics for each symptom entity
- **Visualizations**: F1 score distribution histograms showing model performance across entities
- **Summary**: Training time, hyperparameters, and key metrics saved to `summary.json`

Results are saved in `runs/{MODEL_NAME}/run_{idx}/` for easy comparison across different hyperparameter configurations.

## Notes

- The training script uses a fixed random seed (18) for reproducibility
- Model checkpoints and large files are gitignored (see `.gitignore`)
- The dataset column `token_label_ids` is renamed to `labels` for training compatibility
- Best model selection is based on overall F1 score during validation
- Training automatically handles device selection (CUDA > MPS > CPU)

## License

[TODO]

## Citation

[TODO]
