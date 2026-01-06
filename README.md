# BERT Symptom NER

A Named Entity Recognition (NER) system for extracting medical symptoms from text using BERT-based models. This project trains token classification models to identify and classify symptom entities in medical narratives with support for both positive and negative symptom detection.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# Required
HF_TOKEN=your_huggingface_token

# Optional
WANDB_API_KEY=your_wandb_key
USE_WANDB=true
BIO_PORTAL_API_KEY=your_bioportal_key

# For GCP training
SAVE_TO_GCS=true
BUCKET_NAME=ner_training_data_results
HUGGINGFACE_REPO_ID=Rogarcia18/symptoms_ner_v00
HUGGINGFACE_REPO_ID_BIOBERT=Rogarcia18/symptoms_ner_v00_biobert
HUGGINGFACE_MODEL_REPO_ID=your_username/your_model_repo
```

### Run Training

**Local Training:**
```bash
python 7_trainer.py
```

**GCP Training:**
```bash
# Build and deploy to GCP (see TRAINING_ON_GCP_DOCKER.md for details)
./deploy_to_gcp.sh
# Or run directly:
python 7_trainer_gcp.py
```

### Load and Use a Trained Model

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load model and tokenizer from checkpoint
checkpoint_path = "runs/distilbert-base-uncased/run_0/checkpoint-XXXX"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
model.eval()

# Inference
text = "Patient reports chest pain and shortness of breath."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Decode predictions
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = [id2label[str(pred.item())] for pred in predictions[0]]
```

## Project Structure

### Data Pipeline (Notebooks 1-6)

The workflow follows this sequence:

1. **`1_build_symptom_dict.ipynb`** - Extracts symptom ontology from BioPortal (DOID), creates hierarchical symptom dictionary (~895 symptoms)
2. **`2_generate_tokenized_synthetic_data.ipynb`** - Generates synthetic training examples using templates (affirmed/negated), creates word-level tokens and BIO labels
3. **`3_wordpiece_tokenization_distillbert.ipynb`** - Converts word-level tokens to WordPiece subwords for DistilBERT, aligns labels
4. **`3_wordpiece_tokenization_biobert.ipynb`** - Converts word-level tokens to WordPiece subwords for BioBERT, aligns labels
5. **`4_generate_splits.ipynb`** - Creates train/validation/test splits (80/10/10)
6. **`5_save_dataset_to_hf_hub.ipynb`** - Uploads dataset to HuggingFace Hub
7. **`6_save_ids2tokens_to_hf_hub.ipynb`** - Saves label ID mappings to HuggingFace Hub

### Training Scripts

- **`7_trainer.py`** - Local training script (CUDA/MPS/CPU)
- **`7_trainer_gcp.py`** - GCP-optimized training script (Vertex AI)
- **`7_trainer.ipynb`** - Training notebook (alternative)

### Key Files

- **`metrics.py`** - Evaluation functions (seqeval integration, F1 plots)
- **`hyperparam_sets.py`** - Hyperparameter configurations for each model
- **`config.py`** - Environment variable management
- **`gcp_utils.py`** - Google Cloud Storage utilities
- **`hf_utils.py`** - HuggingFace Hub utilities
- **`inference/app.py`** - FastAPI inference endpoint (skeleton)

## How to Run

### Configure Hyperparameters

Edit `hyperparam_sets.py` or modify the script directly:

```python
# In 7_trainer.py or 7_trainer_gcp.py
from hyperparam_sets import distilbert_hyperparams, biobert_hyperparams

# Select configuration (0, 1, 2, etc.)
idx = 0
hyperparameters = distilbert_hyperparams[idx]  # or biobert_hyperparams[idx]

train(hyperparameters, idx=idx)
```

### Training Process

The training script automatically:

1. Loads dataset from HuggingFace Hub
2. Selects hyperparameters from `hyperparam_sets.py`
3. Initializes model and tokenizer (DistilBERT or BioBERT)
4. **Freezes backbone**, trains only classification head
5. Trains with specified hyperparameters
6. Evaluates on validation and test sets
7. Saves metrics, plots, and summary to `runs/{MODEL_NAME}/run_{idx}/`

### Device Selection

Device selection is automatic (CUDA > MPS > CPU). No manual configuration needed.

### Output Location

All results are saved to:
```
runs/{MODEL_NAME}/run_{idx}/
├── checkpoint-*/          # Best model checkpoint
├── val_metrics.json       # Validation metrics
├── test_metrics.json      # Test metrics
├── val_f1_bins_plot.png   # Validation F1 distribution
├── test_f1_bins_plot.png  # Test F1 distribution
└── summary.json           # Training summary
```

## Logic Explanation

### BIO Tagging Scheme

Each symptom entity uses BIO (Beginning-Inside-Outside) tagging with symptom ID and polarity:

- **B-SYMPTOM_{id}_POS**: Beginning of positive symptom entity
- **I-SYMPTOM_{id}_POS**: Inside positive symptom entity
- **B-SYMPTOM_{id}_NEG**: Beginning of negative symptom entity
- **I-SYMPTOM_{id}_NEG**: Inside negative symptom entity
- **O**: Outside any entity

Example: `"Patient has chest pain"` → `["O", "O", "B-SYMPTOM_s0123_POS", "I-SYMPTOM_s0123_POS"]`

### WordPiece Tokenization

- Word-level tokens are split into WordPiece subwords (e.g., "hypotension" → `["hyp", "##ot", "##ension"]`)
- Labels are aligned: first subword gets the original label, continuation subwords get `I-` variant
- Special tokens (`[CLS]`, `[SEP]`) are labeled as `-100` (ignored in loss)

### Evaluation Metrics

- **Overall F1**: Micro-averaged F1 score (sequence-level, from seqeval)
- **Overall Precision/Recall**: Sequence-level metrics
- **Overall Accuracy**: Token-level accuracy
- **Per-Entity Metrics**: F1, precision, recall for each symptom entity type

### Model Selection

- Best model is selected based on **overall F1 score** during validation
- Only the best checkpoint is saved (`save_total_limit=1`)
- Model backbone is frozen; only the classification head is trained

## Key Features

### Supported Models

- **DistilBERT**: `distilbert-base-uncased` - Lightweight, fast training
- **BioBERT**: `dmis-lab/biobert-base-cased-v1.1` - Domain-specific for biomedical text

### Hyperparameter Configurations

Pre-configured sets in `hyperparam_sets.py`:

- **DistilBERT**: Learning rates (2e-5 to 5e-5), batch sizes (16-32), epochs (10-20)
- **BioBERT**: Learning rates (1e-5 to 5e-5), batch sizes (16-32), epochs (20-30)

### Training Features

- **Dataset**: Loaded from HuggingFace Hub (separate repos for DistilBERT/BioBERT)
- **Device Support**: Automatic CUDA/MPS/CPU detection
- **Evaluation**: After each epoch on validation set
- **Checkpointing**: Saves only best model
- **Experiment Tracking**: Optional Weights & Biases integration
- **GCP Support**: Docker containers for Vertex AI training with NVIDIA T4 GPUs

### Evaluation Output

After training, generates:

- **Metrics JSON**: Overall and per-entity metrics (precision, recall, F1)
- **F1 Distribution Plots**: Histograms with color coding (green ≥0.8, orange ≥0.5, red <0.5)
- **Summary JSON**: Training time, hyperparameters, key metrics

## Dataset Structure

Each example contains:

```json
{
  "text": "Patient notes icteric eyes.",
  "word_tokens": ["Patient", "notes", "icteric", "eyes", "."],
  "word_labels": ["O", "O", "B-SYMPTOM_s0697_POS", "I-SYMPTOM_s0697_POS", "O"],
  "tokens": ["[CLS]", "patient", "notes", "ict", "##eric", "eyes", ".", "[SEP]"],
  "input_ids": [101, 5776, 3964, 25891, 22420, 2159, 1012, 102],
  "token_labels": ["None", "O", "O", "B-SYMPTOM_s0697_POS", "I-SYMPTOM_s0697_POS", "I-SYMPTOM_s0697_POS", "O", "None"],
  "token_label_ids": [-100, 3498, 3498, 1393, 3139, 3139, 3498, -100]
}
```

Note: `token_label_ids` is renamed to `labels` during training for HuggingFace compatibility.

## GCP Training

For training on Google Cloud Platform, see [TRAINING_ON_GCP_DOCKER.md](TRAINING_ON_GCP_DOCKER.md) for complete setup instructions.

Quick deployment:
```bash
./deploy_to_gcp.sh
```

This builds a Docker image, pushes to Artifact Registry, and creates a Vertex AI Custom Job with NVIDIA T4 GPU.

## Project Structure

```
bert_symptom_ner/
├── 1_build_symptom_dict.ipynb              # BioPortal ontology extraction
├── 2_generate_tokenized_synthetic_data.ipynb # Synthetic data generation
├── 3_wordpiece_tokenization_distillbert.ipynb # DistilBERT tokenization
├── 3_wordpiece_tokenization_biobert.ipynb   # BioBERT tokenization
├── 4_generate_splits.ipynb                  # Train/val/test splits
├── 5_save_dataset_to_hf_hub.ipynb           # Dataset upload
├── 6_save_ids2tokens_to_hf_hub.ipynb        # Label mappings upload
├── 7_trainer.py                              # Local training script
├── 7_trainer_gcp.py                          # GCP training script
├── 7_trainer.ipynb                           # Training notebook
├── metrics.py                                # Evaluation functions
├── hyperparam_sets.py                        # Hyperparameter configurations
├── config.py                                 # Environment configuration
├── gcp_utils.py                              # GCS utilities
├── hf_utils.py                               # HuggingFace Hub utilities
├── base_symptom_dict.csv                     # Symptom dictionary
├── requirements.txt                          # Python dependencies
├── Dockerfile.train                          # Docker image for GCP
├── deploy_to_gcp.sh                          # GCP deployment script
├── inference/
│   └── app.py                                # FastAPI inference (skeleton)
└── runs/                                     # Training outputs (gitignored)
    └── {MODEL_NAME}/
        └── run_{idx}/
            ├── checkpoint-*/
            ├── val_metrics.json
            ├── test_metrics.json
            ├── val_f1_bins_plot.png
            ├── test_f1_bins_plot.png
            └── summary.json
```

## Notes

- Training uses fixed random seed (18) for reproducibility
- Model backbone is frozen; only classification head is trained
- Best model selection based on overall F1 score (micro-averaged)
- Device selection is automatic (CUDA > MPS > CPU)
- Dataset column `token_label_ids` is renamed to `labels` for training
- GCP training automatically uploads results to Google Cloud Storage

## License

[TODO]

## Citation

[TODO]
