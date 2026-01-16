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
HF_USERNAME=your_huggingface_username
VERSION=v01  # or v00, depending on your dataset version

# Optional
WANDB_API_KEY=your_wandb_key
USE_WANDB=true
BIO_PORTAL_API_KEY=your_bioportal_key

# For GCP training
SAVE_TO_GCS=true
BUCKET_NAME=ner_training_data_results
```

The `config.py` automatically constructs dataset repo IDs from `HF_USERNAME` and `VERSION`:
- `{HF_USERNAME}/symptoms_ner_{VERSION}` (DistilBERT)
- `{HF_USERNAME}/symptoms_ner_{VERSION}_biobert` (BioBERT)

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

**Note:** `7_trainer.py` uses a hardcoded dataset repo ID. For version-specific datasets, use `7_trainer_gcp.py` which reads the dataset repo from `hyperparam_sets.py`.

### Load and Use a Trained Model

**Basic Inference:**
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import json
from huggingface_hub import hf_hub_download

# Load model and tokenizer from checkpoint
checkpoint_path = "runs/distilbert-base-uncased/run_0/checkpoint-XXXX"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
model.eval()

# Load label mappings
id2label_path = hf_hub_download(
    repo_id="your_username/symptoms_ner_v01",
    filename="id2label.json",
    repo_type="dataset"
)
with open(id2label_path, "r") as f:
    id2label = json.load(f)

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

**Using Inference Utilities (Recommended):**

The `v01/inference_utils.py` module provides a complete inference pipeline:

1. **Token-level prediction**: Model outputs predictions for each WordPiece token
2. **Word-level aggregation**: Uses `tokenizer.word_ids()` to map subword tokens back to original words
3. **BIO aggregation**: When multiple tokens belong to one word, `B-` labels take priority over `I-` labels
4. **Span extraction**: Converts word-level BIO labels to character-level spans

```python
from v01.inference_utils import predict_word_level, word_labels_to_spans
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
text = "Patient reports chest pain and shortness of breath."

# Get word-level predictions (handles token→word aggregation automatically)
tokens, token_labels, word_ids, words, word_labels = predict_word_level(
    text=text,
    model=model,
    tokenizer=tokenizer,
    id2label=id2label,
    device=device
)

# Extract character-level entity spans
spans = word_labels_to_spans(words, word_labels)
# Returns: [{"start": int, "end": int, "text": str, "label": str}, ...]
# Example: [{"start": 15, "end": 25, "text": "chest pain", "label": "SYMPTOM_POS"}, ...]
```

## Project Structure

### Data Pipeline (Notebooks 1-6)

The workflow follows this sequence:

1. **`1_build_symptom_dict.ipynb`** - Extracts symptom ontology from BioPortal (DOID), creates hierarchical symptom dictionary (~895 symptoms)
2. **`v01/2_generate_tokenized_synthetic_data.ipynb`** - Generates synthetic training examples using templates (affirmed/negated), creates word-level tokens and BIO labels
3. **`v01/3_wordpiece_tokenization_distillbert.ipynb`** - Converts word-level tokens to WordPiece subwords for DistilBERT, aligns labels
4. **`v01/3_wordpiece_tokenization_biobert.ipynb`** - Converts word-level tokens to WordPiece subwords for BioBERT, aligns labels
5. **`v01/4_generate_splits.ipynb`** - Creates train/validation/test splits (80/10/10)
6. **`5_save_dataset_to_hf_hub.ipynb`** - Uploads dataset to HuggingFace Hub
7. **`6_save_ids2tokens_to_hf_hub.ipynb`** - Saves label ID mappings to HuggingFace Hub

### Training Scripts

- **`7_trainer.py`** - Local training script (CUDA/MPS/CPU). Uses hardcoded dataset repo ID.
- **`7_trainer_gcp.py`** - GCP-optimized training script (Vertex AI). Reads dataset repo from hyperparameter config.

### Key Files

- **`metrics.py`** - Evaluation functions:
  - `compute_metrics()`: Returns micro-averaged overall metrics + per-entity metrics (SYMPTOM_POS, SYMPTOM_NEG)
  - `plot_metrics()`: Generates F1 distribution histograms with color-coded bins
- **`hyperparam_sets.py`** - Hyperparameter configurations for each model (includes dataset repo IDs)
- **`config.py`** - Environment variable management and settings
- **`gcp_utils.py`** - Google Cloud Storage utilities
- **`hf_utils.py`** - HuggingFace Hub utilities
- **`v01/inference_utils.py`** - Inference utilities (word-level predictions, span extraction)

## How to Run

### Configure Hyperparameters

Edit `hyperparam_sets.py` to modify hyperparameters or select a configuration:

```python
# In 7_trainer.py or 7_trainer_gcp.py
from hyperparam_sets import distilbert_hyperparams, biobert_hyperparams

# Select configuration (0, 1, 2, etc.)
idx = 0
hyperparameters = distilbert_hyperparams[idx]  # or biobert_hyperparams[idx]

train(hyperparameters, idx=idx)
```

Each hyperparameter set includes:
- `model_name`: Model identifier
- `dataset_repo`: HuggingFace dataset repository ID
- `epoch`: Number of training epochs
- `lr`: Learning rate
- `batch_size`: Batch size
- `weight_decay`: Weight decay
- `warmup_ratio`: Warmup ratio
- `push_to_hub`: Whether to push model to HuggingFace Hub

### Training Process

The training script automatically:

1. Loads dataset from HuggingFace Hub (using repo ID from hyperparameters)
2. Downloads label mappings (`id2label.json`, `label2id.json`) from the dataset repo
3. Selects hyperparameters from `hyperparam_sets.py`
4. Initializes model and tokenizer (DistilBERT or BioBERT)
5. **Freezes backbone**, trains only classification head
6. Trains with specified hyperparameters
7. Evaluates on validation and test sets after each epoch
8. Saves metrics, plots, and summary to `runs/{MODEL_NAME}/run_{idx}/`

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

**v0.1 (Current)**: Uses collapsed labels (5 total) to reduce label sparsity:
- **B-SYMPTOM_POS**: Beginning of positive symptom entity
- **I-SYMPTOM_POS**: Inside positive symptom entity
- **B-SYMPTOM_NEG**: Beginning of negative symptom entity
- **I-SYMPTOM_NEG**: Inside negative symptom entity
- **O**: Outside any entity

Example: `"Patient has chest pain"` → `["O", "O", "B-SYMPTOM_POS", "I-SYMPTOM_POS"]`

**Note**: v0.0 used symptom-specific IDs (e.g., `B-SYMPTOM_s0123_POS`) but failed due to extreme label sparsity (~3,500 labels, ~10 examples each). v0.1 collapses to polarity-only labels for better learning.

### WordPiece Tokenization

- Word-level tokens are split into WordPiece subwords (e.g., "hypotension" → `["hyp", "##ot", "##ension"]`)
- Labels are aligned: first subword gets the original label, continuation subwords get `I-` variant
- Special tokens (`[CLS]`, `[SEP]`) are labeled as `-100` (ignored in loss)

### Evaluation Metrics

Uses `seqeval` for sequence-level evaluation:

- **Overall Metrics** (micro-averaged): F1, precision, recall, accuracy computed across all tokens
- **Per-Entity Metrics**: Separate F1, precision, recall for `SYMPTOM_POS` and `SYMPTOM_NEG` entities
- **F1 Distribution Plots**: Histograms showing per-entity F1 scores with color coding (green ≥0.8, orange ≥0.5, red <0.5)

Micro-averaging aggregates all classes together, giving more weight to frequent classes. This is appropriate for NER where the `O` class dominates.

### Model Selection

- Best model is selected based on **overall F1 score** during validation
- Only the best checkpoint is saved (`save_total_limit=1`)
- Model backbone is frozen; only the classification head is trained

## Key Features

### Supported Models

- **BioBERT**: `dmis-lab/biobert-base-cased-v1.1` - Domain-specific for biomedical text (recommended)
  - Best performance: F1 ~0.79 on test set (v0.1)
  - Better at learning symptom boundaries and negation patterns
- **DistilBERT**: `distilbert-base-uncased` - Lightweight baseline
  - F1 ~0.46 on test set (v0.1)
  - Useful for fast prototyping but significantly underperforms BioBERT

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
  "word_labels": ["O", "O", "B-SYMPTOM_POS", "I-SYMPTOM_POS", "O"],
  "tokens": ["[CLS]", "patient", "notes", "ict", "##eric", "eyes", ".", "[SEP]"],
  "input_ids": [101, 5776, 3964, 25891, 22420, 2159, 1012, 102],
  "token_labels": ["None", "O", "O", "B-SYMPTOM_POS", "I-SYMPTOM_POS", "I-SYMPTOM_POS", "O", "None"],
  "token_label_ids": [-100, 3498, 3498, 1393, 3139, 3139, 3498, -100]
}
```

**Key points:**
- `word_labels`: Original word-level BIO labels (5 labels: B/I-SYMPTOM_POS/NEG, O)
- `token_labels`: WordPiece-aligned labels (continuation subwords get `I-` variant)
- `token_label_ids`: Integer label IDs for training (-100 for special tokens, ignored in loss)
- Column renamed to `labels` during training for HuggingFace compatibility

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
├── 5_save_dataset_to_hf_hub.ipynb           # Dataset upload
├── 6_save_ids2tokens_to_hf_hub.ipynb        # Label mappings upload
├── 7_trainer.py                              # Local training script
├── 7_trainer_gcp.py                          # GCP training script
├── metrics.py                                # Evaluation functions
├── hyperparam_sets.py                        # Hyperparameter configurations
├── config.py                                 # Environment configuration
├── gcp_utils.py                              # GCS utilities
├── hf_utils.py                               # HuggingFace Hub utilities
├── base_symptom_dict.csv                     # Symptom dictionary
├── requirements.txt                          # Python dependencies
├── Dockerfile.train                          # Docker image for GCP
├── deploy_to_gcp.sh                          # GCP deployment script
├── v00/                                      # Version 0.0 notebooks and data
│   ├── 2_generate_tokenized_synthetic_data.ipynb
│   ├── 3_wordpiece_tokenization_*.ipynb
│   ├── 4_generate_splits.ipynb
│   └── data/
├── v01/                                      # Version 0.1 notebooks and data
│   ├── 2_generate_tokenized_synthetic_data.ipynb
│   ├── 3_wordpiece_tokenization_*.ipynb
│   ├── 4_generate_splits.ipynb
│   ├── 8_compare_performance.ipynb
│   ├── inference_test.ipynb
│   ├── inference_utils.py                    # Inference utilities
│   └── data/
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

- **Label Scheme**: v0.1 uses collapsed labels (5 total) to avoid label sparsity issues from v0.0
- **Training**: Fixed random seed (18) for reproducibility; backbone frozen, only classification head trained
- **Model Selection**: Best checkpoint selected based on overall F1 score (micro-averaged) during validation
- **Device**: Automatic selection (CUDA > MPS > CPU)
- **Inference**: Use `v01/inference_utils.py` for proper token→word→span pipeline with BIO aggregation
- **Performance**: BioBERT significantly outperforms DistilBERT (see `PROGRESS_NOTES/v01.md` for details)
- **GCP**: Training automatically uploads results to Google Cloud Storage when enabled

## License

[TODO]

## Citation

[TODO]
