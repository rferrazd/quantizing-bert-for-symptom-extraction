# NER hyperparameter configurations inspired by BioBERT (paper-faithful)
from config import settings

biobert_hyperparams = [
    {
        "model_name":  "dmis-lab/biobert-base-cased-v1.1",
        "dataset_repo" : settings.HUGGINGFACE_REPO_ID_BIOBERT,
        "epoch": 30,                # paper: NER often needs 20+ epochs
        "lr": 3e-5,                 # one of the explicitly tested LRs
        "batch_size": 32,           # common stable setting
        "weight_decay": 0.01,       # BERT default (paper: same as BERT)
        "warmup_ratio": 0.1,        # BERT default (paper: same schedule)
        "push_to_hub": True
    },
    {
        "model_name": "dmis-lab/biobert-base-cased-v1.1",
        "dataset_repo" : settings.HUGGINGFACE_REPO_ID_BIOBERT,
        "epoch": 25,
        "lr": 1e-5,                 # lower LR, often best for token-level NER
        "batch_size": 32,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "push_to_hub": True
    },
    {
        "model_name": "dmis-lab/biobert-base-cased-v1.1",
        "dataset_repo" : settings.HUGGINGFACE_REPO_ID_BIOBERT,
        "epoch": 20,
        "lr": 5e-5,                 # upper bound tested in the paper
        "batch_size": 16,           # smaller batch, higher LR pairing
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "push_to_hub": True
    }
]


distilbert_hyperparams = [
    {
        "model_name": "distilbert-base-uncased",
        "dataset_repo" : settings.HUGGINGFACE_REPO_ID,
        "epoch": 20,                # DistilBERT may converge faster, but NER often needs many epochs
        "lr": 5e-5,                 # Standard for DistilBERT; start with a moderate LR
        "batch_size": 32,           # Common stable setting for DistilBERT
        "weight_decay": 0.01,       # BERT/DistilBERT default
        "warmup_ratio": 0.1,        # Default warmup
        "push_to_hub": True
    },
    {
        "model_name": "distilbert-base-uncased",
        "dataset_repo" : settings.HUGGINGFACE_REPO_ID,
        "epoch": 15,
        "lr": 3e-5,                 # Lower LR for more stable token-level NER
        "batch_size": 32,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "push_to_hub": True
    },
    {
        "model_name": "distilbert-base-uncased",
        "dataset_repo" : settings.HUGGINGFACE_REPO_ID,
        "epoch": 10,
        "lr": 2e-5,                 # Even lower LR; try fewer epochs for model stability
        "batch_size": 16,           # Smaller batch in case of memory constraints or higher LR
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "push_to_hub": True
    }
]