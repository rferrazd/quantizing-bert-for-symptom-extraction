"""Trains a given model with different"""

import random
from datasets import load_dataset
import torch
import os
import json
import time
from dotenv import load_dotenv
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from huggingface_hub import hf_hub_download
# Local imports 
from config import settings
from metrics import compute_metrics, plot_metrics
from gcp_utils import upload_to_gcs, verify_upload

seed = 18
random.seed(seed)
torch.manual_seed(seed)

def print_title(text,n=50):
    print("="*n)
    print(text)
    print("="*n)


# -----------------------------------------------------------
# Load data and id2label/label2id mappings from Hugging Face 
# -----------------------------------------------------------
HUGGINGFACE_REPO_ID = "Rogarcia18/symptoms_ner_v00"


dataset = load_dataset(HUGGINGFACE_REPO_ID)
# NOTE that the actual labels that will be used for training are under the column: "token_label_ids"
dataset = dataset.rename_column("token_label_ids", "labels")

# Download and load id2label.json from the hub
id2label_path = hf_hub_download(
    repo_id=HUGGINGFACE_REPO_ID,
    filename="id2label.json",
    repo_type="dataset"
)
# Download and load label2id.json from the hub
label2id_path = hf_hub_download(
    repo_id=HUGGINGFACE_REPO_ID,
    filename="label2id.json",
    repo_type="dataset"
)
with open(id2label_path, "r") as f:
    id2label = json.load(f)
with open(label2id_path, "r") as f:
    label2id = json.load(f)

# Number of labels 
num_labels = len(id2label)

# -----------------------------------------------------
# Select Available Device
# -----------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed_all(seed)
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")



# ============================================================
# TRAIN FUNCTION
# ============================================================
def train(hyperparameters, idx=0):
    """Train a model with given hyperparameters"""
    MODEL_NAME = hyperparameters["model_name"]
    print_title(f"FINETUNING: {MODEL_NAME}")


    # ============================================================
    # Import model classes based on model_name
    # ============================================================
    if MODEL_NAME == "distilbert-base-uncased":
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForTokenClassification,
        )
        TokenizerClass = DistilBertTokenizerFast
        ModelClass = DistilBertForTokenClassification
    elif "biobert" in MODEL_NAME.lower():
        from transformers import (
            BertTokenizerFast,
            BertForTokenClassification,
        )
        TokenizerClass = BertTokenizerFast
        ModelClass = BertForTokenClassification
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}. Supported models: 'distilbert-base-uncased', BioBERT variants (e.g., 'dmis-lab/biobert-base-cased-v1.1')")

    # Initialize model, tokenizer, and data collator
    model = ModelClass.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
        ).to(device)

    # ============================================================
    # Freeze backbone: train ONLY the last classification layer (head)
    # ============================================================
    base_model = getattr(model, "base_model", None)
    if base_model is None:
        raise AttributeError(
            f"Model {type(model).__name__} does not expose `base_model`; cannot reliably freeze backbone."
        )
    for p in base_model.parameters():
        p.requires_grad = False

    # Sanity check: print trainable vs total parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n 🏋️‍♀️🏋️‍♀️🏋️‍♀️ Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%) 🏋️‍♀️🏋️‍♀️🏋️‍♀️")

    tokenizer = TokenizerClass.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # ============================================================
    # Output Dir + Training Arguments
    # ============================================================
    OUTPUT_DIR = f"runs/{MODEL_NAME}/run_{idx}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    USE_WANDB = os.getenv("USE_WANDB", "false").lower() == "true"


    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        push_to_hub=hyperparameters["push_to_hub"],
        learning_rate=hyperparameters["lr"],
        per_device_train_batch_size=hyperparameters["batch_size"],
        per_device_eval_batch_size=hyperparameters["batch_size"],
        num_train_epochs=hyperparameters["epoch"],
        weight_decay=hyperparameters["weight_decay"],
        warmup_ratio=hyperparameters["warmup_ratio"],
        save_strategy="epoch",  # align checkpointing with eval
        logging_strategy="epoch",  # keep logging light
        load_best_model_at_end=True,  # restores best checkpoint
        metric_for_best_model="f1",  # f1 is returned by compute_metrics
        save_total_limit=1,  # keep only the best checkpoint
        report_to=["wandb"] if USE_WANDB else [],  # enable Weights & Biases logging (cloud safe setup if api key is NOT provided)
        run_name=f"{MODEL_NAME}-lr{hyperparameters['lr']}-bs{hyperparameters['batch_size']}-ep{hyperparameters['epoch']}",  # shows in W&B runs
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # call compute_metrics with the argument id2label=id2label
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, id2label=id2label),
    )

    # ========================
    # Train the model
    # ========================
    print_title(f"Starting training with hyperparameters: {hyperparameters}")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # =============================
    # Evaluation 
    # =============================

    # ----------------------------
    # Validation Set
    # -----------------------------
    print_title("VALIDATION SET EVALUATION")
    val_predictions = trainer.predict(test_dataset=dataset["validation"])
    val_metrics = compute_metrics(val_predictions, id2label=id2label, save_path=f"{OUTPUT_DIR}/val_metrics.json")
    val_plot, val_data = plot_metrics(metrics=val_metrics, save_path=f"{OUTPUT_DIR}/val_f1_bins_plot.png")
    
    
    # ----------------------------
    # Test Set
    # -----------------------------
    print_title("TEST SET EVALUATION")
    test_predictions = trainer.predict(test_dataset=dataset["test"])
    TEST_METRICS_PATH = f"{OUTPUT_DIR}/test_metrics.json"
    test_metrics = compute_metrics(test_predictions, id2label=id2label, save_path=TEST_METRICS_PATH)
    TEST_PLOTS_PATH = f"{OUTPUT_DIR}/test_f1_bins_plot.png"
    test_plot, test_data = plot_metrics(metrics=test_metrics, save_path=TEST_PLOTS_PATH)

    # ----------------------------
    # Save summary
    # -----------------------------
    summary = {
        "model_name": MODEL_NAME,
        "training_time_minutes": round(training_time / 60, 2),
        "hyperparameters": hyperparameters,
        "validation_metrics": {
            "f1": val_metrics.get("f1", 0.0),
            "precision": val_metrics.get("precision", 0.0),
            "recall": val_metrics.get("recall", 0.0),
            "accuracy": val_metrics.get("accuracy", 0.0)
        },
        "test_metrics": {
            "f1": test_metrics.get("f1", 0.0),
            "precision": test_metrics.get("precision", 0.0),
            "recall": test_metrics.get("recall", 0.0),
            "accuracy": test_metrics.get("accuracy", 0.0)
        }
    }

    summary_path = f"{OUTPUT_DIR}/summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print_title(f"Training completed! Summary saved to: {summary_path}")
    print(f"✓ Overall Test F1: {test_metrics.get('f1', 0.0):.4f}")
    print(f"✓ Validation F1: {val_metrics.get('f1', 0.0):.4f}")

    if settings.SAVE_TO_GCS in ("true"):
        upload_to_gcs(local_path = OUTPUT_DIR, gcs_path = OUTPUT_DIR, bucket_name = settings.BUCKET_NAME)
        verify_upload(bucket_name = settings.BUCKET_NAME, gcs_path = OUTPUT_DIR)


        
        


if __name__ == "__main__":
    # ============================================================
    # HYPERPARAMETERS CONFIGURATION
    # ============================================================
    from hyperparam_sets import distilbert_hyperparams
    idx = 0
    hyperparameters = distilbert_hyperparams[idx]

    train(hyperparameters, idx=idx)

