"""
Trains a given model specifically for v02 

Optimized code for training on GCP (Google Cloud Platform)

Important points:
- Avoid manually adding model.to(device) since it could lead to issues with multi-GPU / distributed setups
- Remove device selection part, Trainer object will internally handle this
- Setup a cache in GCP using /tmp
"""

# -----------------------------
# Hugging Face cache directories (GCP / Docker safe)
# -----------------------------
import os
from config import settings 


# HF cache directories are now loaded from config.settings
print("HF_HOME:", settings.HF_HOME)
print("HF_DATASETS_CACHE:", settings.HF_DATASETS_CACHE)

import random
from datasets import load_dataset
import torch
import os
import json
import time
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
)
from huggingface_hub import hf_hub_download
# Local imports 
from metrics import compute_metrics, compute_metrics_complete, plot_metrics
from gcp_utils import upload_to_gcs, verify_upload
from hf_utils import ensure_branch_exists
seed = 18
random.seed(seed)
torch.manual_seed(seed)

def print_title(text,n=50):
    print("="*n)
    print(text)
    print("="*n)


# -----------------------------------------------------
# Check Available Device
# -----------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed_all(seed)
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# -----------------------------------------------------
# Check if saving content to GCS 
# -----------------------------------------------------
SAVE_TO_GCS = os.getenv("SAVE_TO_GCS", "false").lower() == "true"


# ============================================================
# TRAIN FUNCTION
# ============================================================
def train(hyperparameters, idx):
    """Train a model with given hyperparameters"""
    MODEL_NAME = hyperparameters["model_name"]
    DATASET_REPO_ID = hyperparameters["dataset_repo"]
    print_title(f"💕 FINETUNING: {MODEL_NAME}             💕")
    print_title(f"💕 Dataset Repo ID: {DATASET_REPO_ID}   💕")

    # -----------------------------------------------------------
    # Load data and id2label/label2id mappings from Hugging Face 
    # -----------------------------------------------------------
    # Disable progress bars for dataset loading to reduce log verbosity
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BAR"] = "1"
    
    dataset = load_dataset(DATASET_REPO_ID)
    # NOTE that the actual labels that will be used for training are under the column: "token_label_ids"
    dataset = dataset.rename_column("token_label_ids", "labels")

    # Download and load id2label.json from the hub
    id2label_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="id2label.json",
        repo_type="dataset"
    )
    # Download and load label2id.json from the hub
    label2id_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="label2id.json",
        repo_type="dataset"
    )
    with open(id2label_path, "r") as f:
        id2label = json.load(f)
    with open(label2id_path, "r") as f:
        label2id = json.load(f)

    # Number of labels 
    num_labels = len(id2label)

    # Initialize model, tokenizer, and data collator
    # NOTE: `dmis-lab/biobert-base-cased-v1.1` ships a legacy `config.json` without `model_type`,
    # which breaks `AutoConfig`/`AutoModel` detection. Fall back to explicit BERT classes.
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME,
            num_labels=num_labels,
            id2label=id2label,  # String keys from JSON - kept as strings due to kwargs overwrite in from_dict()
            label2id=label2id,
        )
    except ValueError as e:
        if "model_type" not in str(e):
            raise
        config = BertConfig.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        model = BertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME,
            config=config,
        )

    # =====================================================================
    # Freeze backbone: train ONLY the last classification layer (head)
    # =====================================================================
    # For *ForTokenClassification models, the head is typically `classifier`.
    # Freezing the backbone prevents updates to the transformer weights.
    
    # TRAINING THE ENTIRE MODEL
    # for name, param in model.named_parameters():
    #     if not name.startswith("classifier"):
    #         param.requires_grad = False

    # Sanity check: print trainable vs total parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n 🏋️‍♀️🏋️‍♀️🏋️‍♀️ Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%) 🏋️‍♀️🏋️‍♀️🏋️‍♀️")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except (ValueError, OSError):
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # ============================================================
    # Output Dir + Training Arguments
    # ============================================================
    OUTPUT_DIR = f"{settings.VERSION}/runs/{MODEL_NAME}/run_{idx}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    USE_WANDB = os.getenv("USE_WANDB", "false").lower() == "true"
    
    # Prepare Hugging Face Hub configuration (if push_to_hub is enabled)
    hub_model_repo_id = None
    hub_branch = None
    if hyperparameters["push_to_hub"] and settings.HUGGINGFACE_MODEL_REPO_ID:
        hub_model_repo_id = settings.HUGGINGFACE_MODEL_REPO_ID
        # Create a unique branch name for this run
        # Format: run-{idx}-{model_short}-lr{lr}-bs{bs}-ep{ep}
        # Replace slashes and special chars for valid branch name
        model_short = MODEL_NAME.split("/")[-1].replace("-", "_")
        hub_branch = f"run-{idx}-{model_short}-lr{hyperparameters['lr']}-bs{hyperparameters['batch_size']}-ep{hyperparameters['epoch']}"
        # Replace dots and other invalid chars
        hub_branch = hub_branch.replace(".", "_").replace("e-", "e").replace("e+", "e")
        # Ensure that the branch exists before trying to push to HF otherwise it will raise an error
        ensure_branch_exists(repo_id=hub_model_repo_id, branch_name=hub_branch, repo_type="model")
        print(f"📤 Will push to Hugging Face Hub: {hub_model_repo_id} (branch: {hub_branch})")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        push_to_hub=hyperparameters["push_to_hub"],
        hub_model_id=hub_model_repo_id,  # Repository ID for pushing
        hub_strategy="end",  # Push only the final best model (after load_best_model_at_end)
        hub_revision=hub_branch,  # Branch name (creates new branch for each run)
        learning_rate=hyperparameters["lr"],
        per_device_train_batch_size=hyperparameters["batch_size"],
        per_device_eval_batch_size=hyperparameters["batch_size"],
        num_train_epochs=hyperparameters["epoch"],
        weight_decay=hyperparameters["weight_decay"],
        warmup_ratio=hyperparameters["warmup_ratio"],
        save_strategy="epoch",  # align checkpointing with eval
        logging_strategy="epoch",  # keep logging light
        load_best_model_at_end=True,  # restores best checkpoint
        # means that metric_for_best_model the greater it is the better it is
        greater_is_better=True,
        metric_for_best_model="f1",  # f1 is returned by compute_metrics
        save_total_limit=1,  # keep only the best checkpoint
        report_to=["wandb"] if USE_WANDB else [],  # enable Weights & Biases logging (cloud safe setup if api key is NOT provided)
        run_name=f"{MODEL_NAME}-lr{hyperparameters['lr']}-bs{hyperparameters['batch_size']}-ep{hyperparameters['epoch']}",  # shows in W&B runs
        disable_tqdm=True,  # disable progress bars to reduce log verbosity
        #dataloader_num_workers=0,  # disable multiprocessing to reduce worker log spam
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
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
    print(f"✅ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Explicitly push to Hub if enabled (hub_strategy="end" should do this, but explicit is safer)
    if hyperparameters["push_to_hub"] and hub_model_repo_id:
        print_title("PUSHING MODEL TO HUGGING FACE HUB")
        try:
            trainer.push_to_hub()
            print(f"✅ Model successfully pushed to {hub_model_repo_id} (branch: {hub_branch})")
        except Exception as e:
            print(f"❌ Failed to push model to Hub: {e}")
            print("⚠️ Model was trained successfully but push to Hub failed. Check logs above.")

    # =============================
    # Evaluation 
    # =============================

    # ----------------------------
    # Validation Set
    # -----------------------------
    print_title("VALIDATION SET EVALUATION")
    val_predictions = trainer.predict(test_dataset=dataset["validation"])
    val_metrics_path = f"{OUTPUT_DIR}/val_metrics.json"
    #val_metrics = compute_metrics(val_predictions, id2label=id2label, save_path=val_metrics_path)
    val_metrics, val_metrics_complete = compute_metrics_complete(val_predictions, id2label=id2label, save_path=val_metrics_path)
    val_plot, val_data = plot_metrics(metrics=val_metrics_complete, save_path=f"{OUTPUT_DIR}/val_f1_bins_plot.png", metrics_file_path=val_metrics_path)

    # ----------------------------
    # Test Set
    # -----------------------------
    print_title("TEST SET EVALUATION")
    test_predictions = trainer.predict(test_dataset=dataset["test"])
    test_metrics_path = f"{OUTPUT_DIR}/test_metrics.json"
    # test_metrics = compute_metrics(test_predictions, id2label=id2label, save_path=test_metrics_path)
    test_metrics, test_metrics_complete = compute_metrics_complete(test_predictions, id2label=id2label, save_path=test_metrics_path)
    test_plot, test_data = plot_metrics(metrics=test_metrics_complete, save_path=f"{OUTPUT_DIR}/test_f1_bins_plot.png", metrics_file_path=test_metrics_path)

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

    if SAVE_TO_GCS:
        print_title("UPLOADING RESULTS TO GCS")
        
        # Upload entire OUTPUT_DIR (includes checkpoint, metrics, plots, etc.)
        # This preserves the complete directory structure: runs/{MODEL_NAME}/run_{idx}/
        upload_success = upload_to_gcs(
            local_path=OUTPUT_DIR,
            gcs_path=OUTPUT_DIR,
            bucket_name=settings.BUCKET_NAME
        )
        verify_success = verify_upload(
            bucket_name=settings.BUCKET_NAME,
            gcs_path=OUTPUT_DIR
        )
        
        if upload_success and verify_success:
            print_title(f"✅ All results uploaded to gs://{settings.BUCKET_NAME}/{OUTPUT_DIR}")
        else:
            print_title(f"⚠️ Upload completed with errors. Check logs above for details.")


if __name__ == "__main__":
    # ============================================================
    # HYPERPARAMETERS CONFIGURATION
    # ============================================================
    from hyperparam_sets import distilbert_hyperparams, biobert_hyperparams
    
    idx = 2 # BEST PERFORMING RUN FROM V01
    hyperparameters = biobert_hyperparams[idx]    

    train(hyperparameters, idx=idx)

