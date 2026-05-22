"""
7_trainer_gcp.py

V04 training entrypoint for GCP. Loads the symptom-ner-dataset-v04 from Hugging Face,
fine-tunes BioBERT (full backbone, no freezing) with HF Trainer, evaluates on
validation (IID), template_ood, and symptom_ood sets, saves metrics, and uploads
results to GCS.
"""

import random
from datasets import load_dataset
import torch
import os, json, time
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    set_seed,
)
from huggingface_hub import hf_hub_download
# Local imports
from config import settings
from metrics import compute_metrics, compute_metrics_complete, plot_metrics
from gcp_utils import upload_to_gcs, verify_upload
from hf_utils import ensure_branch_exists

seed = 18
random.seed(seed)
torch.manual_seed(seed)
# transformers.set_seed seeds python random, numpy, torch (cpu+cuda), and dataloader workers
set_seed(seed)

# HF cache directories are now loaded from config.settings
print("HF_HOME:", settings.HF_HOME)
print("HF_DATASETS_CACHE:", settings.HF_DATASETS_CACHE)


def print_title(text: str, n: int = 50) -> None:
    """Prints a section header with surrounding separator lines."""
    print("=" * n)
    print(text)
    print("=" * n)


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
def train(hyperparameters: dict, idx: int) -> None:
    """
    Fine-tunes a token classification model with the given hyperparameters.
    Evaluates on validation, template_ood, and symptom_ood after training.
    Saves metrics/plots locally and optionally uploads to GCS.
    """
    MODEL_NAME = hyperparameters["model_name"]
    DATASET_REPO_ID = hyperparameters["dataset_repo"]
    print_title(f"FINETUNING: {MODEL_NAME}")
    print_title(f"Dataset Repo ID: {DATASET_REPO_ID}")

    # -----------------------------------------------------------
    # Load data and id2label/label2id mappings from Hugging Face
    # -----------------------------------------------------------
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BAR"] = "1"

    dataset = load_dataset(DATASET_REPO_ID, token=settings.HF_TOKEN)
    # token_label_ids is the aligned WordPiece label field; rename to 'labels' for Trainer
    dataset = dataset.rename_column("token_label_ids", "labels")

    train_ds = dataset["train"]
    validation_ds = dataset["validation"]
    template_ood_ds = dataset["template_ood"]
    symptom_ood_ds = dataset["symptom_ood"]

    # Smoke test mode: tiny slices + force no Hub push to validate the code path end-to-end
    if os.getenv("SMOKE_TEST", "false").lower() == "true":
        print_title("SMOKE TEST MODE — slicing splits to 20 samples each")
        train_ds = train_ds.select(range(20))
        validation_ds = validation_ds.select(range(20))
        template_ood_ds = template_ood_ds.select(range(20))
        symptom_ood_ds = symptom_ood_ds.select(range(20))
        hyperparameters = {**hyperparameters, "push_to_hub": False, "epoch": 1, "batch_size": 2}

    # Download label mappings from hub (private repo — token required)
    id2label_path = hf_hub_download(
        repo_id=DATASET_REPO_ID, filename="id2label.json", repo_type="dataset", token=settings.HF_TOKEN
    )
    label2id_path = hf_hub_download(
        repo_id=DATASET_REPO_ID, filename="label2id.json", repo_type="dataset", token=settings.HF_TOKEN
    )
    with open(id2label_path) as f:
        id2label = json.load(f)
    with open(label2id_path) as f:
        label2id = json.load(f)

    num_labels = len(id2label)

    # -----------------------------------------------------------
    # Load model — fall back to explicit BertForTokenClassification
    # if biobert config.json is missing model_type
    # -----------------------------------------------------------
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME,
            num_labels=num_labels,
            id2label=id2label,
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

    # Full backbone fine-tuning — all parameters are trainable
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

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

    hub_model_repo_id = None
    hub_branch = None
    if hyperparameters["push_to_hub"] and settings.HUGGINGFACE_MODEL_REPO_ID:
        hub_model_repo_id = settings.HUGGINGFACE_MODEL_REPO_ID
        model_short = MODEL_NAME.split("/")[-1].replace("-", "_")
        hub_branch = (
            f"run-{idx}-{model_short}"
            f"-lr{hyperparameters['lr']}"
            f"-bs{hyperparameters['batch_size']}"
            f"-ep{hyperparameters['epoch']}"
        )
        hub_branch = hub_branch.replace(".", "_").replace("e-", "e").replace("e+", "e")
        ensure_branch_exists(repo_id=hub_model_repo_id, branch_name=hub_branch, repo_type="model")
        print(f"Will push to Hugging Face Hub: {hub_model_repo_id} (branch: {hub_branch})")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        # Hub push is handled explicitly after training (see trainer.push_to_hub call below)
        # so we can target the per-run branch instead of `main`. hub_model_id is still needed
        # so Trainer knows the target repo when push_to_hub() is called explicitly.
        push_to_hub=False,
        hub_model_id=hub_model_repo_id,
        remove_unused_columns=True,  # drops `symptoms` (nested dict), `tokens`, etc. before collation
        learning_rate=hyperparameters["lr"],
        per_device_train_batch_size=hyperparameters["batch_size"],
        per_device_eval_batch_size=hyperparameters["batch_size"],
        num_train_epochs=hyperparameters["epoch"],
        weight_decay=hyperparameters["weight_decay"],
        warmup_ratio=hyperparameters["warmup_ratio"],
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        report_to=["wandb"] if USE_WANDB else [],
        run_name=f"{MODEL_NAME}-lr{hyperparameters['lr']}-bs{hyperparameters['batch_size']}-ep{hyperparameters['epoch']}",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, id2label=id2label),
    )

    # ========================
    # Train
    # ========================
    print_title(f"Starting training with hyperparameters: {hyperparameters}")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s ({training_time/60:.2f} min)")

    if hyperparameters["push_to_hub"] and hub_model_repo_id:
        print_title("PUSHING MODEL TO HUGGING FACE HUB")
        try:
            # Pass branch explicitly here — `hub_revision` is not a valid TrainingArguments kwarg
            trainer.push_to_hub(revision=hub_branch)
            print(f"Model pushed to {hub_model_repo_id} (branch: {hub_branch})")
        except Exception as e:
            print(f"Push to Hub failed: {e}")

    # =============================
    # Evaluation — all three sets
    # =============================

    # Validation (IID)
    print_title("VALIDATION SET (IID)")
    val_preds = trainer.predict(test_dataset=validation_ds)
    val_metrics_path = f"{OUTPUT_DIR}/val_metrics.json"
    val_metrics, val_metrics_complete = compute_metrics_complete(val_preds, id2label=id2label, save_path=val_metrics_path)
    plot_metrics(metrics=val_metrics_complete, save_path=f"{OUTPUT_DIR}/val_f1_bins_plot.png", metrics_file_path=val_metrics_path)

    # Template OOD
    print_title("TEMPLATE OOD SET")
    template_preds = trainer.predict(test_dataset=template_ood_ds)
    template_metrics_path = f"{OUTPUT_DIR}/template_ood_metrics.json"
    template_metrics, template_metrics_complete = compute_metrics_complete(template_preds, id2label=id2label, save_path=template_metrics_path)
    plot_metrics(metrics=template_metrics_complete, save_path=f"{OUTPUT_DIR}/template_ood_f1_bins_plot.png", metrics_file_path=template_metrics_path)

    # Symptom OOD
    print_title("SYMPTOM OOD SET")
    symptom_preds = trainer.predict(test_dataset=symptom_ood_ds)
    symptom_metrics_path = f"{OUTPUT_DIR}/symptom_ood_metrics.json"
    symptom_metrics, symptom_metrics_complete = compute_metrics_complete(symptom_preds, id2label=id2label, save_path=symptom_metrics_path)
    plot_metrics(metrics=symptom_metrics_complete, save_path=f"{OUTPUT_DIR}/symptom_ood_f1_bins_plot.png", metrics_file_path=symptom_metrics_path)

    # OOD gap summary
    print_title("OOD GAP SUMMARY")
    val_f1 = val_metrics.get("f1", 0.0)
    template_f1 = template_metrics.get("f1", 0.0)
    symptom_f1 = symptom_metrics.get("f1", 0.0)
    print(f"  Validation (IID)  F1: {val_f1:.4f}")
    print(f"  Template OOD      F1: {template_f1:.4f}  (gap: {val_f1 - template_f1:+.4f})")
    print(f"  Symptom  OOD      F1: {symptom_f1:.4f}  (gap: {val_f1 - symptom_f1:+.4f})")

    # =============================
    # Save summary
    # =============================
    summary = {
        "model_name": MODEL_NAME,
        "training_time_minutes": round(training_time / 60, 2),
        "hyperparameters": hyperparameters,
        "validation_metrics": {
            "f1": val_f1,
            "precision": val_metrics.get("precision", 0.0),
            "recall": val_metrics.get("recall", 0.0),
            "accuracy": val_metrics.get("accuracy", 0.0),
        },
        "template_ood_metrics": {
            "f1": template_f1,
            "precision": template_metrics.get("precision", 0.0),
            "recall": template_metrics.get("recall", 0.0),
            "accuracy": template_metrics.get("accuracy", 0.0),
        },
        "symptom_ood_metrics": {
            "f1": symptom_f1,
            "precision": symptom_metrics.get("precision", 0.0),
            "recall": symptom_metrics.get("recall", 0.0),
            "accuracy": symptom_metrics.get("accuracy", 0.0),
        },
        "ood_gaps": {
            "val_minus_template_ood": round(val_f1 - template_f1, 4),
            "val_minus_symptom_ood": round(val_f1 - symptom_f1, 4),
        },
    }

    summary_path = f"{OUTPUT_DIR}/summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to: {summary_path}")

    if SAVE_TO_GCS:
        print_title("UPLOADING RESULTS TO GCS")
        upload_success = upload_to_gcs(
            local_path=OUTPUT_DIR,
            gcs_path=OUTPUT_DIR,
            bucket_name=settings.BUCKET_NAME,
        )
        verify_success = verify_upload(
            bucket_name=settings.BUCKET_NAME,
            gcs_path=OUTPUT_DIR,
        )
        if upload_success and verify_success:
            print(f"All results uploaded to gs://{settings.BUCKET_NAME}/{OUTPUT_DIR}")
        else:
            print("Upload completed with errors. Check logs above.")


if __name__ == "__main__":
    from hyperparam_sets import biobert_hyperparams_v04

    idx = 0
    hyperparameters = biobert_hyperparams_v04[idx]
    train(hyperparameters, idx=idx)
