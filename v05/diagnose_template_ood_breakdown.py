"""
diagnose_template_ood_breakdown.py

V05 template_ood F1 collapsed to 0.683 (vs V04 0.869). The hypothesis from the
split-construction audit (step 2) is that this is a holdout-composition artifact:
the seed=42 affirmed holdout landed on 2 of the 8 brand-new V05 P3 modifier
templates (indices 40 and 47), and those V05-new templates are intrinsically
harder than V04-era templates.

This script tests that hypothesis by computing F1:
  1. Per `template_group` on the template_ood split.
  2. Within `affirmed`, partitioning by whether the source template is a V05 P3
     modifier template ("sudden onset of {SYMPTOM_POS}" or "persistent
     {SYMPTOM_POS} that has not improved") vs a V04-era affirmed template.

If the F1 collapse is concentrated on the 2 P3 templates and V04-era affirmed
templates score >= 0.97, the holdout-composition hypothesis is confirmed.

The 5 held-out affirmed templates (per v05/data/splits/split_artifacts.json):
  idx 1  — "{SYMPTOM_POS} was documented during evaluation."          [V04]
  idx 7  — "Reports ongoing {SYMPTOM_POS} since yesterday."           [V04]
  idx 17 — "Presentation notable for {SYMPTOM_POS} and additional findings."   [V04]
  idx 40 — "Patient reports sudden onset of {SYMPTOM_POS}."           [V05 P3]
  idx 47 — "Acknowledges persistent {SYMPTOM_POS} that has not improved." [V05 P3]

Run: python v05/diagnose_template_ood_breakdown.py
"""

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from config import settings
from metrics import compute_metrics


# --- Identify the V05 P3 modifier templates by their distinctive prefixes ----
# Both P3 templates that landed in the holdout, plus the regex patterns we'll
# use to classify each affirmed template_ood sample by template origin.
P3_HELDOUT_TEMPLATES = [
    # idx 40
    "Patient reports sudden onset of {SYMPTOM_POS}.",
    # idx 47
    "Acknowledges persistent {SYMPTOM_POS} that has not improved.",
]

# Compile to text-matching regexes by escaping then replacing the placeholder.
def template_to_regex(template: str) -> re.Pattern:
    """Convert '{SYMPTOM_POS}' templates into a regex that matches any filled text."""
    escaped = re.escape(template).replace(r"\{SYMPTOM_POS\}", r".+?")
    return re.compile(f"^{escaped}$")


P3_REGEXES = [template_to_regex(t) for t in P3_HELDOUT_TEMPLATES]


def is_p3_modifier_text(text: str) -> bool:
    """True if the text was generated from a V05 P3 modifier template."""
    return any(rx.match(text) for rx in P3_REGEXES)


# --- Load model + tokenizer + dataset ---------------------------------------
LOCAL_MODEL_DIR = (
    f"{PROJECT_ROOT}/v05/downloaded_models/dmis-lab/biobert-base-cased-v1.1/run_0"
)
if not os.path.isdir(LOCAL_MODEL_DIR):
    raise FileNotFoundError(
        f"Model not found at {LOCAL_MODEL_DIR}. Run the notebook download cell first."
    )

print(f"Loading model from {LOCAL_MODEL_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(LOCAL_MODEL_DIR)
id2label = model.config.id2label
# Trainer.predict expects a string-keyed dict for compute_metrics → enforce it.
id2label_str = {str(k): v for k, v in id2label.items()}

print(f"Loading template_ood split from {settings.HUGGINGFACE_DATASET_REPO_ID} ...")
dataset = load_dataset(
    settings.HUGGINGFACE_DATASET_REPO_ID,
    split="template_ood",
    token=settings.HF_TOKEN,
)
dataset = dataset.rename_column("token_label_ids", "labels")
print(f"  template_ood: {len(dataset):,} samples")
print(f"  template_group counts: {dict((g, sum(1 for x in dataset['template_group'] if x == g)) for g in set(dataset['template_group']))}")

# --- Build the partitions ---------------------------------------------------
# Five partitions:
#   1. affirmed_v04      — affirmed samples whose source template is V04-era (not P3)
#   2. affirmed_v05_p3   — affirmed samples whose source is a V05 P3 modifier template
#   3. negated           — all negated samples
#   4. distractor        — all distractor samples
#   5. hda               — all HDA samples
partitions: dict[str, list[int]] = {
    "affirmed_v04": [],
    "affirmed_v05_p3": [],
    "negated": [],
    "distractor": [],
    "hda": [],
}

for i, row in enumerate(dataset):
    group = row["template_group"]
    if group == "affirmed":
        bucket = "affirmed_v05_p3" if is_p3_modifier_text(row["text"]) else "affirmed_v04"
    else:
        bucket = group
    partitions[bucket].append(i)

print("\nPartition sizes:")
for name, idxs in partitions.items():
    print(f"  {name:20s}: {len(idxs):>6,} samples")

# --- Run predictions once on the full split, then slice predictions ----------
# This is cheaper than running Trainer.predict five times.
print("\nRunning predictions on full template_ood split ...")
# remove_unused_columns drops `text`, `tokens`, `symptoms` etc. before collation.
# DataCollatorForTokenClassification handles padding of variable-length input_ids/labels.
training_args = TrainingArguments(
    output_dir="/tmp/diagnose_template_ood",
    per_device_eval_batch_size=32,
    remove_unused_columns=True,
    report_to=[],
    disable_tqdm=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=lambda ep: compute_metrics(ep, id2label=id2label_str),
)

# Trainer.predict on the full set returns predictions aligned to dataset order.
full_pred = trainer.predict(test_dataset=dataset)
print(f"  predictions shape: {full_pred.predictions.shape}")
print(f"  label_ids shape:   {full_pred.label_ids.shape}")

# --- Compute per-partition metrics by manually slicing the prediction arrays -
import numpy as np
from transformers.trainer_utils import PredictionOutput


def slice_prediction(pred: PredictionOutput, idxs: list[int]) -> PredictionOutput:
    """Return a PredictionOutput restricted to the given row indices."""
    return PredictionOutput(
        predictions=pred.predictions[idxs],
        label_ids=pred.label_ids[idxs],
        metrics=None,
    )


print("\n" + "=" * 70)
print(f"  {'partition':<22} {'samples':>8}  {'F1':>8}  {'precision':>10}  {'recall':>8}")
print("=" * 70)

results: dict[str, dict] = {}
for name, idxs in partitions.items():
    if not idxs:
        print(f"  {name:<22} {0:>8}  (empty partition — skipped)")
        continue
    sliced = slice_prediction(full_pred, idxs)
    m = compute_metrics(sliced, id2label=id2label_str)
    results[name] = m
    print(
        f"  {name:<22} {len(idxs):>8,}  "
        f"{m['f1']:>8.4f}  {m['precision']:>10.4f}  {m['recall']:>8.4f}"
    )

print("=" * 70)

# Also compute the global metric for sanity (should match the training summary 0.6832).
m_all = compute_metrics(full_pred, id2label=id2label_str)
print(
    f"  {'TOTAL (sanity)':<22} {len(dataset):>8,}  "
    f"{m_all['f1']:>8.4f}  {m_all['precision']:>10.4f}  {m_all['recall']:>8.4f}"
)
print("=" * 70)

# --- Save results -----------------------------------------------------------
import json

out_path = PROJECT_ROOT / "v05" / "template_ood_breakdown.json"
out = {
    "partition_sizes": {k: len(v) for k, v in partitions.items()},
    "partition_metrics": results,
    "overall_metric_sanity_check": m_all,
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved breakdown to {out_path}")

# --- Print verdict ----------------------------------------------------------
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
v04_aff_f1 = results.get("affirmed_v04", {}).get("f1", 0.0)
p3_aff_f1 = results.get("affirmed_v05_p3", {}).get("f1", 0.0)
print(f"  affirmed_v04    F1: {v04_aff_f1:.4f}  (threshold for confirmation: >= 0.97)")
print(f"  affirmed_v05_p3 F1: {p3_aff_f1:.4f}")
print(f"  delta:              {v04_aff_f1 - p3_aff_f1:+.4f}")
if v04_aff_f1 >= 0.97 and p3_aff_f1 < v04_aff_f1 - 0.10:
    print("\n  ✅ HOLDOUT-COMPOSITION HYPOTHESIS CONFIRMED")
    print("  V04-era templates generalize cleanly; the P3 templates dominate the F1 drop.")
elif v04_aff_f1 < 0.95:
    print("\n  ❌ HYPOTHESIS REFUTED — V04-era templates also regressed.")
    print("  The template_ood F1 collapse is a real generalization regression, not just holdout composition.")
else:
    print("\n  ⚠️  PARTIAL — V04-era templates score reasonably but P3 isn't dominant. Mixed cause.")
