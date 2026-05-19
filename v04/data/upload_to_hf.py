"""
upload_to_hf.py

Carves a validation split from train_wordpiece.jsonl, then uploads all four
dataset splits (train, validation, template_ood, symptom_ood) to the
HuggingFace Hub dataset repo defined in config.py.

Split strategy
--------------
- 90% of train_wordpiece.jsonl  -> "train" split on Hub
- 10% of train_wordpiece.jsonl  -> "validation" split on Hub
- template_ood_wordpiece.jsonl  -> "template_ood" split on Hub
- symptom_ood_wordpiece.jsonl   -> "symptom_ood" split on Hub

The 90/10 split is stratified by template_group so each group keeps its
proportional representation in both train and validation.

One warning: step 2 overwrites the local train_wordpiece.jsonl. 
If you ever need to regenerate from scratch, re-run dataset_generator.py followed by wordpiece_alignment.py — do not re-run upload_to_hf.py twice expecting to re-split the already-trimmed file.

Seed: 42 (consistent with all other V04 splits).

Run: /opt/anaconda3/bin/python3 v04/data/upload_to_hf.py
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from hf_utils import upsert_to_hf_repo

SPLITS_DIR = PROJECT_ROOT / settings.VERSION / "data" / "splits"

VAL_FRACTION = 0.10
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    """Load all lines from a JSONL file into a list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(
    rows: list[dict],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Split rows into (train, validation) stratified by template_group so each
    group keeps its proportional representation in both halves.

    Returns (train_rows, val_rows).
    """
    rng = random.Random(seed)

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get("template_group", "unknown")].append(row)

    train_rows, val_rows = [], []
    for group_name, group_rows in groups.items():
        rng.shuffle(group_rows)
        n_val = max(1, round(len(group_rows) * val_fraction))
        val_rows.extend(group_rows[:n_val])
        train_rows.extend(group_rows[n_val:])

    return train_rows, val_rows


def main() -> None:
    """Carve the validation split, save locally, then upload all four splits."""
    repo_id = settings.HUGGINGFACE_DATASET_REPO_ID
    token = settings.HF_TOKEN

    print(f"Target repo: {repo_id}")
    print(f"Splits dir:  {SPLITS_DIR}\n")

    # ------------------------------------------------------------------
    # 1. Carve train / validation from train_wordpiece.jsonl
    # ------------------------------------------------------------------
    train_path = SPLITS_DIR / "train_wordpiece.jsonl"
    val_path = SPLITS_DIR / "validation_wordpiece.jsonl"

    if val_path.exists():
        # Split already done — reusing existing files to avoid re-splitting a
        # previously trimmed train_wordpiece.jsonl.
        n_train = sum(1 for l in open(train_path) if l.strip())
        n_val = sum(1 for l in open(val_path) if l.strip())
        print(f"  Validation split already exists — skipping carve.")
        print(f"  train: {n_train:,}  |  validation: {n_val:,}\n")
    else:
        print(f"Loading {train_path.name} ...")
        all_train = load_jsonl(train_path)
        print(f"  {len(all_train):,} samples total")

        train_rows, val_rows = stratified_split(all_train, VAL_FRACTION, SEED)
        print(f"  -> train: {len(train_rows):,}  |  validation: {len(val_rows):,}")

        save_jsonl(val_rows, val_path)
        print(f"  Validation split saved to {val_path.name}")

        save_jsonl(train_rows, train_path)
        print(f"  train_wordpiece.jsonl updated to {len(train_rows):,} samples\n")

    # ------------------------------------------------------------------
    # 2. Upload all four splits
    # ------------------------------------------------------------------
    uploads = [
        ("train",        train_path),
        ("validation",   val_path),
        ("template_ood", SPLITS_DIR / "template_ood_wordpiece.jsonl"),
        ("symptom_ood",  SPLITS_DIR / "symptom_ood_wordpiece.jsonl"),
    ]

    for split_name, path in uploads:
        print(f"--- Uploading split: {split_name} ---")
        upsert_to_hf_repo(
            jsonl_path=path,
            repo_id=repo_id,
            split_name=split_name,
            token=token,
        )
        print()

    print("All splits uploaded.")
    print(f"Dataset: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
