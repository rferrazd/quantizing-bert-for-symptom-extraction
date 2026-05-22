"""
upload_to_hf.py

Carves a validation split from train_wordpiece.jsonl, then uploads all four
dataset splits (train, validation, template_ood, symptom_ood) plus the label
mapping JSON files to the HuggingFace Hub dataset repo defined in config.py.

Split strategy
--------------
- 90% of train_wordpiece.jsonl  -> "train" split on Hub
- 10% of train_wordpiece.jsonl  -> "validation" split on Hub
- template_ood_wordpiece.jsonl  -> "template_ood" split on Hub
- symptom_ood_wordpiece.jsonl   -> "symptom_ood" split on Hub

The 90/10 split is stratified by template_group so each group keeps its
proportional representation in both halves. Seed 42, consistent with all
other V04 splits.

Warning: the carve step overwrites train_wordpiece.jsonl with the 90% slice.
The script is idempotent — if validation_wordpiece.jsonl already exists the
carve is skipped. To start fresh, delete validation_wordpiece.jsonl and
re-run wordpiece_alignment.py to restore the full train file before running
this script again.

Run: /opt/anaconda3/bin/python3 v04/data/upload_to_hf.py
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from hf_utils import upsert_to_hf_repo

SPLITS_DIR  = PROJECT_ROOT / settings.VERSION / "data" / "splits"
LABELS_DIR  = PROJECT_ROOT / settings.VERSION / "data"
VAL_FRACTION = 0.10
SEED = 42

LABEL_FILES = ("label2id.json", "id2label.json")
SPLIT_FILES: list[tuple[str, str]] = [
    ("train",        "train_wordpiece.jsonl"),
    ("validation",   "validation_wordpiece.jsonl"),
    ("template_ood", "template_ood_wordpiece.jsonl"),
    ("symptom_ood",  "symptom_ood_wordpiece.jsonl"),
]


def load_jsonl(path: Path) -> list[dict]:
    """Load all non-empty lines from a JSONL file into a list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    """Write a list of dicts to a JSONL file, one JSON object per line."""
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_lines(path: Path) -> int:
    """Count non-empty lines in a file without loading it into memory."""
    with open(path) as f:
        return sum(1 for line in f if line.strip())


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
    for group_rows in groups.values():
        rng.shuffle(group_rows)
        n_val = max(1, round(len(group_rows) * val_fraction))
        val_rows.extend(group_rows[:n_val])
        train_rows.extend(group_rows[n_val:])

    return train_rows, val_rows


def carve_validation_split(train_path: Path, val_path: Path) -> None:
    """
    Carve 10% of train_wordpiece.jsonl into validation_wordpiece.jsonl.
    Skips if validation_wordpiece.jsonl already exists to prevent re-splitting
    an already-trimmed train file.
    """
    if val_path.exists():
        n_train = count_lines(train_path)
        n_val = count_lines(val_path)
        print(f"Validation split exists — skipping carve.")
        print(f"  train: {n_train:,}  |  validation: {n_val:,}")
        return

    print(f"Carving validation split from {train_path.name} ...")
    all_train = load_jsonl(train_path)
    print(f"  {len(all_train):,} samples loaded")

    train_rows, val_rows = stratified_split(all_train, VAL_FRACTION, SEED)
    save_jsonl(val_rows, val_path)
    save_jsonl(train_rows, train_path)
    print(f"  train: {len(train_rows):,}  |  validation: {len(val_rows):,}")


def upload_splits(repo_id: str, token: str) -> None:
    """Upload all four wordpiece-aligned JSONL splits to the HF dataset repo."""
    for split_name, filename in SPLIT_FILES:
        print(f"Uploading split: {split_name} ...")
        upsert_to_hf_repo(
            jsonl_path=SPLITS_DIR / filename,
            repo_id=repo_id,
            split_name=split_name,
            token=token,
        )


def upload_label_mappings(repo_id: str, token: str) -> None:
    """Upload label2id.json and id2label.json to the root of the HF dataset repo."""
    api = HfApi()
    for filename in LABEL_FILES:
        path = LABELS_DIR / filename
        print(f"Uploading {filename} ...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"  Done.")


def main() -> None:
    """Carve validation split, upload all splits, upload label mappings."""
    repo_id = settings.HUGGINGFACE_DATASET_REPO_ID
    token = settings.HF_TOKEN

    print(f"Target repo : {repo_id}")
    print(f"Splits dir  : {SPLITS_DIR}\n")

    # carve_validation_split(
    #     train_path=SPLITS_DIR / "train_wordpiece.jsonl",
    #     val_path=SPLITS_DIR / "validation_wordpiece.jsonl",
    # )
    # print()
    # upload_splits(repo_id, token)
    print()
    upload_label_mappings(repo_id, token)

    print(f"\nUpload complete.")
    print(f"https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
