"""
verify_wordpiece.py

Sanity-checks the three wordpiece-aligned JSONL files produced by
dataset_generator.py before uploading to HuggingFace or running training.

Checks performed
----------------
1. Sequence length consistency  — input_ids, token_label_ids, attention_mask
                                   must all have the same length per sample.
2. BIO continuity               — no I-X tag without a preceding B-X of the
                                   same type.
3. Label id range               — every token_label_id is -100 or in [0, 4].
4. Label distribution           — token-level counts per label across each split.
5. HDA symptom coverage (train) — how many of the 873 train symptoms actually
                                   appear in HDA samples (K=40 coverage check).

Run: /opt/anaconda3/bin/python3 v04/data/verify_wordpiece.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

VERSION_DIR = PROJECT_ROOT / settings.VERSION
SPLITS_DIR = VERSION_DIR / "data" / "splits"

SPLITS = {
    "train":        SPLITS_DIR / "train_wordpiece.jsonl",
    "template_ood": SPLITS_DIR / "template_ood_wordpiece.jsonl",
    "symptom_ood":  SPLITS_DIR / "symptom_ood_wordpiece.jsonl",
}

VALID_LABEL_IDS = {-100, 0, 1, 2, 3, 4}
ID2LABEL = {
    0: "O",
    1: "B-SYMPTOM_POS",
    2: "I-SYMPTOM_POS",
    3: "B-SYMPTOM_NEG",
    4: "I-SYMPTOM_NEG",
}

# Maps each I- type to the B- that must precede it.
REQUIRED_PREDECESSOR = {
    2: 1,  # I-SYMPTOM_POS must follow B-SYMPTOM_POS (1) or I-SYMPTOM_POS (2)
    4: 3,  # I-SYMPTOM_NEG must follow B-SYMPTOM_NEG (3) or I-SYMPTOM_NEG (4)
}


def load_jsonl(path: Path) -> list[dict]:
    """Load all lines from a JSONL file into a list of dicts."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def check_split(name: str, samples: list[dict]) -> tuple[bool, dict]:
    """
    Run all structural checks on one split. Returns (passed, stats) where
    stats contains label counts and symptom coverage data for the report.
    """
    length_errors = 0
    bio_errors = 0
    range_errors = 0
    label_counts: Counter = Counter()
    hda_symptom_ids: set[str] = set()
    all_symptom_ids: set[str] = set()

    for i, sample in enumerate(samples):
        ids  = sample["input_ids"]
        lids = sample["token_label_ids"]
        mask = sample["attention_mask"]

        # --- Check 1: sequence length consistency ---
        if not (len(ids) == len(lids) == len(mask)):
            length_errors += 1

        # --- Check 2: BIO continuity ---
        prev_lid = -100
        for lid in lids:
            if lid in REQUIRED_PREDECESSOR:
                required = REQUIRED_PREDECESSOR[lid]
                if prev_lid != required and prev_lid != lid:
                    bio_errors += 1
                    break
            prev_lid = lid

        # --- Check 3: label id range ---
        if any(lid not in VALID_LABEL_IDS for lid in lids):
            range_errors += 1

        # --- Check 4: label counts (skip -100 special tokens) ---
        for lid in lids:
            if lid != -100:
                label_counts[ID2LABEL[lid]] += 1

        # --- Check 5: symptom coverage (all splits, HDA group only for K check) ---
        for sym in sample.get("symptoms", []):
            sid = sym.get("symptom_id")
            if sid:
                all_symptom_ids.add(sid)
                if sample.get("template_group") == "hda":
                    hda_symptom_ids.add(sid)

    passed = (length_errors == 0 and bio_errors == 0 and range_errors == 0)
    stats = {
        "n_samples":      len(samples),
        "length_errors":  length_errors,
        "bio_errors":     bio_errors,
        "range_errors":   range_errors,
        "label_counts":   label_counts,
        "hda_symptom_ids": hda_symptom_ids,
        "all_symptom_ids": all_symptom_ids,
    }
    return passed, stats


def print_report(name: str, passed: bool, stats: dict, train_symptom_pool_size: int) -> None:
    """Print the formatted check report for one split."""
    status = "PASS" if passed else "FAIL"
    print(f"\n{'='*58}")
    print(f"  [{status}]  {name}  ({stats['n_samples']:,} samples)")
    print(f"{'='*58}")

    # Structural checks
    checks = [
        ("Sequence length consistency", stats["length_errors"]),
        ("BIO continuity",              stats["bio_errors"]),
        ("Label id range",              stats["range_errors"]),
    ]
    for check_name, n_errors in checks:
        mark = "OK" if n_errors == 0 else f"FAIL ({n_errors} samples)"
        print(f"  {check_name:<32} {mark}")

    # Label distribution
    total_tokens = sum(stats["label_counts"].values())
    print(f"\n  Label distribution ({total_tokens:,} labeled tokens):")
    for label in ["O", "B-SYMPTOM_POS", "I-SYMPTOM_POS", "B-SYMPTOM_NEG", "I-SYMPTOM_NEG"]:
        count = stats["label_counts"].get(label, 0)
        pct = 100 * count / total_tokens if total_tokens else 0
        print(f"    {label:<20} {count:>8,}  ({pct:5.1f}%)")

    # HDA symptom coverage (K=40 check) — only meaningful for train
    if name == "train" and stats["hda_symptom_ids"]:
        covered = len(stats["hda_symptom_ids"])
        pct = 100 * covered / train_symptom_pool_size if train_symptom_pool_size else 0
        print(f"\n  K=40 HDA symptom coverage:")
        print(f"    {covered} / {train_symptom_pool_size} train symptoms appear in HDA samples  ({pct:.1f}%)")

    # Overall symptom coverage
    print(f"\n  Unique symptoms seen (all groups): {len(stats['all_symptom_ids']):,}")


def spot_check(name: str, samples: list[dict], n: int = 3) -> None:
    """Print a word→subword→label table for n evenly-spaced samples."""
    print(f"\n{'─'*58}")
    print(f"  SPOT CHECK: {name} (showing {n} samples)")
    print(f"{'─'*58}")
    step = max(1, len(samples) // n)
    for i in range(0, len(samples), step)[:n]:
        s = samples[i]
        print(f"\n  Sample {i} | group={s.get('template_group','?')}")
        print(f"  Text: {s['text'][:80]}")
        print(f"  {'subword':<18} {'label_id':>8}  label")
        for tok, lid in zip(s["tokens"], s["token_label_ids"]):
            label = "SPECIAL" if lid == -100 else ID2LABEL[lid]
            print(f"    {tok:<18} {lid:>8}  {label}")


def main() -> None:
    """Run all checks and print a full report for every split."""
    # Load train symptoms pool size for coverage calculation
    train_symptoms_path = PROJECT_ROOT / "base_symptom_dict.csv"
    train_symptom_pool_size = 0
    if train_symptoms_path.exists():
        import csv
        with open(train_symptoms_path) as f:
            train_symptom_pool_size = sum(1 for _ in csv.reader(f)) - 1  # minus header

    print(f"Verifying wordpiece splits in: {SPLITS_DIR}")
    print(f"Train symptom pool size (base_symptom_dict.csv): {train_symptom_pool_size}")

    all_passed = True
    for split_name, path in SPLITS.items():
        if not path.exists():
            print(f"\n[MISSING] {split_name}: {path}")
            all_passed = False
            continue

        samples = load_jsonl(path)
        passed, stats = check_split(split_name, samples)
        print_report(split_name, passed, stats, train_symptom_pool_size)
        spot_check(split_name, samples, n=2)
        all_passed = all_passed and passed

    print(f"\n{'='*58}")
    print(f"  OVERALL: {'ALL CHECKS PASSED' if all_passed else 'FAILURES DETECTED — do not upload'}")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
