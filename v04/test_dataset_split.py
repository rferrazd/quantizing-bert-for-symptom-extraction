"""
test_dataset_split.py

Smoke tests for v04/dataset_split.py.

Covers:
  - split_template_groups: correct counts, determinism, no overlap, indices round-trip
  - split_symptoms_df: correct counts, determinism, no overlap
  - save_split_artifacts / load_split_artifacts: round-trip through JSON

Run with:
    python3 v04/test_dataset_split.py
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from v04.dataset_templates import TEMPLATE_GROUPS
from data_preparation.dataset_split import (
    EVAL_SPLIT_SEED,
    HELDOUT_COUNTS,
    HELDOUT_SYMPTOM_COUNT,
    load_split_artifacts,
    save_split_artifacts,
    split_symptoms_df,
    split_template_groups,
)


def section(title: str) -> None:
    """Print a visible section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def ok(msg: str) -> None:
    """Print a passing check."""
    print(f"  PASS  {msg}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYMPTOMS_DF = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "base_symptom_dict.csv")
)

# ---------------------------------------------------------------------------
# 1) split_template_groups — correct counts
# ---------------------------------------------------------------------------
section("1) split_template_groups — correct counts")

train, heldout, indices_map = split_template_groups(TEMPLATE_GROUPS)

for group_name, templates in TEMPLATE_GROUPS:
    n_held = HELDOUT_COUNTS.get(group_name, 0)
    train_entry  = next(t for n, t in train   if n == group_name)
    heldout_entry = next(t for n, t in heldout if n == group_name)

    assert len(heldout_entry) == n_held, (
        f"{group_name}: expected {n_held} held-out templates, got {len(heldout_entry)}"
    )
    assert len(train_entry) == len(templates) - n_held, (
        f"{group_name}: expected {len(templates) - n_held} train templates, got {len(train_entry)}"
    )
    ok(f"{group_name}: train={len(train_entry)}, heldout={len(heldout_entry)}")

# ---------------------------------------------------------------------------
# 2) split_template_groups — no overlap between train and heldout
# ---------------------------------------------------------------------------
section("2) split_template_groups — no overlap")

for group_name, _ in TEMPLATE_GROUPS:
    train_set   = set(next(t for n, t in train   if n == group_name))
    heldout_set = set(next(t for n, t in heldout if n == group_name))
    overlap = train_set & heldout_set
    assert not overlap, f"{group_name}: {len(overlap)} template(s) appear in both splits!"
    ok(f"{group_name}: no overlap")

# ---------------------------------------------------------------------------
# 3) split_template_groups — determinism
# ---------------------------------------------------------------------------
section("3) split_template_groups — determinism")

train2, heldout2, indices_map2 = split_template_groups(TEMPLATE_GROUPS)

for (n, t), (n2, t2) in zip(train, train2):
    assert t == t2, f"Non-deterministic train split for group {n!r}"
for (n, h), (n2, h2) in zip(heldout, heldout2):
    assert h == h2, f"Non-deterministic heldout split for group {n!r}"
assert indices_map == indices_map2, "Non-deterministic indices_map"
ok("Two runs with the same seed produce identical splits")

# ---------------------------------------------------------------------------
# 4) split_template_groups — indices are valid and consistent with template lists
# ---------------------------------------------------------------------------
section("4) split_template_groups — indices are consistent with heldout lists")

original = {name: tmpls for name, tmpls in TEMPLATE_GROUPS}

for group_name, heldout_tmpls in heldout:
    idxs = indices_map[group_name]
    assert len(idxs) == len(heldout_tmpls), (
        f"{group_name}: {len(idxs)} indices but {len(heldout_tmpls)} heldout templates"
    )
    # Each index in the map should point to the corresponding heldout template
    for idx, tmpl in zip(sorted(idxs), heldout_tmpls):
        assert original[group_name][idx] == tmpl, (
            f"{group_name} index {idx}: expected {original[group_name][idx]!r}, got {tmpl!r}"
        )
    ok(f"{group_name}: indices match heldout template list")

# ---------------------------------------------------------------------------
# 5) split_symptoms_df — correct counts
# ---------------------------------------------------------------------------
section("5) split_symptoms_df — correct counts")

train_s, heldout_s = split_symptoms_df(SYMPTOMS_DF)

assert len(heldout_s) == HELDOUT_SYMPTOM_COUNT, (
    f"Expected {HELDOUT_SYMPTOM_COUNT} held-out symptoms, got {len(heldout_s)}"
)
assert len(train_s) == len(SYMPTOMS_DF) - HELDOUT_SYMPTOM_COUNT, (
    f"Expected {len(SYMPTOMS_DF) - HELDOUT_SYMPTOM_COUNT} train symptoms, got {len(train_s)}"
)
ok(f"train={len(train_s)}, heldout={len(heldout_s)}")

# ---------------------------------------------------------------------------
# 6) split_symptoms_df — no overlap
# ---------------------------------------------------------------------------
section("6) split_symptoms_df — no overlap")

train_ids   = set(train_s["id"])
heldout_ids = set(heldout_s["id"])
assert train_ids.isdisjoint(heldout_ids), "Symptom IDs appear in both train and heldout!"
ok("No symptom id in both splits")

# ---------------------------------------------------------------------------
# 7) split_symptoms_df — determinism
# ---------------------------------------------------------------------------
section("7) split_symptoms_df — determinism")

train_s2, heldout_s2 = split_symptoms_df(SYMPTOMS_DF)
assert list(train_s["id"]) == list(train_s2["id"]),   "Non-deterministic train symptom split"
assert list(heldout_s["id"]) == list(heldout_s2["id"]), "Non-deterministic heldout symptom split"
ok("Two runs with the same seed produce identical splits")

# ---------------------------------------------------------------------------
# 8) save / load round-trip
# ---------------------------------------------------------------------------
section("8) save_split_artifacts / load_split_artifacts — round-trip")

with tempfile.TemporaryDirectory() as tmp:
    heldout_symptom_ids = heldout_s["id"].tolist()
    save_split_artifacts(indices_map, heldout_symptom_ids, tmp)

    artifact_path = os.path.join(tmp, "split_artifacts.json")
    assert os.path.exists(artifact_path), "Artifact file not created"
    ok("Artifact file created")

    loaded_indices, loaded_symptom_ids = load_split_artifacts(tmp)

    assert loaded_indices == indices_map, "Loaded indices_map does not match saved"
    ok("Loaded heldout_template_indices matches original")

    assert loaded_symptom_ids == heldout_symptom_ids, "Loaded symptom ids do not match saved"
    ok("Loaded heldout_symptom_ids matches original")

# ---------------------------------------------------------------------------
# 9) load_split_artifacts returns None when file absent
# ---------------------------------------------------------------------------
section("9) load_split_artifacts — returns None when no artifact exists")

with tempfile.TemporaryDirectory() as tmp:
    result = load_split_artifacts(tmp)
    assert result is None, f"Expected None, got {result!r}"
    ok("Returns None when split_artifacts.json is absent")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  ALL CHECKS PASSED")
print("=" * 60)
