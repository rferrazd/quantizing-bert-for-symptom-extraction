"""
dataset_split.py

Deterministic train / eval split for the v04 template groups and symptom pool.

Responsibilities:
  - Define how many templates to hold out per group (HELDOUT_COUNTS).
  - Define how many symptoms to hold out (HELDOUT_SYMPTOM_COUNT).
  - Provide split_template_groups(): splits a TemplateGroups list into
    (train_groups, heldout_groups) using a fixed seed for reproducibility.
  - Provide split_symptoms_df(): splits the symptoms DataFrame into
    (train_df, heldout_df) using the same seed contract.

All randomness goes through a local random.Random(seed) instance so the
global random state is never touched — callers can interleave other random
operations without disturbing the split.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

# TemplateGroups mirrors the type alias in dataset_generator.py:
# a list of (group_name, [template_str, ...]) pairs.
TemplateGroups = List[Tuple[str, List[str]]]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_SPLIT_SEED: int = 42

# Number of templates to hold out per group for the template-OOD eval set.
HELDOUT_COUNTS: dict[str, int] = {
    "affirmed":   5,
    "negated":    5,
    "distractor": 4,
    "hda":        2,
}

# Number of symptoms to hold out for the symptom-OOD eval set.
HELDOUT_SYMPTOM_COUNT: int = 20

_ARTIFACT_FILENAME = "split_artifacts.json"

# ---------------------------------------------------------------------------
# Template split
# ---------------------------------------------------------------------------

def split_template_groups(
    template_groups: TemplateGroups,
    seed: int = EVAL_SPLIT_SEED,
    counts: dict[str, int] = HELDOUT_COUNTS,
) -> Tuple[TemplateGroups, TemplateGroups, Dict[str, List[int]]]:
    """
    Split each group's template list into train and held-out subsets.

    For every (group_name, templates) pair in template_groups:
      - Draw `counts[group_name]` indices at random (without replacement).
      - Templates at those indices go to heldout_groups; the rest go to train_groups.
      - Groups whose name does not appear in counts are passed through to train unchanged
        (intentional default — safe for new groups added before HELDOUT_COUNTS is updated).

    Uses a seeded random.Random instance so the split is deterministic and does
    not affect the global random state.

    Args:
        template_groups: List of (group_name, [template, ...]) pairs — same
                         shape as TEMPLATE_GROUPS in dataset_templates.py.
        seed:            Integer seed for the local RNG. Default: EVAL_SPLIT_SEED.
        counts:          Mapping from group_name to number of templates to hold out.
                         Default: HELDOUT_COUNTS.

    Returns:
        (train_groups, heldout_groups, heldout_indices): the two TemplateGroups
        lists ready to pass to DatasetGenerator, plus a dict mapping each group
        name to the list of original indices that were held out (used by
        save_split_artifacts for auditing).
    """
    rng = random.Random(seed)

    train_groups: TemplateGroups = []
    heldout_groups: TemplateGroups = []
    heldout_indices_map: Dict[str, List[int]] = {}

    for group_name, templates in template_groups:
        n_heldout = counts.get(group_name, 0)

        if n_heldout == 0:
            # Group not in counts — goes entirely to train.
            train_groups.append((group_name, list(templates)))
            continue

        if n_heldout > len(templates):
            raise ValueError(
                f"Cannot hold out {n_heldout} templates from group {group_name!r}: "
                f"only {len(templates)} templates available."
            )

        # Pick held-out indices without replacement, then sort for stable output.
        heldout_indices = sorted(rng.sample(range(len(templates)), n_heldout))
        heldout_index_set = set(heldout_indices)

        train_tmpls: List[str] = []
        heldout_tmpls: List[str] = []
        for i, tmpl in enumerate(templates):
            if i in heldout_index_set:
                heldout_tmpls.append(tmpl)
            else:
                train_tmpls.append(tmpl)

        train_groups.append((group_name, train_tmpls))
        heldout_groups.append((group_name, heldout_tmpls))
        heldout_indices_map[group_name] = heldout_indices

    return train_groups, heldout_groups, heldout_indices_map


# ---------------------------------------------------------------------------
# Symptom split
# ---------------------------------------------------------------------------

def split_symptoms_df(
    symptoms_df: pd.DataFrame,
    n_heldout: int = HELDOUT_SYMPTOM_COUNT,
    seed: int = EVAL_SPLIT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the symptoms DataFrame into train and held-out subsets.

    Args:
        symptoms_df: Full symptoms pool (must have an 'id' column).
        n_heldout:   Number of symptoms to hold out. Default: HELDOUT_SYMPTOM_COUNT.
        seed:        Integer seed for the local RNG. Default: EVAL_SPLIT_SEED.

    Returns:
        (train_df, heldout_df): both are DataFrames with the same columns as
        symptoms_df, with reset indices.
    """
    if n_heldout > len(symptoms_df):
        raise ValueError(
            f"Cannot hold out {n_heldout} symptoms: pool has only {len(symptoms_df)} rows."
        )

    rng = random.Random(seed)
    all_indices = list(range(len(symptoms_df)))
    heldout_indices = set(rng.sample(all_indices, n_heldout))

    train_mask = [i not in heldout_indices for i in all_indices]
    heldout_mask = [i in heldout_indices for i in all_indices]

    train_df = symptoms_df[train_mask].reset_index(drop=True)
    heldout_df = symptoms_df[heldout_mask].reset_index(drop=True)

    return train_df, heldout_df


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_split_artifacts(
    heldout_template_indices: Dict[str, List[int]],
    heldout_symptom_ids: List[str],
    out_dir: str,
) -> None:
    """
    Persist the split decisions to disk so they can be audited and reloaded.

    Writes a single JSON file (<out_dir>/split_artifacts.json) containing:
      - seed: the constant used to produce this split (for documentation only;
              rerunning split_template_groups / split_symptoms_df with the same
              seed always produces the same result even without the file).
      - heldout_template_indices: {group_name: [idx, ...]} from split_template_groups.
      - heldout_symptom_ids: list of symptom id strings from split_symptoms_df.

    Args:
        heldout_template_indices: Third return value of split_template_groups.
        heldout_symptom_ids:      List of id strings, e.g. heldout_df["id"].tolist().
        out_dir:                  Directory to write the artifact file into.
                                  Created if it does not exist.
    """
    os.makedirs(out_dir, exist_ok=True)
    artifact = {
        "seed": EVAL_SPLIT_SEED,
        "heldout_template_indices": heldout_template_indices,
        "heldout_symptom_ids": heldout_symptom_ids,
    }
    path = os.path.join(out_dir, _ARTIFACT_FILENAME)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)


def load_split_artifacts(
    out_dir: str,
) -> Optional[Tuple[Dict[str, List[int]], List[str]]]:
    """
    Load a previously saved split artifact from disk.

    Args:
        out_dir: Directory that was passed to save_split_artifacts.

    Returns:
        (heldout_template_indices, heldout_symptom_ids) if the artifact file
        exists, or None if it has not been created yet.
    """
    path = os.path.join(out_dir, _ARTIFACT_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        artifact = json.load(f)
    return artifact["heldout_template_indices"], artifact["heldout_symptom_ids"]
