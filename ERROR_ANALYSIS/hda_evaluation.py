"""
hda_evaluation.py

Version-agnostic evaluation harness for the sample HDA hold-out set.

This is the canonical way to score ANY inference pipeline (PyTorch, ONNX, a
future V06 model, an experiment) against the 7 HDAs in sample_hdas.py. It reuses
the existing ErrorCategorizer so the error definitions stay identical to the
notebook that produced the V05 baseline of 8 errors.

The pipeline is injected as a `predict_fn(text) -> spans` callable, so this file
has no dependency on torch, onnx, or any specific model — callers wire that up.
"""

from typing import Callable, Dict, List

from error_analysis.error_categorization import ErrorCategorizer
from error_analysis.error_taxonomy import BehaviouralExample
from sample_hdas import transcriptions_and_hdas


# A prediction function takes raw text and returns a list of span dicts,
# each shaped {"start": int, "end": int, "text": str, "label": str}.
PredictFn = Callable[[str], List[Dict]]


def hda_to_behavioural_example(hda: dict) -> BehaviouralExample:
    """
    Convert one HDA dict from sample_hdas.py into a BehaviouralExample that
    ErrorCategorizer can score.

    The HDA stores entities grouped by label:
        {"SYMPTOM_POS": {"headache": [10, 18], ...}, "SYMPTOM_NEG": {...}}
    ErrorCategorizer wants a flat list of {ent, start, end, label} dicts.
    """
    entities_with_labels = []
    for label, items in hda["entities"].items():
        for phrase, (start, end) in items.items():
            entities_with_labels.append({
                "ent": phrase,
                "start": start,
                "end": end,
                "label": label,  # "SYMPTOM_POS" / "SYMPTOM_NEG"
            })
    return BehaviouralExample(
        example=hda["template"],
        entities_with_labels=entities_with_labels,
    )


def evaluate_on_hdas(predict_fn: PredictFn) -> Dict:
    """
    Run `predict_fn` over every HDA in sample_hdas.py and score it with
    ErrorCategorizer.

    Returns a report dict:
        {
            "total_errors": int,                  # sum of all error_counts
            "error_counts": dict,                 # error type -> count
            "per_hda": [                          # one entry per scored HDA
                {"index": int, "case": str,
                 "present": int, "missing": int, "false_positives": int,
                 "total": int},
                ...
            ],
        }

    `total_errors` is the headline KPI. With the V05 PyTorch pipeline it must
    equal 8 (the documented baseline in PROGRESS_NOTES/v05.md).
    """
    # One categorizer accumulates counts across all HDAs (same as the notebook).
    categorizer = ErrorCategorizer()
    categorizer.error_counts.clear()

    per_hda = []

    for i, hda in enumerate(transcriptions_and_hdas):
        # Some entries may be templates without gold entities — skip them.
        if "entities" not in hda:
            continue

        example = hda_to_behavioural_example(hda)

        # Inject whatever pipeline the caller passed in.
        spans = predict_fn(example.example)

        # Score this HDA. _check_entity_detection ALSO mutates
        # categorizer.error_counts as a side effect (that is the global total).
        errors = categorizer._check_entity_detection(example=example, spans=spans)

        n_present = len(errors["present_errors"])
        n_missing = len(errors["missing_entities"])
        n_fp = len(errors["false_positives"])

        per_hda.append({
            "index": i,
            "case": hda.get("case", f"hda_{i}"),
            "present": n_present,
            "missing": n_missing,
            "false_positives": n_fp,
            "total": n_present + n_missing + n_fp,
        })

    return {
        "total_errors": sum(categorizer.error_counts.values()),
        "error_counts": dict(categorizer.error_counts),
        "per_hda": per_hda,
    }
