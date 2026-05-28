"""
validate_onnx.py

Validates the ONNX-exported V05 models against the PyTorch checkpoint baseline
using the 7-HDA hold-out set in sample_hdas.py.

Runs three pipelines side by side:
    1. PyTorch fp32 (ground truth — must score 8 errors, the documented baseline)
    2. ONNX fp32   (sanity check — should match PyTorch almost exactly)
    3. ONNX INT8   (the model we want to ship — acceptance criterion: <= 10 errors)

Prints:
    - A summary table with total error counts per pipeline
    - A per-HDA breakdown
    - For any HDA where ONNX INT8 differs from PyTorch, the exact span diff

Usage (from project root):
    python -m mobile_app.model_prep.validate_onnx
"""

import json
from typing import Callable, List, Dict

import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForTokenClassification

from inference.v01.inference_utils import predict_word_level, word_labels_to_spans
from mobile_app.model_prep.onnx_inference import predict_word_level_onnx
from error_analysis.hda_evaluation import evaluate_on_hdas
from sample_hdas import transcriptions_and_hdas


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CKPT = "v05/downloaded_models/dmis-lab/biobert-base-cased-v1.1/run_0"
ART = "mobile_app/artifacts/v05"
FP32_ONNX = f"{ART}/model.onnx"
INT8_ONNX = f"{ART}/model_int8.onnx"


# ---------------------------------------------------------------------------
# Build predict_fn wrappers
# Each one takes raw text and returns a list of span dicts.
# ---------------------------------------------------------------------------

def build_pytorch_predict_fn(ckpt_path: str, tok) -> Callable[[str], List[Dict]]:
    """
    Load the PyTorch checkpoint and return a predict_fn(text) -> spans callable.
    This is the ground-truth pipeline; must reproduce the documented 8-error baseline.
    """
    model = AutoModelForTokenClassification.from_pretrained(ckpt_path).eval()
    id2label = model.config.id2label

    def predict(text: str) -> List[Dict]:
        """Run PyTorch inference and return word-level entity spans."""
        *_, word_labels, word_offsets = predict_word_level(
            text, model, tok, id2label, "cpu"
        )
        return word_labels_to_spans(text, word_offsets, word_labels)

    return predict


def build_onnx_predict_fn(model_path: str, tok, id2label: dict) -> Callable[[str], List[Dict]]:
    """
    Load an ONNX model from `model_path` and return a predict_fn(text) -> spans callable.
    Works for both fp32 and INT8 ONNX models.
    """
    session = ort.InferenceSession(model_path)

    def predict(text: str) -> List[Dict]:
        """Run ONNX inference and return word-level entity spans."""
        *_, word_labels, word_offsets = predict_word_level_onnx(
            text, session, tok, id2label
        )
        return word_labels_to_spans(text, word_offsets, word_labels)

    return predict


# ---------------------------------------------------------------------------
# Span diff helper
# ---------------------------------------------------------------------------

def diff_spans(
    text: str,
    spans_a: List[Dict],
    spans_b: List[Dict],
    label_a: str = "A",
    label_b: str = "B",
) -> List[str]:
    """
    Return a list of human-readable lines describing entity spans that differ
    between two predictions on the same text. Only non-O spans are compared.
    Used to show exactly which entities changed between PyTorch and ONNX INT8.
    """
    def entity_set(spans: List[Dict]) -> set:
        return {
            (s["start"], s["end"], s["text"], s["label"])
            for s in spans
            if s["label"] != "O"
        }

    only_in_a = entity_set(spans_a) - entity_set(spans_b)
    only_in_b = entity_set(spans_b) - entity_set(spans_a)

    lines = []
    for start, end, txt, lbl in sorted(only_in_a):
        lines.append(f"    [{label_a} only]  '{txt}'  ({lbl})  chars {start}-{end}")
    for start, end, txt, lbl in sorted(only_in_b):
        lines.append(f"    [{label_b} only]  '{txt}'  ({lbl})  chars {start}-{end}")
    return lines


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_summary_table(reports: Dict[str, Dict]) -> None:
    """Print a one-line-per-pipeline summary with total errors and acceptance status."""
    print("\n" + "=" * 60)
    print(f"{'Pipeline':<18} {'Total errors':>13}  {'Status'}")
    print("-" * 60)
    for name, rep in reports.items():
        total = rep["total_errors"]
        if name == "PyTorch fp32":
            status = "✅ baseline" if total == 8 else f"❌ expected 8, got {total}"
        else:
            status = "✅ PASS (≤ 10)" if total <= 10 else f"❌ FAIL (> 10)"
        print(f"  {name:<16} {total:>13}  {status}")
    print("=" * 60)


def print_per_hda_table(reports: Dict[str, Dict]) -> None:
    """Print per-HDA error counts for every pipeline, side by side."""
    names = list(reports.keys())
    hdas = reports[names[0]]["per_hda"]

    col_w = 12
    header = f"  {'HDA':<18}" + "".join(f"{n:>{col_w}}" for n in names)
    print("\n" + header)
    print("  " + "-" * (18 + col_w * len(names)))
    for i, hda_row in enumerate(hdas):
        case = hda_row["case"]
        row = f"  {case:<18}"
        for name in names:
            total = reports[name]["per_hda"][i]["total"]
            row += f"{total:>{col_w}}"
        print(row)


def print_int8_diffs(predict_pt: Callable, predict_int8: Callable) -> None:
    """
    For each HDA where INT8 differs from PyTorch, print the exact span diff
    so we can see what quantization noise introduced or removed.
    """
    print("\n--- INT8 vs PyTorch span diffs (only HDAs that differ) ---")
    found_any = False
    for i, hda in enumerate(transcriptions_and_hdas):
        if "entities" not in hda:
            continue
        text = hda["template"]
        spans_pt = predict_pt(text)
        spans_int8 = predict_int8(text)

        diff_lines = diff_spans(text, spans_pt, spans_int8, "PyTorch", "INT8")
        if diff_lines:
            found_any = True
            print(f"\n  HDA {i} — {hda.get('case', '')}:")
            for line in diff_lines:
                print(line)
    if not found_any:
        print("  (no differences — INT8 matches PyTorch exactly)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Load all three pipelines, evaluate, and print the validation report."""

    print("Loading tokenizer and models...")
    tok = AutoTokenizer.from_pretrained(CKPT)

    # id2label for ONNX pipelines (keys are str-ints in config.json -> cast to int)
    id2label_onnx = {
        int(k): v
        for k, v in json.load(open(f"{ART}/config.json"))["id2label"].items()
    }

    predict_pt    = build_pytorch_predict_fn(CKPT, tok)
    predict_fp32  = build_onnx_predict_fn(FP32_ONNX, tok, id2label_onnx)
    predict_int8  = build_onnx_predict_fn(INT8_ONNX, tok, id2label_onnx)

    print("Running evaluation on 7 HDAs (this takes ~30s on CPU)...")
    reports = {
        "PyTorch fp32": evaluate_on_hdas(predict_pt),
        "ONNX fp32":    evaluate_on_hdas(predict_fp32),
        "ONNX INT8":    evaluate_on_hdas(predict_int8),
    }

    print_summary_table(reports)
    print_per_hda_table(reports)
    print_int8_diffs(predict_pt, predict_int8)

    # Error type breakdown per pipeline
    print("\n--- Error type breakdown ---")
    for name, rep in reports.items():
        print(f"\n  {name}:")
        for err_type, count in sorted(rep["error_counts"].items(), key=lambda x: -x[1]):
            print(f"    {count:>3}  {err_type}")


if __name__ == "__main__":
    main()
