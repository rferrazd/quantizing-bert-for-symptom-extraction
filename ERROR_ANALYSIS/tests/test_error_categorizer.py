"""Tests for error_analysis.error_categorization.

Three layers:
  1. _is_entity_in_spans  — positional matcher (replaces old text-substring match)
  2. _check_boundaries    — overreach / undereach / <50%-overlap routing
  3. _check_entity_detection — end-to-end integration over each taxonomy bucket

TODO: type_2 / type_4 (CONFLICT label paths) — pending real CONFLICT examples
      from the inference pipeline.
"""

import pytest

from error_analysis.error_categorization import ErrorCategorizer
from error_analysis.error_taxonomy import ErrorTaxonomy, BehaviouralExample


@pytest.fixture
def ec():
    return ErrorCategorizer()


# ---------------------------------------------------------------------------
# Layer 1: _is_entity_in_spans
# ---------------------------------------------------------------------------

class TestIsEntityInSpans:

    def test_clean_match(self, ec):
        entity = {"ent": "fever", "start": 3, "end": 8, "label": "SYMPTOM_POS"}
        spans = [{"text": "No fever", "start": 0, "end": 8, "label": "SYMPTOM_NEG"}]
        assert ec._is_entity_in_spans(entity, spans) == 0

    def test_weak_overlap_still_matches(self, ec):
        # 1 char of overlap is enough — boundary check downstream will flag it
        entity = {"ent": "fever", "start": 3, "end": 8, "label": "SYMPTOM_NEG"}
        spans = [{"text": "No f", "start": 0, "end": 4, "label": "SYMPTOM_NEG"}]
        assert ec._is_entity_in_spans(entity, spans) == 0

    def test_same_text_different_position_does_not_match(self, ec):
        # The repeated-word case: "Sneezing ... patient denies sneezing".
        # Old text-match would (wrongly) return 0. Positional match returns None.
        entity = {"ent": "sneezing", "start": 64, "end": 72, "label": "SYMPTOM_NEG"}
        spans = [
            {"text": "Sneezing", "start": 0, "end": 8, "label": "SYMPTOM_POS"},
        ]
        assert ec._is_entity_in_spans(entity, spans) is None

    def test_picks_largest_overlap(self, ec):
        entity = {"ent": "fever", "start": 10, "end": 15, "label": "SYMPTOM_POS"}
        spans = [
            {"text": "fever",      "start": 0,  "end": 5,  "label": "SYMPTOM_POS"},  # 0
            {"text": "high fever", "start": 5,  "end": 15, "label": "SYMPTOM_POS"},  # 5 ← best
            {"text": "fever",      "start": 14, "end": 19, "label": "SYMPTOM_POS"},  # 1
        ]
        assert ec._is_entity_in_spans(entity, spans) == 1

    def test_skips_O_label(self, ec):
        entity = {"ent": "fever", "start": 3, "end": 8, "label": "SYMPTOM_POS"}
        spans = [
            {"text": "No fever", "start": 0, "end": 8, "label": "O"},
            {"text": "fever",    "start": 3, "end": 8, "label": "SYMPTOM_POS"},
        ]
        assert ec._is_entity_in_spans(entity, spans) == 1

    def test_empty_spans(self, ec):
        entity = {"ent": "fever", "start": 3, "end": 8, "label": "SYMPTOM_POS"}
        assert ec._is_entity_in_spans(entity, []) is None


# ---------------------------------------------------------------------------
# Layer 2: _check_boundaries
# ---------------------------------------------------------------------------

class TestCheckBoundaries:

    def test_perfect_match_returns_none(self, ec):
        assert ec._check_boundaries(10, 15, 10, 15) is None

    def test_overreach_predicted_longer(self, ec):
        # gold 10–15, predicted 5–20  → predicted swallows gold
        assert ec._check_boundaries(10, 15, 5, 20) == ErrorTaxonomy.type_5

    def test_undereach_predicted_contained(self, ec):
        # gold 10–20, predicted 12–18 → predicted strictly inside gold
        assert ec._check_boundaries(10, 20, 12, 18) == ErrorTaxonomy.type_6

    def test_low_overlap_longer_routes_to_overreach(self, ec):
        # gold 3–8 (len 5), predicted 0–6 changed to 0–4 to force <50% overlap;
        # but we want longer-predicted variant: gold 10–15 (len 5), predicted 0–12 (len 12)
        # overlap = min(15,12)-max(10,0)=2 < 2.5 → routed by length
        assert ec._check_boundaries(10, 15, 0, 12) == ErrorTaxonomy.type_5

    def test_low_overlap_shorter_routes_to_undereach(self, ec):
        # gold 3–8 (len 5), predicted 0–4 (len 4): overlap=1 < 2.5, predicted shorter
        assert ec._check_boundaries(3, 8, 0, 4) == ErrorTaxonomy.type_6


# ---------------------------------------------------------------------------
# Layer 3: _check_entity_detection (integration)
# ---------------------------------------------------------------------------

class TestCheckEntityDetection:

    def test_perfect_detection_no_errors(self, ec):
        example = BehaviouralExample(
            example="The patient has fever.",
            entities_with_labels=[
                {"ent": "fever", "start": 16, "end": 21, "label": "SYMPTOM_POS"},
            ],
        )
        spans = [{"text": "fever", "start": 16, "end": 21, "label": "SYMPTOM_POS"}]
        result = ec._check_entity_detection(example=example, spans=spans)
        assert result["present_errors"] == []
        assert result["missing_entities"] == []
        assert result["false_positives"] == []
        assert dict(ec.error_counts) == {}

    def test_type_1_missing_entity(self, ec):
        # Gold entity not detected at all (no overlapping span).
        example = BehaviouralExample(
            example="The patient has fever.",
            entities_with_labels=[
                {"ent": "fever", "start": 16, "end": 21, "label": "SYMPTOM_POS"},
            ],
        )
        spans = []
        result = ec._check_entity_detection(example=example, spans=spans)
        assert len(result["missing_entities"]) == 1
        assert result["missing_entities"][0]["errors"] == [ErrorTaxonomy.type_1]
        assert ec.error_counts[ErrorTaxonomy.type_1] == 1

    def test_type_3_polarity_flip(self, ec):
        # Same span, wrong polarity (POS gold, NEG predicted).
        example = BehaviouralExample(
            example="The patient has fever.",
            entities_with_labels=[
                {"ent": "fever", "start": 16, "end": 21, "label": "SYMPTOM_POS"},
            ],
        )
        spans = [{"text": "fever", "start": 16, "end": 21, "label": "SYMPTOM_NEG"}]
        result = ec._check_entity_detection(example=example, spans=spans)
        assert len(result["present_errors"]) == 1
        assert ErrorTaxonomy.type_3 in result["present_errors"][0]["errors"]
        assert ec.error_counts[ErrorTaxonomy.type_3] == 1

    def test_type_5_overreach(self, ec):
        example = BehaviouralExample(
            example="No fever reported today.",
            entities_with_labels=[
                {"ent": "fever", "start": 3, "end": 8, "label": "SYMPTOM_NEG"},
            ],
        )
        # Predicted span swallows "No fever" but >50% overlaps gold
        spans = [{"text": "No fever", "start": 0, "end": 8, "label": "SYMPTOM_NEG"}]
        result = ec._check_entity_detection(example=example, spans=spans)
        assert len(result["present_errors"]) == 1
        assert ErrorTaxonomy.type_5 in result["present_errors"][0]["errors"]
        assert ec.error_counts[ErrorTaxonomy.type_5] == 1

    def test_type_6_undereach(self, ec):
        example = BehaviouralExample(
            example="The patient has high fever today.",
            entities_with_labels=[
                {"ent": "high fever", "start": 16, "end": 26, "label": "SYMPTOM_POS"},
            ],
        )
        # Predicted catches only "fever" (subset of gold)
        spans = [{"text": "fever", "start": 21, "end": 26, "label": "SYMPTOM_POS"}]
        result = ec._check_entity_detection(example=example, spans=spans)
        assert len(result["present_errors"]) == 1
        assert ErrorTaxonomy.type_6 in result["present_errors"][0]["errors"]
        assert ec.error_counts[ErrorTaxonomy.type_6] == 1

    def test_false_positive_single_type_0(self, ec):
        # Predicted entity with no gold counterpart → type_0 only (no double-count).
        example = BehaviouralExample(
            example="The sky is blue.",
            entities_with_labels=[],
        )
        spans = [{"text": "blue", "start": 11, "end": 15, "label": "SYMPTOM_POS"}]
        result = ec._check_entity_detection(example=example, spans=spans)
        assert len(result["false_positives"]) == 1
        assert result["false_positives"][0]["errors"] == [ErrorTaxonomy.type_0]
        assert ec.error_counts[ErrorTaxonomy.type_0] == 1
        assert ErrorTaxonomy.type_7 not in ec.error_counts

    def test_previously_silent_case_now_reported(self, ec):
        # Regression test: "No fever" at 0–4, gold "fever" at 3–8.
        # Before fix: zero errors anywhere. After fix: should appear *somewhere*.
        example = BehaviouralExample(
            example="No fever reported.",
            entities_with_labels=[
                {"ent": "fever", "start": 3, "end": 8, "label": "SYMPTOM_NEG"},
            ],
        )
        spans = [{"text": "No f", "start": 0, "end": 4, "label": "SYMPTOM_NEG"}]
        result = ec._check_entity_detection(example=example, spans=spans)
        total = (
            len(result["present_errors"])
            + len(result["missing_entities"])
            + len(result["false_positives"])
        )
        assert total > 0, "Silent-consumption regression: low-overlap case produced no error"
        assert len(ec.error_counts) > 0

    def test_repeated_word_distinguished_by_position(self, ec):
        # The user's motivating example: gold negated "sneezing" is the *second*
        # occurrence; model predicts on the *first* (positive context).
        # Positional matcher should NOT match those — gold becomes missing (type_1)
        # and predicted becomes a false positive (type_0).
        text = "Sneezing is very common during winter but the patient denies sneezing."
        gold_start = text.rindex("sneezing")
        gold_end = gold_start + len("sneezing")
        example = BehaviouralExample(
            example=text,
            entities_with_labels=[
                {"ent": "sneezing", "start": gold_start, "end": gold_end, "label": "SYMPTOM_NEG"},
            ],
        )
        spans = [{"text": "Sneezing", "start": 0, "end": 8, "label": "SYMPTOM_POS"}]
        result = ec._check_entity_detection(example=example, spans=spans)
        assert len(result["missing_entities"]) == 1
        assert len(result["false_positives"]) == 1
        assert ec.error_counts[ErrorTaxonomy.type_1] == 1
        assert ec.error_counts[ErrorTaxonomy.type_0] == 1
