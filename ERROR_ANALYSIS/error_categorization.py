"""Error categorization for behavioural NER evaluation (V01+).

Each false-positive span (predicted entity with no gold match) increments *both*
Irrelevant Span Mislabeling (type_0) and Model Overgeneralization (type_7); do
not sum those two buckets as independent counts without deduplicating.
"""


from typing import List,  Dict

from collections import Counter
import datetime
# Local imports
from error_analysis.error_taxonomy import ErrorTaxonomy, BehaviouralExample, Spans


class ErrorCategorizer():

    def __init__(self):
        self.error_counts = Counter()

    def _is_entity_in_spans(self, 
       entity_text: str, 
       spans: List[Spans]
       ):
        # TODO (P1): define types for entity_text and spans and return type
        """
        Check if entity text is contained within any span text and return the span index.
        
        Returns:
            int | None: The index of the first span containing the entity text, or None if not found.
        
        How it works:
        - Uses `enumerate()` to get both the index and span from the spans list
        - For each span, checks if entity_text is a substring of span['text']
        - Returns the index as soon as a match is found (short-circuit evaluation)
        - Returns None if no match is found
        
        Example:
            entity_text = "fever"
            spans = [
                {'text': 'The patient', ...},      # index 0: "fever" not in "The patient" -> continue
                {'text': 'fever or chills', ...},  # index 1: "fever" in "fever or chills" -> return 1
                {'text': 'reports', ...}           # (not reached due to short-circuit)
            ]
            # Returns: 1
        """
        for idx, span in enumerate(spans):
            if entity_text in span['text']:
                return idx
        return None
    
    def _check_boundaries(self, expected_start, expected_end, predicted_start, predicted_end):
        """
        Check for boundary errors (overreach/undereach).
        
        Returns:
            None: No boundary error
            ErrorTaxonomy.type_5: Overreach (predicted span is longer)
            ErrorTaxonomy.type_6: Undereach (predicted span is shorter)
        """
        # Calculate overlap
        overlap_start = max(expected_start, predicted_start)
        overlap_end = min(expected_end, predicted_end)
        overlap_length = max(0, overlap_end - overlap_start)
        
        expected_length = expected_end - expected_start
        predicted_length = predicted_end - predicted_start
        
        # If no meaningful overlap, this isn't a boundary error
        # (it's a different entity or wrong detection)
        if overlap_length < expected_length * 0.5:  # Less than 50% overlap
            # TODO: figure out what kind of error to return
            return None  # Not a boundary issue, likely type_0 or type_7 
        
        # Perfect match
        if expected_start == predicted_start and expected_end == predicted_end:
            return None
        
        # Overreach: predicted span extends beyond expected boundaries
        if predicted_start < expected_start or predicted_end > expected_end:
            return ErrorTaxonomy.type_5
        
        # Undereach: predicted span is contained within expected boundaries
        if predicted_start >= expected_start and predicted_end <= expected_end:
            return ErrorTaxonomy.type_6
        
        # Mixed case (starts before but ends early, or starts late but ends after)
        # This is tricky - could be either, but usually overreach is more common
        if predicted_length > expected_length:
            return ErrorTaxonomy.type_5
        else:
            return ErrorTaxonomy.type_6

    def _check_predicted_labels(self, predicted_label:str, label: str):
        """this goes inside the if block for the detected entities, so we should not see"""

        if predicted_label == label:
            # This will catch the cases where label = "O", 
            # we included some tricky cases where symptoms are present in the cases but NOT in the context of a patient describing the symptoms that she/he does not have.
            return None
        if label == 'O' and predicted_label != 'O':
            # tricky case, where we did not want the model to detect this term as an entity
            # If we fall here then the model detected an irrelevant entity

            # check if there was a conflict:
            if predicted_label.startswith("CONFLICT"):
                return [ErrorTaxonomy.type_0, ErrorTaxonomy.type_2, ErrorTaxonomy.type_4]
            
            return ErrorTaxonomy.type_0
            
        if label != 'O' and predicted_label == 'O':
            return ErrorTaxonomy.type_1 # missed an entity

        if predicted_label.startswith("CONFLICT"):
            return [ErrorTaxonomy.type_2, ErrorTaxonomy.type_4]
        if predicted_label.split("_")[1] != label.split("_")[1]:
            return ErrorTaxonomy.type_3 # incorrect polarity assignment

    def _check_entity_detection(
        self,
        example: BehaviouralExample,
        spans: List[Dict],  # TODO: update to List[Spans]
        ) -> Dict:
        all_errors = []
        missing_entities = []

        # 1) Match expected entities -> predicted span index (lenient substring match)
        present_with_indices = [(e, self._is_entity_in_spans(e["ent"], spans)) for e in example.entities_with_labels]
        present = [(e, idx) for e, idx in present_with_indices if idx is not None]
        missing = [e for e, idx in present_with_indices if idx is None]

        matched_span_indices = {idx for _, idx in present}

        # 2) Missing entities (type_1)
        for e in missing:
            missing_entities.append({
                "entity": e,   # expected entity (ground truth)
                "span": None,  # no predicted span — entity was not detected
                "errors": [ErrorTaxonomy.type_1],
                "reasoning": f"Expected entity '{e['ent']}' was not detected",
            })
            # Increase count for missing entity
            self.error_counts[ErrorTaxonomy.type_1] += 1

        # 3) Errors for detected entities
        for entity_data, span_idx in present:
            span = spans[span_idx]

            expected_start, expected_end = entity_data["start"], entity_data["end"]
            predicted_start, predicted_end = span["start"], span["end"]
            predicted_label = span["label"]
            true_label = entity_data["label"]

            entity_errors = []

            # 3a) Label-related errors
            label_errors = self._check_predicted_labels(predicted_label=predicted_label, label=true_label)
            if label_errors:
                entity_errors.extend(label_errors if isinstance(label_errors, list) else [label_errors])

                # Update the count for label errors
                if isinstance(label_errors, list):
                    for err in label_errors:
                        self.error_counts[err] += 1
                else:
                    self.error_counts[label_errors] += 1

            # 3b) Boundary-related errors
            boundary_error = self._check_boundaries(
                expected_start=expected_start,
                expected_end=expected_end,
                predicted_start=predicted_start,
                predicted_end=predicted_end,
            )
            if boundary_error:
                entity_errors.append(boundary_error)
                # boundary_error is always a single value (never a list)
                self.error_counts[boundary_error] += 1

            if entity_errors:
                all_errors.append({
                    "entity": entity_data,
                    "span": span,
                    "span_idx": span_idx,
                    "errors": entity_errors,
                })

        # 4) False positives: predicted entity spans not matched to any expected entity
        false_positives = []
        for span_idx, span in enumerate(spans):
            if span["label"] == "O":
                continue
            if span_idx in matched_span_indices:
                continue

            # Both types attached to the record for downstream inspection; counters below double-count.
            error_types = [ErrorTaxonomy.type_0, ErrorTaxonomy.type_7]
            false_positives.append({
                "entity": None,  # no matching expected entity — this is a false positive
                "span": span,
                "span_idx": span_idx,
                "errors": error_types,
                "reasoning": f"Predicted entity span '{span['text']}' not in expected entities",
            })

            # Fix double-count (pick one):
            # - Single bucket: increment only type_0 *or* type_7, and set `error_types` above to a one-element list.
            # - Split FPs: add heuristics (span length, label POS vs NEG, proximity to gold O text) and assign
            #   type_0 vs type_7 exclusively, incrementing once per span.
            # - Keep double-count for legacy dashboards: document that (type_0 + type_7) / 2 ≈ FP span count.
            self.error_counts[error_types[0]] += 1
            self.error_counts[error_types[1]] += 1

        return {
            "present_errors": all_errors,  # boundary+label errors per expected entity span
            "missing_entities": missing_entities,
            "false_positives": false_positives,
        }


def _run_through_behavioural_set(
    error_categorizer: ErrorCategorizer,
    behavioural_set: Dict[str, List[BehaviouralExample]],
    model,
    tokenizer,
    id2label: dict,
    device: str,
) -> dict:
    """Run inference + `_check_entity_detection` on every example.

    Clears `error_categorizer.error_counts` so one full sweep has a single aggregate counter.
    Returns dict with `error_counts` (Counter), plus flat lists of error records for inspection.
    """

    # lazy import
    from inference.v01.inference_utils import word_labels_to_spans, predict_word_level
    error_categorizer.error_counts.clear()

    total_present_errors: List[dict] = []
    total_missing_entities: List[dict] = []
    total_false_positives: List[dict] = []

    for ex_type, examples in behavioural_set.items():
        print(f"Category: {ex_type}")
        for ex in examples:
            text = ex.example
            tokens, token_labels, word_ids, words, word_labels, word_offsets = predict_word_level(
                text=text,
                model=model,
                tokenizer=tokenizer,
                id2label=id2label,
                device=device,
            )
            spans = word_labels_to_spans(
                text=text,
                word_offsets=word_offsets,
                word_labels=word_labels,
            )
            errors = error_categorizer._check_entity_detection(example=ex, spans=spans)
            pe = errors.get("present_errors", [])
            me = errors.get("missing_entities", [])
            fp = errors.get("false_positives", [])
            total_present_errors.extend(pe)
            total_missing_entities.extend(me)
            total_false_positives.extend(fp)
        print(f"  Processed {len(examples)} examples\n")

    return {
        "error_counts": error_categorizer.error_counts,
        "present_errors": total_present_errors,
        "missing_entities": total_missing_entities,
        "false_positives": total_false_positives,
    }


# --------------------
# DUMMY EXAMPLE USAGE
# --------------------
if __name__ == "__main__":
    # Example dummy taxonomy types (for illustration, replace with real enum/int from your taxonomy)
    class DummyErrorTaxonomy:
        type_0 = "TYPE_0_FALSE_POSITIVE"
        type_1 = "TYPE_1_MISSED"
        type_2 = "TYPE_2_CONFLICT"
        type_3 = "TYPE_3_POLARITY"
        type_4 = "TYPE_4_OTHER"
        type_5 = "TYPE_5_OVERREACH"
        type_6 = "TYPE_6_UNDEREACH"
        type_7 = "TYPE_7_NO_OVERLAP"
    # Replace this with: "from error_analysis.error_taxonomy import ErrorTaxonomy" and use ErrorTaxonomy in real code
    ErrorTaxonomy = DummyErrorTaxonomy

    # Minimal dummy BehaviouralExample for testing
    class DummyBehaviouralExample:
        def __init__(self):
            self.entities_with_labels = [
                {"ent": "fever", "start": 10, "end": 15, "label": "B_SYMP"},
                {"ent": "cough", "start": 20, "end": 25, "label": "B_SYMP"}
            ]

    # Dummy predicted spans, one matches, one is a false positive, one is an overreach
    dummy_spans = [
        {"text": "the patient reports fever", "start": 10, "end": 15, "label": "B_SYMP"},  # match
        {"text": "no cough seen", "start": 20, "end": 22, "label": "B_SYMP"},              # undereach
        {"text": "nausea/vomiting", "start": 30, "end": 45, "label": "B_SYMP"},            # false positive
        {"text": "irrelevant", "start": 0, "end": 9, "label": "O"}                         # should be ignored
    ]

    # Create an instance of the categorizer and run detection
    categorizer = ErrorCategorizer()
    errors_result = categorizer._check_entity_detection(
        example=DummyBehaviouralExample(),
        spans=dummy_spans,
    )

    # Print results for illustration
    import pprint
    print("=== Errors result (dummy example) ===")
    pprint.pprint(errors_result)
    print("=== Error counts ===")
    pprint.pprint(dict(categorizer.error_counts))

