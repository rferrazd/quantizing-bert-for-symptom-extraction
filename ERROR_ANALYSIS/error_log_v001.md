# Error Logs v0.1

> script is at the bottom of the file v01/inference_test.ipynb

Total of {X} cases in real_world_cases.py

## Case: case_multiple_symptoms
**Date:** 2026-01-16
**Reviewer:** HUMAN
**Model:** `dmis-lab/biobert-base-cased-v1.1`
**Model Version:** `v0.1`
**Inference Pipeline Version:** `v0.1`
**Case ID / Name:** `case_multiple_symptoms`
**Input Text:** The patient, male 65 years ols reports hair shedding, blisters on his head, and knee pain.
**Predicted Spans (Excluding 'O'):** [{'start': 19, 'end': 21, 'text': '65', 'label': 'SYMPTOM_POS'}, {'start': 40, 'end': 53, 'text': 'hair shedding', 'label': 'SYMPTOM_POS'}, {'start': 56, 'end': 78, 'text': 'blisters on his head ,', 'label': 'SYMPTOM_POS'}, {'start': 83, 'end': 92, 'text': 'knee pain', 'label': 'SYMPTOM_POS'}]
**Expected behaviour:** [{'text': 'hair shedding', 'label': 'SYMPTOM_POS'}, {'text': 'knee pain', 'label': 'SYMPTOM_POS'}, {'text': 'exanthema', 'label': 'SYMPTOM_POS'}]
error classificatin: `Irrelevant Span Mislabeling` , `Boundary Overreach` 

## Case: case_negation
**Date:** 2026-01-16
**Reviewer:** HUMAN
**Model:** `dmis-lab/biobert-base-cased-v1.1`
**Model Version:** `v0.1`
**Inference Pipeline Version:** `v0.1`
**Case ID / Name:** `case_negation`
**Input Text:** The patient does not have muscle necrosis, only signs of muscle cramps.
**Predicted Spans (Excluding 'O'):** [{'start': 26, 'end': 41, 'text': 'muscle necrosis', 'label': 'SYMPTOM_NEG'}, {'start': 58, 'end': 71, 'text': 'muscle cramps', 'label': 'SYMPTOM_NEG'}]
**Expected behaviour:**  [{'text': 'muscle necrosis', 'label': 'SYMPTOM_NEG'}, {'text': 'muscle cramps', 'label': 'SYMPTOM_POS'}]
**Error classificatin:** `Incorrect Polarity Assignment`



## Case: case_adjacent_entities
**Date:** 2026-01-16
**Reviewer:** HUMAN
**Model:** `dmis-lab/biobert-base-cased-v1.1`
**Model Version:** `v0.1`
**Inference Pipeline Version:** `v0.1`
**Case ID / Name:** `case_multiple_symptoms`
**Input Text:** The patient has a facial edema rash and knee pain.
**Predicted Spans:** [
    {'start': 18, 'end': 35, 'text': 'facial edema rash', 'label': 'SYMPTOM_POS'}, 
    {'start': 40, 'end': 49, 'text': 'knee pain', 'label': 'SYMPTOM_POS'}] 
**Expected entities:**   [
        {"text": "facial edema", "label": "SYMPTOM_POS"},
        {"text": "rash", "label": "SYMPTOM_POS"},
        {"text": "knee pain", "label": "SYMPTOM_POS"}
    ]
**Error classificatin:** `Boundary Overreach`


## Case: case_no_symptoms
**Date:** 2026-01-16
**Reviewer:** HUMAN
**Model:** `dmis-lab/biobert-base-cased-v1.1`
**Model Version:** `v0.1`
**Inference Pipeline Version:** `v0.1`
**Case ID / Name:** `case_multiple_symptoms`
**Input Text:** Marie Claire is feeling well today.
**Predicted Spans:** [{'start': 0, 'end': 5, 'text': 'Marie', 'label': 'SYMPTOM_POS'}, {'start': 6, 'end': 12, 'text': 'Claire', 'label': 'CONFLICT-I-SYMPTOM_POS-I-SYMPTOM_NEG-I-SYMPTOM_POS'}, {'start': 24, 'end': 28, 'text': 'well', 'label': 'SYMPTOM_POS'}]

**Expected entities:** []
**Error classificatin :** `Irrelevant Span Mislabeling`
