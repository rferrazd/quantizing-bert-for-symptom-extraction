# Error Log


When running an inference on the real world examples, the following logging template will be used to assess the error. Perhaps, later on, a LLM maybe used instead of a human to label the errors. 

Based on the logs, we will decide on what to improve for the next version. If any blockers appear that imped the improvement based on that certain category we will pivot to the second.

By logging the errors we will be able to:
    * build intuition about failure modes

    * track patterns, not anecdotes (i.e., isolated one-off 
    mistakes that may not reflect systematic issues)

    * decide which part of our system to improve (data,model,or inference logic)

The logs will become:

    * inputs for data enrichment

    * acceptance criteria for the systems (model, inference, data) upgrades


## Entry

**Date:** 
**Reviewer:** HUMAN
**Model:** 
**Model Version:** (e.g. v0.1)
**Inference Pipeline Version:** (e.g. span_v1)
**Case ID / Name:** (optional)
**Input Text:** <raw input text here>
**Predicted Spans:** [{span_info}]
**expected behaviour:** [{ground_truth}]
**error classificatin:** Enum[ERROR_TAXONOMY.md]


**Date:** 2026-01-16
**Reviewer:** HUMAN
**Model:** `dmis-lab/biobert-base-cased-v1.1`
**Model Version:** `v0.1`
**Inference Pipeline Version:** `v0.1`
**Case ID / Name:** `case_multiple_symptoms`
**Case ID / Name:** 
**Input Text:**
**Predicted Spans:** 
**expected behaviour:** 
**error classificatin:** 


## Impact Assessment 

## Error Frequency / Patterns


## Action


## Final Notes (optional)
