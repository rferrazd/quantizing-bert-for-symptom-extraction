# Error Log

When running inference on real-world or behavioural examples, use this template to record what failed and why.

Based on the logs, we decide what to improve next. If a category is blocked by tooling or data, note it and pivot to the next priority.

Logging errors helps you to:

- build intuition about failure modes
- track **patterns**, not one-off anecdotes
- choose whether to improve **data**, **model**, or **inference** logic

Logs become:

- inputs for data enrichment
- acceptance criteria for model, inference, and data upgrades

---

## Report header (behavioural sweep or milestone)

Use this block for a **batch report** (e.g. full behavioural set) or a named review pass.

**Date:**  
**Reviewer:** HUMAN | AI | CODE  
**Model:** (e.g. `dmis-lab/biobert-base-cased-v1.1` + checkpoint id such as `run_2`)  
**Model Version:** (e.g. v0.1)  
**Inference Pipeline Version:** (e.g. v01)  
**Case ID / Name:** (optional title for this log entry)  
**Behavioural Dataset:** (path, e.g. `v01/behavioural_set.json`)  
**Source Notebook / Script:** (e.g. `v01/categorize_errors_in_behavioural_set.ipynb`)

---

## Aggregate statistics

Summarize counts from your error taxonomy (see `ERROR_TAXONOMY.md` for definitions).

Paste a counter or table, for example:

- Irrelevant Span Mislabeling: N  
- Model Overgeneralization: N  
- …

---

## Error examples (by taxonomy)

Group illustrative records **under each error type** so patterns are easy to scan.

For each example, prefer a small structured blob (JSON or dict-style) that matches your pipeline output, e.g.:

- `entity` (ground truth, or `null` for false positives)
- `span` (prediction, or `null` for missing entities)
- `errors` (list of taxonomy labels)
- `reasoning` (optional; required for some cases such as misses)

Use fenced `json` blocks when pasting multi-line examples.

---

## Single-example deep dive (optional)

For one sentence or clinical snippet, you can still use the original per-case layout:

**Input Text:** (raw clinical text)  
**Predicted Spans:** (list of span dicts: start, end, text, label)  
**Expected behaviour:** (ground-truth entities / labels)  
**Error classification:** (labels from `ERROR_TAXONOMY.md`)

---

## Brief model / training context (optional)

Fill what you know; point to `summary.json` / `test_metrics.json` under the run directory when available.

- Number of parameters (total vs trainable if backbone was frozen)  
- Training data size / split  
- Hyperparameters (see repo: `hyperparam_sets.py` for the sweep slot that matches your `run_*` index, and `7_trainer.py` / `7_trainer_gcp.py` for how they map into `TrainingArguments`)

---

## Impact assessment

What user-facing or clinical risk does this pattern imply? How often does it appear relative to others?

---

## Action points

Concrete next steps (data fixes, label rules, inference changes, or new training runs).

---

## Final notes (optional)

Anything that does not fit above (open questions, follow-up experiments).
