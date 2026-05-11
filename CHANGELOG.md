# Changelog

## 2026-05-07

### `PROGRESS_NOTES/v03.md`

- Documented dataset scale (templates × rows, Hub id, splits) alongside the completed **V03** BioBERT run.
- Added **V032** block: head-only / frozen-backbone training (40 epochs) with validation and test JSON metrics and behavioural taxonomy counts (JSON fences).
- New **V03 vs V032 comparison**: IID synthetic metrics versus behavioural stress-test interpretation; behavioural bank vs [`v03/dataset_templates.py`](v03/dataset_templates.py); note that **`Irrelevant Span Mislabeling` and `Model Overgeneralization` double-count each false-positive span**; why boundary-overreach totals are not comparable in isolation; conclusions and next steps.
- Naming / cleanup: subsection title **V032** (replacing the old V03.1 placeholder); fixed malformed code fence; normalized behavioural-set JSON formatting.

### `ERROR_ANALYSIS/error_categorization.py`

- Module docstring: single-line advisory that **each false-positive span increments both type\_0 and type\_7** in aggregate counters.
- Inline comments at the false-positive handler: **how to fix** (single bucket, exclusive split with heuristics, or keep legacy double-count and interpret `(type_0 + type_7) / 2`).

### Other (same release)

- `hyperparam_sets.py`: BioBERT run entry **epochs → 40** for the V032 sweep; **`distilbert_hyperparams`** block commented out.
- `v03/9_categorize_errors_in_behavoural_set.ipynb`: behavioural categorization notebook refreshed (paths/cells/output).
