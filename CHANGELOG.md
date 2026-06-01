# Changelog

## 2026-05-31 — Phase 2 mobile app (Blocks A–C complete)

### New: React Native iOS app at `mobile_app/app/SymptomNerApp/`
RN 0.85.3 single screen running V05 fully on-device via `onnxruntime-react-native`. iOS-first. Ships the 108 MB INT8 ONNX + vocab + config; version-flexible (v06 swap = one script + rebuild).

### TypeScript NER pipeline (parity-verified against Python)
- `src/ner/tokenizer.ts` — port of `BertTokenizerFast` (BertNormalizer + BertPreTokenizer + WordPiece) with per-token char offsets into the original text. Lowercase=true matches the exported `tokenizer.json` (see open question in project memory: cased BioBERT fed lowercased input — to investigate, not blocking).
- `src/ner/aggregate.ts` — port of `inference_utils.aggregate_token_predictions_to_words` + `word_labels_to_spans` (BIO with B-priority, CONFLICT, robust I-without-B).
- `src/ner/infer.ts` — port of `mobile_app/model_prep/onnx_inference.predict_word_level_onnx` (forward pass + orchestration). Defensive truncation to `max_position_embeddings`.
- `src/ner/session.ts` — ONNX session loader (fetch → `Uint8Array`, clears promise on failure so retries work).
- `src/ner/modelConfig.ts` — reads `id2label`, `numLabels`, `maxSeqLen` from bundled `config.json`; no model-version hardcoded in app code.

### Parity tests (33 total, all passing)
- `src/ner/__tests__/tokenizer.test.ts` — 25 cases vs Python `AutoTokenizer` (IDs, `word_ids`, char offsets; uppercase, punctuation, numbers, subwords, accent stripping, control char, tab).
- `src/ner/__tests__/aggregate.test.ts` — 8 cases vs Python `inference_utils` (single-entity, multi-subword B-priority, CONFLICT, I-without-B, multi-word continuation).
- On-device verification: 6 example notes produce spans bit-for-bit identical to Python `onnx_inference.py` over `model_int8.onnx`.

### App
- `App.tsx` — single screen: brand mark scaffold, disclaimer, project description, 6 tap-to-load example clinical notes (varied negation cues: "denies", "no X", "negative for", "without"), clinical-note input with live token counter enforcing `maxSeqLen − 2` cap, Run button, POS/NEG badge output, searchable collapsible list of 893 trained symptoms.
- `onRun` race-safe via monotonic `runIdRef`; CONFLICT spans filtered from UI and logged to Metro.
- Outer container is a `FlatList` with `ListHeaderComponent` (replaces a `FlatList`-inside-`ScrollView` nesting that broke virtualization).

### Asset pipeline
- `mobile_app/model_prep/bundle_app_assets.py` — copies `<version>/model.onnx` + `vocab.txt` (→ `vocab.json`) + `config.json` into `assets/model/` under generic filenames. Atomic writes, UTF-8 BOM + CRLF tolerant.
- `metro.config.js` — `.onnx` registered as a Metro asset.
- `assets/model/` gitignored (regenerable build output).

### App identity + deployment
- `scripts/generate_app_icon.py` — generates the iOS app icon in 8 sizes (rounded square, POS-green / NEG-red horizontal split, charcoal frame); auto-updates `AppIcon.appiconset/Contents.json`.
- `mobile_app/app/SymptomNerApp/README.md` — iOS-only run/setup instructions, troubleshooting for the gotchas hit (port 8081 collision, missing iOS Simulator runtime in Xcode 26, `LANG=en_US.UTF-8` pod-install requirement).
- `mobile_app/DEPLOY_IOS.md` (new) — Apple Developer Program ($99/yr), TestFlight vs. App Store paths, review risks for medical-adjacent content, free-pricing recommendation.

## 2026-05-19 — WordPiece alignment, validation split, HuggingFace upload

- `data_preparation/wordpiece_alignment.py` — new module. `align_sample()` maps word-level BIO labels to BioBERT subwords via `word_ids()`. `align_file()` processes a full JSONL. LABEL2ID hardcoded (5 labels). `__main__` aligns all three splits.
- `v04/data/verify_wordpiece.py` — sanity-check script. All checks passed: sequence length, BIO continuity, label id range, label distribution, K=40 HDA coverage (866/873 = 97%).
- `v04/data/upload_to_hf.py` — carves 90/10 train/validation split (stratified by template_group, seed 42), uploads all four splits to `Rogarcia18/symptom-ner-dataset-v04` (private).
- `hf_utils.py` — implemented `upsert_to_hf_repo()`: loads JSONL → HF Dataset → `push_to_hub()`.
- `config.py` — added `HUGGINGFACE_DATASET_REPO_ID`.
- **Artifacts**: `v04/data/splits/{train,validation,template_ood,symptom_ood}_wordpiece.jsonl`; train=82,360 / val=9,152 / template_ood=12,262 / symptom_ood=2,440. HF upload complete.

## 2026-05-18 — dataset split + full v04 generation

### New files
- `data_preparation/dataset_split.py`: deterministic train/eval split module. Constants: `EVAL_SPLIT_SEED=42`, `HELDOUT_COUNTS` (5 affirmed, 5 negated, 4 distractor, 2 hda), `HELDOUT_SYMPTOM_COUNT=20`. Functions: `split_template_groups` (returns train groups, heldout groups, heldout indices map), `split_symptoms_df`, `save_split_artifacts`, `load_split_artifacts`.
- `v04/test_dataset_split.py`: 9-check smoke test covering counts, no-overlap, determinism, index consistency, and save/load round-trip.

### Modified files
- `data_preparation/dataset_generator.py` — `__main__` rewritten to produce three datasets: train (K=40 HDA), template_ood (K=20 HDA), symptom_ood (K=20 HDA). Prints held-out indices, saves `v04/splits/split_artifacts.json`, writes combined `not_found.jsonl` tagged by split.
- `v04/dataset_templates.py` — `NEGATED_TEMPLATES` brought to 40 (added `"Patient denies {SYMPTOM_NEG}."`, a high-frequency cue absent from single-symptom templates).
- `test_pipeline_walkthrough.py` — section 13 added: toy demo of `split_template_groups` and `split_symptoms_df`, showing before/after sizes and determinism.

### Generated artifacts (v04/)
- `v04/splits/split_artifacts.json` — held-out template indices + held-out symptom ids (seed 42)
- `v04/train_raw.jsonl` + `v04/train_tokenized.jsonl` — **91,512 samples** (35×873 aff + 35×873 neg + 34×873 dist + 18 hda×K=40)
- `v04/template_ood_raw.jsonl` + `v04/template_ood_tokenized.jsonl` — **12,262 samples** (5+5+4 groups ×873 + 2 hda×K=20)
- `v04/symptom_ood_raw.jsonl` + `v04/symptom_ood_tokenized.jsonl` — **2,440 samples** (35+35+34+18 templates ×20 held-out symptoms)
- `v04/not_found.jsonl` — **0 entries** (no tokenization failures across any split)

## 2026-05-18 — session `ceec7de1-2a8b-4ead-b0da-11d3fcd0a6c9`

- `data_preparation/dataset_generator.py`: fixed cross-polarity dedup in `build_fill_map` (single `used_ids` set + retry cap); moved save block out of per-group loop; `import json` at module level; `__main__` now uses methods' `save_in_jsonl_path` and calls `gen.tokenize_all_samples(samples, …)` correctly; `Literal[None|str]` → `Optional[str]`.
- `test_pipeline_walkthrough.py`: new step-by-step walkthrough of every helper and method in the generator (learning aid).

## 2026-05-13

- `error_analysis/error_categorization.py`: positional `_is_entity_in_spans`; `_check_boundaries` routes <50% overlap to type_5/type_6 by length; FP counts only type_0.
- `error_analysis/tests/test_error_categorizer.py`: 19 tests, all passing.
- `PROGRESS_NOTES/v03.md`: post-fix V03/V032 counts; V032's overreach drop was scorer artifact.

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
