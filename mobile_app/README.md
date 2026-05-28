

# mobile_app/

Home for the entire mobile app initiative. One subfolder per phase.

| Folder | Phase | Status |
|---|---|---|
| `model_prep/` | Phase 1 — Python model export & validation | In progress |
| `shared/` | Cross-phase utilities | In progress |
| `artifacts/v05/` | Generated model files (gitignored) | Done |
| `app/` | Phase 2 — React Native app | Not started |

---

## Error-counting rule

Errors are counted by `ErrorCategorizer._check_entity_detection` in
`error_analysis/error_categorization.py`. For each HDA it finds:
- **Missing entities** — gold span not detected at all (type_1)
- **Present errors** — detected but wrong label or wrong boundary (type_0,3,5,6)
- **False positives** — predicted span with no gold match (type_0)

Total = `sum(error_counts.values())` across all 7 HDAs.
The harness that runs this is `error_analysis/hda_evaluation.py`.

---

## Phase 1 validation results (7-HDA hold-out, V05)

| Pipeline | Total errors | Acceptance (≤ 10) |
|---|---|---|
| PyTorch fp32 (baseline) | **8** | ✅ |
| ONNX INT8 (~110 MB) | **13** | ❌ |
| ONNX fp32 (~411 MB) | not yet run | — |

**INT8 regression breakdown** — 5 new errors vs PyTorch baseline:

| Error type | PyTorch | ONNX INT8 | Delta |
|---|---|---|---|
| Irrelevant Span Mislabeling | 6 | 9 | +3 |
| Boundary Undereach | 1 | 2 | +1 |
| Incorrect Polarity Assignment | 1 | 0 | -1 |
| Tokenization Artifacts | 0 | 1 | +1 |
| BIO Sequencing Errors | 0 | 1 | +1 |

Affected HDAs: `medium_rash`, `hda6`, `hda8` (+1 each).

**Decision:** proceeding to Phase 2 with INT8. The quantization recipe
(try `per_channel=True`) will be revisited before app-store submission.
INT8 at ~110 MB fits Apple's 200 MB cellular cap; fp32 (411 MB) and fp16
(~220 MB) do not.

To reproduce: `python -m mobile_app.model_prep.validate_onnx`

---

## Artifacts (v05)

```
861B   config.json
411M   model.onnx          (fp32, export reference)
104M   model_int8.onnx     (INT8, used by app)
695B   special_tokens_map.json
653K   tokenizer.json
1.3K   tokenizer_config.json
208K   vocab.txt
```
Total ~432 MB on disk (gitignored).