# BERT Symptom NER

Biomedical NER for extracting patient-reported (POS) and patient-denied (NEG) symptoms from "History of Present Illness" clinical notes. Two tracks: a Python ML pipeline (synthetic data → BioBERT fine-tuning → behavioural eval) and a React Native iOS app that runs the trained model entirely on-device.

## Status

| Track | State |
|---|---|
| ML pipeline | V05 shipped — **8 errors** on the 7-HDA behavioural set; V06 deferred until app ships |
| iOS app | Block C complete — TS pipeline parity-verified vs. Python; pre-deployment polish remaining |

## Layout

```
bert_symptom_ner/
├── data_preparation/        synthetic data generation, splits, WordPiece alignment
├── inference/v01/           shared inference helpers (PyTorch + ONNX paths)
├── error_analysis/          behavioural eval (7-HDA set) + error categorization
├── mobile_app/
│   ├── artifacts/<ver>/     ONNX export + tokenizer (gitignored)
│   ├── model_prep/          ONNX export, quantize, app-asset bundling
│   └── app/SymptomNerApp/   React Native iOS app
├── v0X/                     versioned datasets, training runs, notebooks
└── PROGRESS_NOTES/, hyperparam_sets.py, metrics.py, config.py
```

## ML pipeline

End-to-end:
1. Build symptom dictionary from BioPortal ontology
2. Generate synthetic POS/NEG samples (`data_preparation/dataset_generator.py`)
3. WordPiece-align labels (`data_preparation/wordpiece_alignment.py`)
4. Split + push to Hugging Face Hub
5. Train BioBERT (`v0X/7_trainer_gcp.py`, hyperparameters in `hyperparam_sets.py`)
6. Evaluate on validation + test + 7-HDA behavioural set
7. Categorize errors (`error_analysis/`)

### Setup

```bash
pip install -r requirements.txt
```

`.env`:
```
HF_TOKEN=...
HF_USERNAME=...
VERSION=v05
# Optional: USE_WANDB, WANDB_API_KEY, BIO_PORTAL_API_KEY, SAVE_TO_GCS, BUCKET_NAME
```
`config.py` derives `{HF_USERNAME}/symptoms_ner_{VERSION}` dataset repos.

### Label scheme

5-class collapsed BIO since V01: `O`, `B/I-SYMPTOM_POS`, `B/I-SYMPTOM_NEG`.

### Version history

| Ver | Notes | HDA errors |
|---|---|---|
| v00 | Symptom-specific labels — too sparse, abandoned | — |
| v01 | Frozen-encoder collapsed-label baseline | — |
| v02 | Experimental; perfect-metrics debugging | — |
| v03 | First full BioBERT run | 38 |
| v04 | Dataset + template revisions | 34 |
| v05 | **Current best**, shipped on-device | 8 |

### Entry points

- Training: `v0X/7_trainer_gcp.py` (current default V05); `hyperparam_sets.py` for sweep config
- Inference (Python): `inference/v01/inference_utils.predict_word_level` + `word_labels_to_spans`
- Inference (ONNX): `mobile_app/model_prep/onnx_inference.predict_word_level_onnx`
- Behavioural eval: `error_analysis/hda_evaluation.py`

## iOS mobile app

React Native 0.85.3, iOS-first, ONNX Runtime on-device. Single screen: type a clinical note → tap Run → see POS/NEG symptom badges. Also includes a searchable list of all 893 trained symptoms.

The TypeScript NER pipeline mirrors the Python pipeline 1:1 (tokenizer → forward pass → BIO aggregation → character spans). Parity is enforced by 33 unit tests against Python ground truth, and on-device output is bit-for-bit identical to `onnx_inference.py` on the same inputs.

- **Run it**: [mobile_app/app/SymptomNerApp/README.md](mobile_app/app/SymptomNerApp/README.md)
- **Deploy it**: [mobile_app/DEPLOY_IOS.md](mobile_app/DEPLOY_IOS.md)

## Notes

- BioBERT (`dmis-lab/biobert-base-cased-v1.1`) is the main backbone.
- Device selection for training: CUDA → MPS → CPU.
- Some naming is intentionally inconsistent (e.g. `ERROR_ANALYSIS` vs `error_analysis`); see `PROGRESS_NOTES/` for history.

## License

[TODO]

## Citation

[TODO]
