

**Date:** 03/30/2026 
**Reviewer:** HUMAN
**Model:** biobert-base-cased-v1.1/run_2
**Model Version:** v0.1
**Inference Pipeline Version:** v0.1
**Case ID / Name:** First Error Report
**Behavioural Dataset:** v01/behavioural_set.json
**Source Notebook:** v01/categorize_errors_in_behavioural_set.ipynb


# Brief Model Details

These details can be found in the GCP bucket
```
{
  "model_name": "dmis-lab/biobert-base-cased-v1.1",
  "training_time_minutes": 7.01,
  "hyperparameters": {
    "model_name": "dmis-lab/biobert-base-cased-v1.1",
    "dataset_repo": "Rogarcia18/symptoms_ner_v01_biobert",
    "epoch": 20,
    "lr": 5e-05,
    "batch_size": 16,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "push_to_hub": true
  },
  "validation_metrics": {
    "f1": 0.809000255689082,
    "precision": 0.7444705882352941,
    "recall": 0.8857782754759238,
    "accuracy": 0.968895800933126
  },
  "test_metrics": {
    "f1": 0.7906741003547896,
    "precision": 0.7222222222222222,
    "recall": 0.8734602463605823,
    "accuracy": 0.9649692297288158
  }
}
```
# Geral Statistics:
'Irrelevant Span Mislabeling': 75,
'Model Overgeneralization': 74,
'Missing Expected Entity': 11,
'Boundary Overreach': 8,
'Tokenization Artifacts': 4,
'BIO Sequencing Errors': 4,
'Incorrect Polarity Assignment': 3



## Error Examples

## *Irrelevant Span Mislabeling*
{'entity': {'ent': 'fixed dilated pupils',
    'label': 'O',
    'start': 33,
    'end': 53},
   'span': {'start': 33,
    'end': 53,
    'text': 'fixed dilated pupils',
    'label': 'SYMPTOM_POS'},
   'span_idx': 3,
   'errors': ['Irrelevant Span Mislabeling']}



   

## *Model Overgeneralization*
```json
{'entity': None,
   'span': {'start': 61,
    'end': 72,
    'text': 'prostration',
    'label': 'CONFLICT-I-SYMPTOM_NEG-I-SYMPTOM_POS-I-SYMPTOM_NEG'},
   'span_idx': 8,
   'errors': ['Irrelevant Span Mislabeling', 'Model Overgeneralization']
}

{
    'entity': None,
   'span': {'start': 26,
    'end': 68,
    'text': 'superficial lump that started two days ago',
    'label': 'SYMPTOM_POS'},
   'span_idx': 3,
   'errors': ['Irrelevant Span Mislabeling', 'Model Overgeneralization']
}

```
## *Missing Expected Entity*
```json
{'entity': {'ent': 'localized superficial lump',
    'label': 'SYMPTOM_POS',
    'start': 16,
    'end': 42},
   'span': None,
   'errors': ['Missing Expected Entity'],
   'reasoning': "Expected entity 'localized superficial lump' was not detected"}
```
## *Boundary Overreach*
```json
{'entity': {'ent': 'tetanic convulsion',
    'label': 'SYMPTOM_POS',
    'start': 14,
    'end': 32},
   'span': {'start': 14,
    'end': 33,
    'text': 'tetanic convulsion,',
    'label': 'SYMPTOM_POS'},
   'span_idx': 2,
   'errors': ['Boundary Overreach']}
```
## *Tokenization Artifacts and BIO Sequencing Errors*
```json
{'entity': {'ent': 'steatorrhea',
    'label': 'SYMPTOM_POS',
    'start': 50,
    'end': 61},
   'span': {'start': 50,
    'end': 61,
    'text': 'steatorrhea',
    'label': 'CONFLICT-B-SYMPTOM_NEG-I-SYMPTOM_POS-I-SYMPTOM_NEG-I-SYMPTOM_NEG-I-SYMPTOM_NEG'},
   'span_idx': 8,
   'errors': ['Tokenization Artifacts', 'BIO Sequencing Errors']}

{'entity': {'ent': 'bradypnea',
    'label': 'SYMPTOM_POS',
    'start': 13,
    'end': 22},
   'span': {'start': 13,
    'end': 22,
    'text': 'bradypnea',
    'label': 'CONFLICT-B-SYMPTOM_NEG-I-SYMPTOM_POS-I-SYMPTOM_NEG-I-SYMPTOM_NEG'},
   'span_idx': 2,
   'errors': ['Tokenization Artifacts', 'BIO Sequencing Errors']}
```

## *Incorrect Polarity Assignment*
```json
{'entity': {'ent': 'dysphonia',
    'label': 'SYMPTOM_POS',
    'start': 13,
    'end': 22},
   'span': {'start': 13,
    'end': 22,
    'text': 'dysphonia',
    'label': 'SYMPTOM_NEG'},
   'span_idx': 1,
   'errors': ['Incorrect Polarity Assignment']}
```


# Action Points

