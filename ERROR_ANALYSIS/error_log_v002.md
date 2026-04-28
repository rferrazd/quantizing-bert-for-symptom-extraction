

**Date:** 04/28/2026 
**Reviewer:** HUMAN
**Model:** biobert-base-cased-v1.1/run_2
**Model Version:** v0.2
**Inference Pipeline Version:** v0.1
**Case ID / Name:** First Error Report for v0.2
**Behavioural Dataset:** v01/behavioural_set.json
**Source Notebook:** v02/categorize_errors_in_behavioural_set.ipynb


# Brief Model Details

These details can be found in the GCP bucket
```
{
  "model_name": "dmis-lab/biobert-base-cased-v1.1",
  "training_time_minutes": 21.0,
  "hyperparameters": {
    "model_name": "dmis-lab/biobert-base-cased-v1.1",
    "dataset_repo": "Rogarcia18/symptoms_ner_v02_biobert",
    "epoch": 20,
    "lr": 5e-05,
    "batch_size": 16,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "push_to_hub": true
  },
  "validation_metrics": {
    "f1": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "accuracy": 1.0
  },
  "test_metrics": {
    "f1": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "accuracy": 1.0
  }
}
```
# Geral Statistics:
'Irrelevant Span Mislabeling': 30,
'Model Overgeneralization': 27,
'Missing Expected Entity': 0,
'Boundary Overreach': 31,
'Tokenization Artifacts': 1,
'BIO Sequencing Errors': 1,
'Incorrect Polarity Assignment': 1



## Error Examples

## *Irrelevant Span Mislabeling*
```json
{'entity': {'ent': 'hepatic dysfunction',
   'label': 'O',
   'start': 45,
   'end': 64},
  'span': {'start': 45,
   'end': 64,
   'text': 'hepatic dysfunction',
   'label': 'SYMPTOM_POS'},
  'span_idx': 4,
  'errors': ['Irrelevant Span Mislabeling']}
```



## *Model Overgeneralization*
```json
{'entity': None,
  'span': {'start': 11,
   'end': 21,
   'text': 'complaints',
   'label': 'SYMPTOM_NEG'},
  'span_idx': 2,
  'errors': ['Irrelevant Span Mislabeling', 'Model Overgeneralization'],
  'reasoning': "Predicted entity span 'complaints' not in expected entities"},

 {'entity': None,
  'span': {'start': 10,
   'end': 23,
   'text': 'questionnaire',
   'label': 'SYMPTOM_POS'},
  'span_idx': 1,
  'errors': ['Irrelevant Span Mislabeling', 'Model Overgeneralization'],
  'reasoning': "Predicted entity span 'questionnaire' not in expected entities"}

```


## *Missing Expected Entity*
```json
```
## *Boundary Overreach*
```json
{'entity': {'ent': 'bradypnea',
   'label': 'SYMPTOM_POS',
   'start': 13,
   'end': 22},
  'span': {'start': 13,
   'end': 38,
   'text': 'bradypnea since yesterday',
   'label': 'SYMPTOM_POS'},
  'span_idx': 2,
  'errors': ['Boundary Overreach']}

```

## *Tokenization Artifacts and BIO Sequencing Errors*
```json
{'entity': {'ent': 'chills',
    'label': 'SYMPTOM_NEG',
    'start': 148,
    'end': 154},
   'span': {'start': 148,
    'end': 154,
    'text': 'chills',
    'label': 'CONFLICT-I-SYMPTOM_NEG-I-SYMPTOM_POS'},
   'span_idx': 9,
   'errors': ['Tokenization Artifacts', 'BIO Sequencing Errors']}
```

## *Incorrect Polarity Assignment*
```json
{'entity': {'ent': 'wheezing',
    'label': 'SYMPTOM_POS',
    'start': 30,
    'end': 38},
   'span': {'start': 30,
    'end': 49,
    'text': 'wheezing is related',
    'label': 'SYMPTOM_NEG'},
   'span_idx': 5,
   'errors': ['Incorrect Polarity Assignment', 'Boundary Overreach']}
```



# Action Points

- Decide on next steps:
   - improve dataset with more examples
      - more POS and NEG examples 
      - more challenging examples 
   - training
      - the entire model
      - finetune only the classification layer


