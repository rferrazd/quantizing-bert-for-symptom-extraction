---
library_name: transformers
base_model: dmis-lab/biobert-base-cased-v1.1
tags:
- generated_from_trainer
metrics:
- precision
- recall
- f1
- accuracy
model-index:
- name: symptom-ner-bert-models
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# symptom-ner-bert-models

This model is a fine-tuned version of [dmis-lab/biobert-base-cased-v1.1](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1613
- Precision: 0.7445
- Recall: 0.8858
- F1: 0.8090
- Accuracy: 0.9689

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 20

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:-----:|:---------------:|:---------:|:------:|:------:|:--------:|
| 1.3757        | 1.0   | 893   | 1.0429          | 0.0112    | 0.0095 | 0.0103 | 0.5819   |
| 0.8467        | 2.0   | 1786  | 0.6362          | 0.3330    | 0.4922 | 0.3972 | 0.8143   |
| 0.5817        | 3.0   | 2679  | 0.4627          | 0.5037    | 0.6909 | 0.5826 | 0.8915   |
| 0.4621        | 4.0   | 3572  | 0.3742          | 0.5820    | 0.7592 | 0.6589 | 0.9211   |
| 0.3931        | 5.0   | 4465  | 0.3188          | 0.6336    | 0.8057 | 0.7094 | 0.9360   |
| 0.3485        | 6.0   | 5358  | 0.2805          | 0.6629    | 0.8247 | 0.7350 | 0.9458   |
| 0.316         | 7.0   | 6251  | 0.2527          | 0.6841    | 0.8415 | 0.7547 | 0.9523   |
| 0.2927        | 8.0   | 7144  | 0.2318          | 0.7022    | 0.8555 | 0.7713 | 0.9567   |
| 0.2743        | 9.0   | 8037  | 0.2157          | 0.7126    | 0.8634 | 0.7808 | 0.9594   |
| 0.2603        | 10.0  | 8930  | 0.2033          | 0.7160    | 0.8667 | 0.7842 | 0.9617   |
| 0.2484        | 11.0  | 9823  | 0.1933          | 0.7249    | 0.8718 | 0.7916 | 0.9635   |
| 0.2394        | 12.0  | 10716 | 0.1854          | 0.7323    | 0.8763 | 0.7979 | 0.9658   |
| 0.2327        | 13.0  | 11609 | 0.1789          | 0.7350    | 0.8791 | 0.8006 | 0.9667   |
| 0.227         | 14.0  | 12502 | 0.1738          | 0.7377    | 0.8802 | 0.8027 | 0.9674   |
| 0.2227        | 15.0  | 13395 | 0.1697          | 0.7385    | 0.8807 | 0.8034 | 0.9675   |
| 0.2197        | 16.0  | 14288 | 0.1666          | 0.7399    | 0.8824 | 0.8049 | 0.9682   |
| 0.2157        | 17.0  | 15181 | 0.1642          | 0.7420    | 0.8841 | 0.8068 | 0.9684   |
| 0.2158        | 18.0  | 16074 | 0.1626          | 0.7433    | 0.8852 | 0.8081 | 0.9686   |
| 0.2134        | 19.0  | 16967 | 0.1616          | 0.7445    | 0.8858 | 0.8090 | 0.9689   |
| 0.2126        | 20.0  | 17860 | 0.1613          | 0.7445    | 0.8858 | 0.8090 | 0.9689   |


### Framework versions

- Transformers 4.57.3
- Pytorch 2.9.1+cu126
- Datasets 4.4.2
- Tokenizers 0.22.2
