"""
File containing the functions for evaluation a NER model, trained via HuggingFace's transformers library

"""
import json
import numpy as np
from typing import Dict
from transformers import EvalPrediction  # already imported in Trainer land
from collections import Counter
from seqeval.metrics import f1_score, precision_score, recall_score

with open('id2label.json', 'r') as f:
    id2label = json.load(f)

# print("Len of id2label: ", len(id2label))

def compute_metrics(eval_pred: EvalPrediction, id2label: Dict):
    """
    Trainer expects compute_metrics to be a callable that takes one argument—eval_pred—and returns a dict of metric_name -> value (floats/ints). Concretely:
    The argument is an EvalPrediction with:
    eval_pred.predictions: model outputs for the eval set. For token classification these are usually logits shaped (batch, seq_len, num_labels).
    eval_pred.label_ids: the gold labels shaped (batch, seq_len).
    """
    # 1) Unpack eval_pred
        # preds = (bs,seq_len, num_labels)
        # labels = (bs, seq_len)
    logits, labels = eval_pred

    # 2) Get highest prediction for each token
    preds = logits.argmax(-1)

    # 3) Mask out position where labels = -100, and create the lists:
            # true_labels: len(true_labels) == len(true_preds)
            # true_labels[0]: true labels for each token in sentence/row 0
    
    true_labels = []
    true_preds = []

    # Iterate through each row (sentence)
    for pred_row, label_row in zip(preds,labels):       
        # store preds and labels for a sentence
        sent_labels = []
        sent_preds = []
        # Loop through each token in the sentence
        for p_id, l_id in zip(pred_row,label_row):  
            if l_id == -100:
                continue # skip special/padding tokens
            sent_preds.append(id2label[str(p_id)])
            sent_labels.append(id2label[str(l_id)])
        
        # Append list of labels/preds for each sentence
        true_labels.append(sent_labels)
        true_preds.append(sent_preds)

    # 5) Precision/recall/F1 with seqeval for each class (average=None -> per label)
    precision = precision_score(true_labels, true_preds, average=None)
    recall = recall_score(true_labels, true_preds, average=None)
    f1 = f1_score(true_labels, true_preds, average=None)

    metrics = {
        "precision": np.array(precision),
        "recall": np.array(recall),
        "f1": np.array(f1),
        "macro_f1": float(np.mean(f1)),
        # weighted_f1 = true_occurences_of_label (TP) * f1
        # micro_f1 = considers sum of all TP, FP, FN across all labels to compute f1
    }
    return metrics

def plot_metrics(metrics: Dict, save_path: str = "per_label_f1.png", top_k: int | None = None):
    """
    Plot per-label F1 using outputs from compute_metrics (which returns per-label arrays).
    - metrics: dict from compute_metrics containing at least "f1". If "labels" is present,
      it will be used as tick labels; otherwise numeric indices are used. If "support" is
      present, bars are sorted by support desc; otherwise keep the given order.
    - save_path: where to save the PNG
    - top_k: if set, plot only the top_k labels by support (or by current order if support missing)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot_per_label_f1_from_metrics] matplotlib not available: {exc}")
        return None

    labels = metrics.get("labels")
    f1s = metrics.get("f1")
    supports = metrics.get("support")

    if f1s is None:
        print("[plot_per_label_f1_from_metrics] Missing f1 in metrics.")
        return None

    # Ensure numpy arrays
    f1s = np.array(f1s)
    if labels is None:
        labels = np.array([f"label_{i}" for i in range(len(f1s))])
    else:
        labels = np.array(labels)
    supports = np.array(supports) if supports is not None else np.arange(len(f1s))[::-1]

    # Sort by support descending if support is present (or by placeholder supports)
    order = np.argsort(-supports)
    labels = labels[order]
    f1s = f1s[order]
    supports = supports[order]

    if top_k is not None:
        labels = labels[:top_k]
        f1s = f1s[:top_k]
        supports = supports[:top_k]

    plt.figure(figsize=(8, max(3, 0.4 * len(labels))))
    plt.barh(labels, f1s, color="steelblue")
    for i, (f1, sup) in enumerate(zip(f1s, supports)):
        plt.text(f1 + 0.01, i, f"f1={f1:.3f}, n={sup}", va="center")
    plt.xlim(0, 1)
    plt.xlabel("F1-score")
    plt.title("Per-label F1 (seqeval)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path

if __name__ == "__main__" :
    import torch

    slice = 3
    dummy_id2label = dict(list(id2label.items())[:slice])
    print("Dummy labels:\n", dummy_id2label)
    
    logits = np.array([
        # 2 samples, 4 tokens, 3 labels 
        [[2.0, 0.5, 0.1], [0.1, 1.0, 0.2], [0.2, 0.2, 0.2], [0.3, 0.4, 0.6]],
        [[0.1, 0.2, 2.5], [0.3, 0.4, 0.1], [0.5, 0.2, 0.3], [1.5, 0.2, 0.1]],
    ])
    label_ids = np.array([
        [0, -100, 1, 2],
        [2, 1, 0, -100],
    ])

    print("logits_pt shape:", logits.shape)
    print("label_ids_pt shape:", label_ids.shape)

    # EvalPrediction would be input to compute_metrics
    eval_pred = EvalPrediction(predictions=logits, label_ids=label_ids)
    preds, labels = eval_pred
    print("Most prob class for each token: ", preds.argmax(-1))
    print(f"Labels ({labels.shape}):", labels)

    print("METRICS:\n")
    metrics = compute_metrics(eval_pred, id2label = dummy_id2label)
    for m in metrics:
        print(f" ==== {m} ====")
        print(metrics[m])
        if hasattr(metrics[m], "shape"):
            print(f"Metrics shape: {metrics[m].shape}")

        print()

