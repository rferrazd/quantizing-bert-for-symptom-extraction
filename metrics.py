"""
File containing the functions for evaluation a NER model, trained via HuggingFace's transformers library

"""
import time
import json
import numpy as np
from typing import Dict, Union
from seqeval.metrics import f1_score, precision_score, recall_score
#from transformers import EvalPrediction, PredictionOutput
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
)

with open('id2label.json', 'r') as f:
    id2label = json.load(f)

# FUNCTION TO RETURN OVERALL METRICS
def compute_metrics(eval_pred: EvalPrediction, id2label: Dict, save_path:str|None = None):
    """
    Trainer expects compute_metrics to be a callable that takes one argument—eval_pred—and returns a dict of metric_name -> value (floats/ints). Concretely:
    The argument is an EvalPrediction with:
    eval_pred.predictions: model outputs for the eval set. For token classification these are usually logits shaped (batch, seq_len, num_labels).
    eval_pred.label_ids: the gold labels shaped (batch, seq_len).
    
    Returns only overall metrics (single values) to avoid printing long per-label arrays.
    Use compute_per_label_metrics() separately for detailed per-label analysis.
    """
    # 1) Unpack eval_pred
    logits, labels = eval_pred

    # 2) Get highest prediction for each token
    preds = logits.argmax(-1)

    # 3) Mask out positions where labels = -100, and create the lists:
    true_labels = []
    true_preds = []

    # Iterate through each row (sentence)
    for pred_row, label_row in zip(preds, labels):       
        # Store preds and labels for a sentence
        sent_labels = []
        sent_preds = []
        # Loop through each token in the sentence
        for p_id, l_id in zip(pred_row, label_row):  
            if l_id == -100:
                continue  # Skip special/padding tokens
            sent_preds.append(id2label[str(p_id)])
            sent_labels.append(id2label[str(l_id)])
        
        # Append list of labels/preds for each sentence
        true_labels.append(sent_labels)
        true_preds.append(sent_preds)

    # 4) Compute overall metrics (micro-averaged) - single values

    # Micro-averaged metrics aggregate the contributions of all classes to compute the average metric.
    # In the context of NER sequence evaluation, micro F1 considers all tokens together and calculates
    # the global number of true positives, false positives, and false negatives. So:
    #
    # - Micro Precision = (sum of true positives for all classes) / (sum of predicted positives for all classes)
    # - Micro Recall    = (sum of true positives for all classes) / (sum of actual positives for all classes)
    # - Micro F1        = harmonic mean of micro-precision and micro-recall
    #
    # Micro F1 is usually dominated by the most frequent classes and gives more weight to performance on them.

    precision_micro = precision_score(true_labels, true_preds, average='micro')
    recall_micro = recall_score(true_labels, true_preds, average='micro')
    f1_micro = f1_score(true_labels, true_preds, average='micro')
    
    # 5) Compute macro F1 (average of per-label F1s)
    # Macro-averaged metrics first compute precision, recall, and F1 for each class/label independently,
    # and then take the unweighted mean across labels:
    #
    # - F1_per_label = F1 for label1, F1 for label2, ...
    # - Macro F1 = mean(F1_per_label)
    #
    # Macro F1 treats all classes equally, regardless of their frequency, so it is sensitive to how well
    # the model does on rare labels. If performance on rare entities is critical, macro F1 is a key metric.

    f1_per_label = f1_score(true_labels, true_preds, average=None)
    macro_f1 = float(np.mean(f1_per_label))
    
    # 6) Compute token-level accuracy
    total_tokens = 0
    correct_tokens = 0
    for pred_row, label_row in zip(preds, labels):
        for p_id, l_id in zip(pred_row, label_row):
            if l_id == -100:
                continue
            total_tokens += 1
            if p_id == l_id:
                correct_tokens += 1
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    # Return only overall metrics (single values) - these will be printed
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision_micro),
        "recall": float(recall_micro),
        "f1": float(f1_micro),
        "macro_f1": float(macro_f1),
        "timestamp" : time.strftime("%Y%m%d_%H%M%S")
    }

    if save_path:
        # Save metrics in a json
        with open(save_path, 'w') as f:
            json.dump(metrics,f,indent=1)

    return metrics

# FUNCTION TO RETURN PER-LABEL METRICS
def compute_per_label_metrics(eval_pred: Union[EvalPrediction,PredictionOutput], id2label: Dict, save_path:str|None = None):
    """
    Compute detailed per-label metrics. Use this function separately when you need
    per-label precision, recall, and F1 scores for analysis.
    
    Args:
        eval_pred: EvalPrediction object with predictions and label_ids
        id2label: Dictionary mapping label IDs to label names
    
    Returns:
        Dictionary with per-label metrics:
        - "precision": list of per-label precision scores
        - "recall": list of per-label recall scores
        - "f1": list of per-label F1 scores
        - "macro_f1": average F1 across all labels
        - "label_names": list of label names in order
    """
    # 1) Unpack eval_pred

    # Check datatype: 
    if isinstance(eval_pred, EvalPrediction):
        # unpack
        logits, labels = eval_pred
    elif isinstance(eval_pred, PredictionOutput):
        # unpack
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        raise TypeError("Expected eval_pred to be of type EvalPrediction or PredictionOutput, got {}.".format(type(eval_pred)))


    # 2) Get highest prediction for each token
    preds = logits.argmax(-1)

    # 3) Mask out position where labels = -100, and create the lists:
    true_labels = []
    true_preds = []

    # Iterate through each row (sentence)
    for pred_row, label_row in zip(preds, labels):       
        # store preds and labels for a sentence
        sent_labels = []
        sent_preds = []
        # Loop through each token in the sentence
        for p_id, l_id in zip(pred_row, label_row):  
            if l_id == -100:
                continue # skip special/padding tokens
            sent_preds.append(id2label[str(p_id)])
            sent_labels.append(id2label[str(l_id)])
        
        # Append list of labels/preds for each sentence
        true_labels.append(sent_labels)
        true_preds.append(sent_preds)

    # 4) Precision/recall/F1 with seqeval for each class (average=None -> per label)
    precision = precision_score(true_labels, true_preds, average=None)
    recall = recall_score(true_labels, true_preds, average=None)
    f1 = f1_score(true_labels, true_preds, average=None)

    # 5) Get label names in order
    label_names = [id2label[str(i)] for i in range(len(f1))]

    # Cast numpy types to Python primitives/lists so it's JSON-serializable
    metrics = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "macro_f1": float(np.mean(f1)),
        "label_names": label_names,
        "timestamp" : time.strftime("%Y%m%d_%H%M%S")
    }

    if save_path:
        # Save metrics in a json
        with open(save_path, 'w') as f:
            json.dump(metrics,f,indent=1)
        
    return metrics



def plot_metrics(metrics: Dict, save_path: str = "per_label_f1.png", top_k: int | None = None):
    """
    Plot per-label F1 using outputs from compute_per_label_metrics (which returns per-label arrays).
    - save_path: where to save the PNG
    - top_k: if set, plot only the top_k labels 
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot_per_label_f1_from_metrics] matplotlib not available: {exc}")
        return None

    labels = metrics.get("label_names")
    f1s = metrics.get("f1")

    if f1s is None:
        print("[plot_per_label_f1_from_metrics] Missing f1 in metrics.")
        return None

    # Ensure numpy arrays
    f1s = np.array(f1s)
    if labels is None:
        labels = np.array([f"label_{i}" for i in range(len(f1s))])
    else:
        labels = np.array(labels)

    # Sort by support descending order highest --> lowest support
    order = np.argsort(-f1s)
    labels = labels[order]
    f1s = f1s[order]

    if top_k is not None:
        labels = labels[:top_k]
        f1s = f1s[:top_k]


    plt.figure(figsize=(8, max(3, 0.4 * len(labels))))
    plt.barh(labels, f1s, color="steelblue")
    for i, f1 in enumerate(f1s):
        plt.text(f1 + 0.01, i, f"f1={f1:.3f}", va="center")
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

    print("\n=== OVERALL METRICS (from compute_metrics) ===\n")
    metrics = compute_metrics(eval_pred, id2label=dummy_id2label)
    for m in metrics:
        print(f"{m}: {metrics[m]}")

    print("\n=== PER-LABEL METRICS (from compute_per_label_metrics) ===\n")
    per_label_metrics = compute_per_label_metrics(eval_pred, id2label=dummy_id2label)
    print(f"Number of labels: {len(per_label_metrics['f1'])}")
    print(f"Macro F1: {per_label_metrics['macro_f1']:.4f}")
    print(f"Sample per-label F1s: {per_label_metrics['f1'][:5]}")

    print("\n=== PLOT RESULTS  ===\n")
    path = plot_metrics(metrics=per_label_metrics,save_path = "dummy_per_label_f1.png", top_k = None)
    print(f"plot saved at {path}")
