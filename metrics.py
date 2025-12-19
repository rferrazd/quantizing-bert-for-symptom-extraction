"""
File containing the functions for evaluation a NER model, trained via HuggingFace's transformers library

"""
import time
import json
import numpy as np
from typing import Dict, Union
import evaluate
#from transformers import EvalPrediction, PredictionOutput
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
)
seqeval = evaluate.load("seqeval")
with open('id2label.json', 'r') as f:
    id2label = json.load(f)

# FUNCTION TO RETURN OVERALL METRICS
def compute_metrics(eval_pred: Union[EvalPrediction,PredictionOutput], id2label: Dict = id2label, save_path:str|None = None):
    """
    Returns overall metrics and per-entity and per-token labels
    Watch video for reference: https://www.youtube.com/watch?v=ujubwa_oa-0 (1:05:00)
    """
    # 1) Unpack eval_pred

    # Check datatype: 
    if isinstance(eval_pred, EvalPrediction):
        # unpack
        predictions, labels = eval_pred
    elif isinstance(eval_pred, PredictionOutput):
        # unpack
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        raise TypeError("Expected eval_pred to be of type EvalPrediction or PredictionOutput, got {}.".format(type(eval_pred)))

    predictions = np.argmax(predictions, axis =-1) # axis=2
    # Remove ignored index (e.g., padding tokens) and convert to actual labels
    true_predictions = [
        [id2label[str(p)] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[str(l)] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute the scores using seqeval
    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    # Extract overall metrics
    metrics = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
    
    # Include all per-entity metrics (keys starting with "SYMPTOM_")
    for key, value in results.items():
        if key.startswith("SYMPTOM_"):
            metrics[key] = value

    if save_path:
        # Save metrics in a json
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=1, default=str)  # default=str handles numpy types
        
    return metrics

def plot_metrics(metrics: Dict, save_path: str | None = None, top_k: int | None = None, bins: int = 20):
    """
    Plot histogram of per-entity F1 scores for entities starting with "SYMPTOM_".
    - metrics: dict from compute_metrics containing per-entity metrics (keys starting with "SYMPTOM_")
    - save_path: where to save the PNG (optional, if None the plot is only displayed)
    - bins: number of bins for the histogram (default: 20)
    
    Returns:
        tuple: (save_path or None, data_dict) where data_dict groups entity labels by F1 score bins.
               Each bin key maps to a dict with:
               - "count": number of entities in the bin
               - "entities": list of entity labels in the bin
               - "bin_range": tuple of (bin_start, bin_end)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot_metrics] matplotlib not available: {exc}")
        return None, {}

    # Extract entity keys that start with "SYMPTOM_"
    entity_keys = [k for k in metrics.keys() if k.startswith("SYMPTOM_")]
    
    if not entity_keys:
        print("[plot_metrics] No entity keys found starting with 'SYMPTOM_' in metrics.")
        return None, {}
    
    # Extract F1 scores for each entity (keep track of which entity has which F1)
    entity_f1s = []
    entity_label_to_f1 = {}  # Map entity label to its F1 score
    
    for entity_key in entity_keys:
        entity_metrics = metrics[entity_key]
        # Handle both dict format (from seqeval.compute) and direct value
        if isinstance(entity_metrics, dict):
            f1_score = entity_metrics.get("f1", 0.0)
        else:
            f1_score = entity_metrics
        f1_score = float(f1_score)
        entity_f1s.append(f1_score)
        entity_label_to_f1[entity_key] = f1_score
    
    # Convert to numpy array
    entity_f1s = np.array(entity_f1s)
    
    # Get overall F1 for reference
    overall_f1 = metrics.get("f1", None)
    
    # Create the histogram to get bin edges
    plt.figure(figsize=(10, 6))
    n, bins_edges, patches = plt.hist(entity_f1s, bins=bins, range=(0, 1), 
                                       color="steelblue", edgecolor="black", alpha=0.7)
    
    # Group entity labels by bins
    data = {}
    for i in range(len(bins_edges) - 1):
        bin_start = bins_edges[i]
        bin_end = bins_edges[i + 1]
        # Create bin label
        bin_label = f"{bin_start:.3f}-{bin_end:.3f}"
        
        # Find all entities that fall into this bin
        # For the last bin, include entities equal to bin_end (1.0)
        if i == len(bins_edges) - 2:
            entities_in_bin = [entity_key for entity_key in entity_keys 
                             if bin_start <= entity_label_to_f1[entity_key] <= bin_end]
        else:
            entities_in_bin = [entity_key for entity_key in entity_keys 
                             if bin_start <= entity_label_to_f1[entity_key] < bin_end]
        
        data[bin_label] = {
            "count": len(entities_in_bin),
            "entities": entities_in_bin,
            "bin_range": (float(bin_start), float(bin_end))
        }
    
    # Color bars based on F1 score ranges (optional visual enhancement)
    for i, patch in enumerate(patches):
        bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
        if bin_center >= 0.8:
            patch.set_facecolor("green")
        elif bin_center >= 0.5:
            patch.set_facecolor("orange")
        else:
            patch.set_facecolor("red")
    
    # Add vertical line for overall F1 if available
    if overall_f1 is not None:
        plt.axvline(x=overall_f1, color="purple", linestyle="--", linewidth=2, 
                   label=f"Overall F1: {overall_f1:.4f}")
        plt.legend()
    
    # Add statistics text
    stats_text = (
        f"Total entities: {len(entity_f1s)}\n"
        f"Mean F1: {np.mean(entity_f1s):.4f}\n"
        f"Median F1: {np.median(entity_f1s):.4f}\n"
        f"F1 = 1.0: {np.sum(entity_f1s == 1.0)} ({100*np.sum(entity_f1s == 1.0)/len(entity_f1s):.1f}%)\n"
        f"F1 = 0.0: {np.sum(entity_f1s == 0.0)} ({100*np.sum(entity_f1s == 0.0)/len(entity_f1s):.1f}%)"
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
             fontsize=9)
    
    plt.xlabel("F1-score", fontsize=12)
    plt.ylabel("Number of Entities", fontsize=12)
    plt.title("Distribution of Per-Entity F1 Scores (seqeval)", fontsize=14, fontweight="bold")
    plt.xlim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    # Only save if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path, data
    else:
        # Display the plot
        plt.show()
        return None, data

if __name__ == "__main__" :

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

    # 5) Compute macro F1 (average of per-label F1s)
    # Macro-averaged metrics first compute precision, recall, and F1 for each class/label independently,
    # and then take the unweighted mean across labels:
    #
    # - F1_per_label = F1 for label1, F1 for label2, ...
    # - Macro F1 = mean(F1_per_label)
    #
    # Macro F1 treats all classes equally, regardless of their frequency, so it is sensitive to how well
    # the model does on rare labels. If performance on rare entities is critical, macro F1 is a key metric.

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
    print("Overall metrics:")
    for m in ["precision", "recall", "f1", "accuracy"]:
        if m in metrics:
            print(f"  {m}: {metrics[m]}")
    
    # Extract per-entity metrics
    entity_keys = [k for k in metrics.keys() if k.startswith("SYMPTOM_")]
    print(f"\n=== PER-ENTITY METRICS (from compute_metrics) ===\n")
    print(f"Number of entities: {len(entity_keys)}")
    if entity_keys:
        # Calculate macro F1 from per-entity metrics
        entity_f1s = []
        for entity_key in entity_keys:
            entity_metrics = metrics[entity_key]
            if isinstance(entity_metrics, dict):
                f1_score = entity_metrics.get("f1", 0.0)
            else:
                f1_score = entity_metrics
            entity_f1s.append(float(f1_score))
        macro_f1 = np.mean(entity_f1s) if entity_f1s else 0.0
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Sample per-entity F1s (first 5): {entity_f1s[:5]}")

    print("\n=== PLOT RESULTS  ===\n")
    path = plot_metrics(metrics=metrics, save_path="dummy_per_entity_f1.png", top_k=10)
    if path:
        print(f"Plot saved at {path}")
