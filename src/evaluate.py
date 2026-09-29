"""
evaluate.py - Evaluation Metrics, Diagnostics, and Visualization Utilities.
"""

from typing import Dict, List, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Computes all standard binary classification metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion matrix components (TP, TN, FP, FN)

    Args:
        y_true: True binary labels shape (m, 1) or (m,).
        y_pred: Predicted binary labels shape (m, 1) or (m,).
        target_names: Names for class 0 and 1 (default ['Benign', 'Malignant']).

    Returns:
        metrics_dict: Dictionary containing metric values.
    """
    y_t = y_true.flatten().astype(int)
    y_p = y_pred.flatten().astype(int)

    acc = accuracy_score(y_t, y_p)
    prec = precision_score(y_t, y_p, zero_division=0)
    rec = recall_score(y_t, y_p, zero_division=0)
    f1 = f1_score(y_t, y_p, zero_division=0)
    cm = confusion_matrix(y_t, y_p)

    # In binary classification where 0=Benign, 1=Malignant:
    # TN: Benign correctly classified as Benign
    # FP: Benign incorrectly classified as Malignant
    # FN: Malignant incorrectly classified as Benign
    # TP: Malignant correctly classified as Malignant
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm,
    }
    return metrics


def print_evaluation_report(
    metrics: Dict[str, float],
    split_name: str = "Fixed 20% Held-Out Test Split"
) -> None:
    """Prints a clear summary report of classification performance on benchmark data."""
    cm = metrics["confusion_matrix"]
    tn = metrics["true_negatives"]
    fp = metrics["false_positives"]
    fn = metrics["false_negatives"]
    tp = metrics["true_positives"]

    print("=" * 70)
    print(f"             MODEL PERFORMANCE REPORT [{split_name.upper()}]")
    print("=" * 70)
    print(f"  Accuracy  : {metrics['accuracy'] * 100:.2f}%  (Target: > 80.00%)")
    print(f"  Precision : {metrics['precision'] * 100:.2f}%  (Malignant Precision)")
    print(f"  Recall    : {metrics['recall'] * 100:.2f}%  (Malignant Sensitivity)")
    print(f"  F1 Score  : {metrics['f1_score'] * 100:.2f}%")
    print("-" * 70)
    print("  CONFUSION MATRIX BREAKDOWN:")
    print(f"    * True Negatives  (Benign -> Benign)       : {tn:>4}")
    print(f"    * False Positives (Benign -> Malignant)    : {fp:>4}  (Type I Error)")
    print(f"    * False Negatives (Malignant -> Benign)    : {fn:>4}  (Type II Error)")
    print(f"    * True Positives  (Malignant -> Malignant) : {tp:>4}")
    print("-" * 70)
    print(f"  Confusion Matrix Grid:")
    print(f"                     Predicted Benign (0)  Predicted Malignant (1)")
    print(f"    Actual Benign   :        {tn:^10}            {fp:^10}")
    print(f"    Actual Malignant:        {fn:^10}            {tp:^10}")
    print("=" * 70)


def plot_loss_curve(
    history: Dict[str, List[float]],
    save_path: str = "plots/training_loss.png"
) -> None:
    """
    Plots training loss (and validation loss if available) across epochs.
    Saves figure to specified directory.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(epochs, history["train_loss"], label="Training Loss (BCE)", color="#1f77b4", linewidth=2)

    if "val_loss" in history and len(history["val_loss"]) > 0:
        plt.plot(epochs, history["val_loss"], label="Validation Loss", color="#ff7f0e", linestyle="--", linewidth=2)

    plt.title("Neural Network Optimization Loss Curve", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    print(f"[Plot Saved] Loss curve successfully exported to: {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    target_names: List[str] = ["Benign", "Malignant"],
    save_path: str = "plots/confusion_matrix.png"
) -> None:
    """
    Plots a visually clear Confusion Matrix heatmap annotated with raw counts
    and percentages.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Breast Cancer Classification Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, fontsize=11)
    plt.yticks(tick_marks, target_names, fontsize=11)

    total_samples = np.sum(cm)
    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / total_samples) * 100
            plt.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > thresh else "black",
                fontsize=12,
                fontweight="bold"
            )

    plt.ylabel("Actual True Diagnosis", fontsize=11, labelpad=8)
    plt.xlabel("Model Predicted Diagnosis", fontsize=11, labelpad=8)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    print(f"[Plot Saved] Confusion matrix heatmap successfully exported to: {save_path}")
