"""
metrics.py

Comprehensive evaluation metrics for the HAR multiclass classifier.
Computes confusion matrix, precision, recall, F1-score, and AUC,
then saves visualisations and numeric results.

Usage:
    from metrics import compute_all_metrics

    metrics = compute_all_metrics(
        model, test_dataset, class_names=ACTIVITY_LABELS,
        output_dir="logs/metrics/ffn"
    )

Version: 1.0
Date: 20-05-2026
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from logger_setup import logger


def _collect_predictions(model, dataset):
    """Run the model on a tf.data.Dataset and collect predictions + labels.

    Args:
        model:   A compiled Keras model.
        dataset: A batched tf.data.Dataset yielding (features, labels).

    Returns:
        y_true:  numpy array of integer labels, shape (N,).
        y_pred:  numpy array of predicted class indices, shape (N,).
        y_proba: numpy array of class probabilities, shape (N, num_classes).
    """
    all_labels = []
    all_proba = []

    for features, labels in dataset:
        proba = model(features, training=False)
        all_labels.append(labels.numpy())
        all_proba.append(proba.numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_proba = np.concatenate(all_proba, axis=0)
    y_pred = np.argmax(y_proba, axis=1)

    return y_true, y_pred, y_proba


def plot_confusion_matrix(cm, class_names, output_dir, normalize=True):
    """Plot and save a confusion matrix heatmap.

    Args:
        cm:          Confusion matrix array, shape (n, n).
        class_names: List of class name strings.
        output_dir:  Directory to save the plot.
        normalize:   If True, normalize rows to show percentages.
    """
    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_display = cm_norm
        fmt = ".2%"
        title = "Confusion Matrix (Normalized)"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Tick labels
    tick_marks = np.arange(len(class_names))
    short_names = [n.replace("WALKING_", "W_") for n in class_names]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(short_names, fontsize=11)

    # Cell annotations
    thresh = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            value = cm_display[i, j]
            text = f"{value:{fmt}}" if normalize else f"{value:{fmt}}"
            ax.text(
                j, i, text,
                ha="center", va="center", fontsize=10,
                color="white" if value > thresh else "black",
            )

    ax.set_ylabel("True Label", fontsize=13)
    ax.set_xlabel("Predicted Label", fontsize=13)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {save_path}")

    return save_path


def compute_all_metrics(model, dataset, class_names, output_dir):
    """Compute and save all evaluation metrics.

    Generates:
        - Confusion matrix heatmap (PNG)
        - Per-class precision, recall, F1
        - Macro-averaged AUC score
        - Summary JSON file

    Args:
        model:       A compiled/loaded Keras model.
        dataset:     A batched tf.data.Dataset yielding (features, labels).
        class_names: List of activity label strings.
        output_dir:  Directory where plots and metrics JSON are saved.

    Returns:
        A dict with all computed metrics.
    """
    logger.info("Computing evaluation metrics...")

    y_true, y_pred, y_proba = _collect_predictions(model, dataset)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, output_dir, normalize=True)

    # --- Classification Report (precision, recall, f1 per class) ---
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Log the text version
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    logger.info(f"\nClassification Report:\n{report_text}")

    # Macro metrics
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # AUC (one-vs-rest, macro)
    try:
        auc_macro = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        )
    except ValueError as e:
        logger.warning(f"Could not compute AUC: {e}")
        auc_macro = None

    # Accuracy 
    accuracy = np.mean(y_true == y_pred)

    # Assemble summary
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "auc_macro": float(auc_macro) if auc_macro is not None else None,
        "per_class": {},
        "confusion_matrix": cm.tolist(),
    }

    for cls_name in class_names:
        if cls_name in report:
            metrics["per_class"][cls_name] = {
                "precision": report[cls_name]["precision"],
                "recall": report[cls_name]["recall"],
                "f1-score": report[cls_name]["f1-score"],
                "support": report[cls_name]["support"],
            }

    # Log summary
    logger.success(
        f"Metrics Summary — "
        f"Accuracy: {accuracy:.4f} | "
        f"F1 (macro): {f1_macro:.4f} | "
        f"Precision: {precision_macro:.4f} | "
        f"Recall: {recall_macro:.4f} | "
        f"AUC: {auc_macro:.4f}" if auc_macro else
        f"Metrics Summary — "
        f"Accuracy: {accuracy:.4f} | "
        f"F1 (macro): {f1_macro:.4f} | "
        f"Precision: {precision_macro:.4f} | "
        f"Recall: {recall_macro:.4f} | "
        f"AUC: N/A"
    )

    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics JSON saved to {json_path}")

    return metrics


def log_metrics_to_tensorboard(metrics_dict, log_dir, step=0):
    """Write scalar metrics to TensorBoard.

    Args:
        metrics_dict: Dict with metric names → float values.
        log_dir:      TensorBoard log directory.
        step:         Global step for the summary writer.
    """
    writer = tf.summary.create_file_writer(log_dir)
    scalar_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_macro"]

    with writer.as_default():
        for key in scalar_keys:
            value = metrics_dict.get(key)
            if value is not None:
                tf.summary.scalar(f"eval/{key}", value, step=step)

        # Per-class F1
        per_class = metrics_dict.get("per_class", {})
        for cls_name, cls_metrics in per_class.items():
            f1_val = cls_metrics.get("f1-score")
            if f1_val is not None:
                safe_name = cls_name.replace(" ", "_")
                tf.summary.scalar(f"eval/f1_{safe_name}", f1_val, step=step)

    writer.flush()
    logger.info(f"Metrics written to TensorBoard at {log_dir}")
