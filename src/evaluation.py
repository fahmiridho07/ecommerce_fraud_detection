"""Evaluation metric helpers for experiment scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | list[list[int]] | None]:
    """Compute thesis metrics from labels, fraud scores, and a fixed threshold."""
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "threshold": float(threshold),
        "confusion_matrix": cm.astype(int).tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return ROC-AUC when both classes are present."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def threshold_selection_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate validation metrics across thresholds from 0.01 to 0.99."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    rows = []
    for threshold in thresholds:
        metrics = binary_classification_metrics(y_true, y_score, float(threshold))
        rows.append(
            {
                "threshold": metrics["threshold"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )
    table = pd.DataFrame(rows)
    best_index = table.sort_values(
        ["mcc", "f1", "threshold"],
        ascending=[False, False, True],
    ).index[0]
    table["selected"] = False
    table.loc[best_index, "selected"] = True
    return table


def selected_threshold_from_table(threshold_table: pd.DataFrame) -> float:
    """Return the validation-selected threshold."""
    selected = threshold_table.loc[threshold_table["selected"], "threshold"]
    if selected.empty:
        raise ValueError("No selected threshold found.")
    return float(selected.iloc[0])


def confusion_matrix_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: dict[str, float],
    split_name: str,
) -> pd.DataFrame:
    """Build a long-format confusion matrix table for one split."""
    rows = []
    for threshold_name, threshold in thresholds.items():
        y_pred = (y_score >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        for true_label in (0, 1):
            for predicted_label in (0, 1):
                rows.append(
                    {
                        "split": split_name,
                        "threshold_type": threshold_name,
                        "threshold": float(threshold),
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(cm[true_label, predicted_label]),
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit("Evaluation helpers are imported by experiment scripts.")
