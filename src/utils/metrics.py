from __future__ import annotations

from typing import Dict, List, Sequence, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


def binary_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def multiclass_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def method_metrics_fake_only(
    y_method_true: Sequence[int],
    y_method_pred: Sequence[int],
) -> Dict[str, float]:
    y_true = np.asarray(y_method_true).astype(int)
    y_pred = np.asarray(y_method_pred).astype(int)
    mask = y_true >= 0
    if mask.sum() == 0:
        return {"method_acc": 0.0, "method_macro_f1": 0.0, "method_weighted_f1": 0.0}
    m = multiclass_metrics(y_true[mask], y_pred[mask])
    return {
        "method_acc": m["acc"],
        "method_macro_f1": m["macro_f1"],
        "method_weighted_f1": m["weighted_f1"],
    }


def joint_5class_from_multitask(
    y_rf_prob: Sequence[float],
    y_method_pred: Sequence[int],
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert multitask outputs into 5-class predictions: 0=Real, 1~4=fake methods."""
    y_rf_prob = np.asarray(y_rf_prob).astype(float)
    y_method_pred = np.asarray(y_method_pred).astype(int)
    is_fake = y_rf_prob >= threshold
    joint = np.zeros_like(y_method_pred, dtype=int)
    joint[is_fake] = y_method_pred[is_fake] + 1
    return joint


def classification_report_dict(y_true: Sequence[int], y_pred: Sequence[int], target_names: Optional[List[str]] = None) -> Dict:
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )


def confusion_matrix_array(y_true: Sequence[int], y_pred: Sequence[int], labels: Optional[List[int]] = None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)
