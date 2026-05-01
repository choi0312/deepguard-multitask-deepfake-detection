from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(
    rf_targets: np.ndarray,
    rf_probs: np.ndarray,
    method_targets: np.ndarray,
    method_preds: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute Task 1, Task 2, and joint metrics."""
    rf_preds = (rf_probs >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "task1_accuracy": float(accuracy_score(rf_targets, rf_preds)),
        "task1_f1": float(f1_score(rf_targets, rf_preds, zero_division=0)),
    }

    fake_mask = method_targets >= 0
    if fake_mask.any():
        metrics["task2_accuracy"] = float(
            accuracy_score(method_targets[fake_mask], method_preds[fake_mask])
        )
        metrics["task2_macro_f1"] = float(
            f1_score(
                method_targets[fake_mask],
                method_preds[fake_mask],
                average="macro",
                zero_division=0,
            )
        )
    else:
        metrics["task2_accuracy"] = 0.0
        metrics["task2_macro_f1"] = 0.0

    joint_targets = np.where(rf_targets == 0, 0, method_targets + 1)
    joint_preds = np.where(rf_preds == 0, 0, method_preds + 1)
    metrics["joint_5class_accuracy"] = float(accuracy_score(joint_targets, joint_preds))

    return metrics
