"""Classification metrics and decision-threshold tuning."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def compute_metrics(y_true, prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) > 1
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob) if both_classes else float("nan"),
        "pr_auc": average_precision_score(y_true, prob) if both_classes else float("nan"),
    }


def best_threshold(y_true, prob, metric: str = "macro_f1") -> float:
    """Threshold that maximises ``metric`` on a validation set (never on test)."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [compute_metrics(y_true, prob, t)[metric] for t in grid]
    return float(grid[int(np.argmax(scores))])
