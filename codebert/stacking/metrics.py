"""Metrics for the binary head (same vs A_faster on the B>=A subset)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


LABEL_NAMES = ("same", "A_faster")   # indices 0 and 1


def expected_calibration_error(probs_pos: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Reliability-diagram ECE on the positive-class probability."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    if total == 0:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (probs_pos >= lo) & (probs_pos < hi if i < n_bins - 1 else probs_pos <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        conf = float(probs_pos[in_bin].mean())
        acc = float(y[in_bin].mean())  # fraction positive in bin
        ece += abs(conf - acc) * n / total
    return float(ece)


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs_pos: np.ndarray,
) -> dict:
    """Return a comprehensive metrics dict for logging."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0,
    )
    macro_f1 = float(np.mean(f))
    try:
        roc = float(roc_auc_score(y_true, probs_pos)) if len(set(y_true.tolist())) > 1 else float("nan")
    except ValueError:
        roc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, probs_pos)) if len(set(y_true.tolist())) > 1 else float("nan")
    except ValueError:
        pr_auc = float("nan")
    brier = float(brier_score_loss(y_true, probs_pos)) if len(set(y_true.tolist())) > 1 else float("nan")
    ece = expected_calibration_error(probs_pos, y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    out = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in range(2)
        },
        "confusion_matrix": cm,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "ece": ece,
    }

    return out
