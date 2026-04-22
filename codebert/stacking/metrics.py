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


def mcnemar_p(preds_a: np.ndarray, preds_b: np.ndarray, y: np.ndarray) -> float:
    """McNemar's test (continuity-corrected) on matched predictions.

    Returns p-value for H0: both classifiers make the same errors.
    preds_a is the baseline, preds_b is the new model.
    """
    from math import exp, lgamma

    a_correct = (preds_a == y)
    b_correct = (preds_b == y)
    # b01: a wrong, b right;   b10: a right, b wrong
    b01 = int(((~a_correct) & b_correct).sum())
    b10 = int((a_correct & (~b_correct)).sum())
    n = b01 + b10
    if n == 0:
        return 1.0
    # Exact binomial two-sided under p=0.5
    k = min(b01, b10)
    # P(X <= k) with p=0.5, two-sided → 2 * sum_{i<=k} C(n,i) * 0.5^n
    log_half_n = -n * np.log(2)
    log_sum = -np.inf
    for i in range(k + 1):
        log_cnk = lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
        log_term = log_cnk + log_half_n
        # log-sum-exp
        m = max(log_sum, log_term)
        log_sum = m + np.log(np.exp(log_sum - m) + np.exp(log_term - m))
    p = 2.0 * float(np.exp(log_sum))
    return min(1.0, p)


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs_pos: np.ndarray,
    bert_pair_pred: np.ndarray | None = None,
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

    if bert_pair_pred is not None:
        # Compare head vs bert baseline on the same filtered subset.
        # bert_pair_pred uses ternary encoding; map A_faster->1, same->0 (matches our y).
        bert_acc = float(accuracy_score(y_true, bert_pair_pred))
        p_mc = mcnemar_p(bert_pair_pred, y_pred, y_true)
        out["bert_pairwise_baseline_comparison"] = {
            "bert_same_vs_A_subset_acc": bert_acc,
            "head_same_vs_A_subset_acc": acc,
            "delta": acc - bert_acc,
            "mcnemar_p": p_mc,
        }

    return out
