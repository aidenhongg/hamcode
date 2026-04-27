"""Evaluation metrics for the pointwise complexity classifier.

Pointwise:
  - accuracy
  - per-class precision/recall/F1
  - macro-F1 (primary)
  - within-1-tier accuracy (soft correct — classes have a natural order)
  - confusion matrix
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from common.labels import (
    IDX_TO_LABEL,
    NUM_POINT_LABELS,
    POINT_LABELS,
    TIER,
)


def pointwise_metrics(preds: Sequence[int], labels: Sequence[int]) -> dict:
    preds_a = np.asarray(preds)
    labels_a = np.asarray(labels)
    acc = float((preds_a == labels_a).mean()) if len(preds_a) else 0.0
    p, r, f, s = precision_recall_fscore_support(
        labels_a, preds_a,
        labels=list(range(NUM_POINT_LABELS)),
        zero_division=0,
    )
    macro_f1 = float(np.mean(f)) if f.size else 0.0

    # Within-1-tier (ordinal soft-correct)
    pred_tiers = np.asarray([TIER[IDX_TO_LABEL[int(x)]] for x in preds_a])
    true_tiers = np.asarray([TIER[IDX_TO_LABEL[int(x)]] for x in labels_a])
    w1 = float((np.abs(pred_tiers - true_tiers) <= 1).mean()) if len(preds_a) else 0.0

    cm = np.zeros((NUM_POINT_LABELS, NUM_POINT_LABELS), dtype=np.int64)
    for t, p_ in zip(labels_a.tolist(), preds_a.tolist()):
        cm[int(t), int(p_)] += 1

    per_class = {
        POINT_LABELS[i]: {
            "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f[i]), "support": int(s[i]),
        } for i in range(NUM_POINT_LABELS)
    }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "within_1_tier_accuracy": w1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def pointwise_metrics_per_language(
    preds: Sequence[int],
    labels: Sequence[int],
    languages: Sequence[str],
) -> dict:
    """Return overall metrics + a `per_language` dict slicing the same metrics
    by row-language. Use this from train.py / lora_train.py to surface per-
    language collapse early."""
    if not (len(preds) == len(labels) == len(languages)):
        raise ValueError(
            f"len mismatch: preds={len(preds)} labels={len(labels)} "
            f"languages={len(languages)}"
        )
    overall = pointwise_metrics(preds, labels)
    by_lang_idx: dict[str, list[int]] = {}
    for i, lang in enumerate(languages):
        by_lang_idx.setdefault(lang, []).append(i)
    per_language: dict[str, dict] = {}
    for lang, idxs in by_lang_idx.items():
        sub_p = [preds[i] for i in idxs]
        sub_l = [labels[i] for i in idxs]
        m = pointwise_metrics(sub_p, sub_l)
        # Trim the per-class confusion to keep the JSON small; keep the headline rows.
        per_language[lang] = {
            "n": len(idxs),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "within_1_tier_accuracy": m["within_1_tier_accuracy"],
            "per_class_f1": {cls: stats["f1"]
                              for cls, stats in m["per_class"].items()},
        }
    overall["per_language"] = per_language
    return overall


def pretty_confusion(matrix: list[list[int]], labels: Sequence[str]) -> str:
    """Small pretty-printer for confusion matrices."""
    w = max(len(l) for l in labels) + 1
    header = " " * (w + 1) + " ".join(f"{l[:6]:>6}" for l in labels)
    lines = [header]
    for i, row in enumerate(matrix):
        lines.append(f"{labels[i][:w]:<{w}} " + " ".join(f"{v:>6d}" for v in row))
    return "\n".join(lines)


def spearman_rank(pred_tiers: Iterable[int], true_tiers: Iterable[int]) -> float:
    """Spearman rank correlation between two tier sequences."""
    a = np.asarray(list(pred_tiers), dtype=float)
    b = np.asarray(list(true_tiers), dtype=float)
    if len(a) < 2:
        return 0.0
    ra = _rankdata(a); rb = _rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float) + 1.0
    return ranks
