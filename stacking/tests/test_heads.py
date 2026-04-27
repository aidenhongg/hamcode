"""Unit tests for the top-4 heads on a tiny linearly-separable synthetic dataset.

Each head must: fit, predict, predict_proba, save, load, and produce
non-trivial accuracy on a synthetic problem.
"""

from __future__ import annotations

# Load torch before xgboost/lightgbm — Windows DLL search-path conflict otherwise.
import torch  # noqa: F401

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stacking.heads import get_head
from stacking.heads.base import HeadRegistry, compute_class_weight


def _make_data(n: int = 400, d: int = 10, seed: int = 0, imbalance: float = 1.0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    X = rng.standard_normal((n, d)).astype(np.float32)
    score = X @ w
    y = (score > 0).astype(np.int64)
    # enforce imbalance
    if imbalance != 1.0:
        class1 = np.where(y == 1)[0]
        keep = int(len(class1) * imbalance)
        drop = class1[keep:]
        mask = np.ones(n, dtype=bool); mask[drop] = False
        X = X[mask]; y = y[mask]
    return X, y


HEAD_NAMES = ["xgb", "lgbm", "mlp", "stacked"]


@pytest.mark.parametrize("head_name", HEAD_NAMES)
def test_head_fit_and_accuracy(head_name):
    X, y = _make_data(n=400, d=10, seed=42)
    split = int(0.8 * len(y))
    Xtr, ytr = X[:split], y[:split]
    Xva, yva = X[split:], y[split:]

    head = get_head(head_name, seed=42)
    summary = head.fit(Xtr, ytr, Xva, yva, class_weight=None)
    preds = head.predict(Xva)
    probs = head.predict_proba(Xva)

    assert preds.shape == (len(yva),)
    assert probs.shape == (len(yva), 2)
    # Probabilities sum to ~1
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    # Non-trivial accuracy on linearly separable problem
    acc = (preds == yva).mean()
    assert acc >= 0.7, f"{head_name}: accuracy {acc:.3f} below 0.7"


@pytest.mark.parametrize("head_name", HEAD_NAMES)
def test_head_save_load_round_trip(head_name):
    X, y = _make_data(n=200, d=8, seed=7)
    Xtr, ytr = X[:160], y[:160]
    Xva, yva = X[160:], y[160:]

    head = get_head(head_name, seed=7)
    head.fit(Xtr, ytr, Xva, yva)
    preds_before = head.predict(Xva)
    probs_before = head.predict_proba(Xva)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "head"
        head.save(out)
        head2 = HeadRegistry.get(head_name).load(out)
    preds_after = head2.predict(Xva)
    probs_after = head2.predict_proba(Xva)

    np.testing.assert_array_equal(preds_before, preds_after)
    # MLP save/load on different device may have minor fp differences; allow tolerance
    np.testing.assert_allclose(probs_before, probs_after, atol=1e-4)


def test_compute_class_weight_balanced():
    y = np.array([0] * 80 + [1] * 20)
    w = compute_class_weight(y)
    # balanced weights: total / (2 * count)
    assert abs(w[0] - 0.625) < 1e-6
    assert abs(w[1] - 2.5) < 1e-6


def test_compute_class_weight_single_class():
    y = np.array([1] * 50)
    w = compute_class_weight(y)
    assert w == {0: 1.0, 1: 1.0}


@pytest.mark.parametrize("head_name", ["xgb", "lgbm"])
def test_head_reproducibility_same_seed(head_name):
    X, y = _make_data(n=200, d=6, seed=1)
    h1 = get_head(head_name, seed=13)
    h1.fit(X, y)
    h2 = get_head(head_name, seed=13)
    h2.fit(X, y)
    # Same seed should produce identical argmax predictions. Raw probas can
    # have tiny floating-point drift when parallelism is enabled, so check
    # predictions and loose proba equivalence.
    np.testing.assert_array_equal(h1.predict(X), h2.predict(X))
    np.testing.assert_allclose(h1.predict_proba(X), h2.predict_proba(X), atol=1e-4)


def test_class_weight_affects_minority_recall():
    """With strong imbalance, class-weighted fit should improve recall on minority."""
    X, y = _make_data(n=1000, d=10, seed=3, imbalance=0.2)  # class 1 is minority
    # 80/20 stratified split to guarantee val has both classes
    c1 = np.where(y == 1)[0]; c0 = np.where(y == 0)[0]
    rng = np.random.default_rng(3); rng.shuffle(c1); rng.shuffle(c0)
    val_c1 = c1[:max(5, len(c1) // 5)]
    val_c0 = c0[:max(5, len(c0) // 5)]
    val_idx = np.concatenate([val_c0, val_c1])
    train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    cw = compute_class_weight(ytr)

    h_plain = get_head("logreg", seed=3)
    h_plain.fit(Xtr, ytr)
    h_weighted = get_head("logreg", seed=3)
    h_weighted.fit(Xtr, ytr, class_weight=cw)

    def _recall(h, Xv, yv):
        p = h.predict(Xv)
        mask = yv == 1
        if not mask.any():
            return 1.0
        return (p[mask] == 1).mean()

    r1 = _recall(h_plain, Xva, yva)
    r2 = _recall(h_weighted, Xva, yva)
    assert r2 >= r1 - 0.05, f"weighting hurt minority recall: plain={r1:.3f} weighted={r2:.3f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
