"""Smoke test for stacking.hp_search.

Runs Optuna with a tiny number of trials on synthetic data to ensure the
search spaces and bookkeeping (trials.jsonl, best_params.json,
seed_results.jsonl, summary.json) all wire up correctly.

Does NOT test best-HP quality — the number of trials is too small.
"""

from __future__ import annotations

# torch first to dodge the Windows DLL-order issue with xgb/lgbm
import torch  # noqa: F401

import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stacking.dataset import FeatureMatrix
from stacking import hp_search


def _mk_features(n=200, d=12, seed=0, n_train=120, n_val=40, n_test=40):
    """Build three FeatureMatrix objects with a simple linear separable signal."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    X = rng.standard_normal((n_train + n_val + n_test, d)).astype(np.float32)
    y = (X @ w > 0).astype(np.int64)
    cols = [f"f_{i}" for i in range(d)]
    mask = np.ones(d, dtype=bool)

    def _mk(lo, hi):
        idx = slice(lo, hi)
        return FeatureMatrix(
            pair_ids=[f"p{i:04d}" for i in range(hi - lo)],
            X=X[idx].copy(),
            y=y[idx].copy(),
            columns=cols,
            scaled_mask=mask,
            variant="v1",
        )

    train = _mk(0, n_train)
    val = _mk(n_train, n_train + n_val)
    test = _mk(n_train + n_val, n_train + n_val + n_test)
    return train, val, test


@pytest.mark.parametrize("head_name", ["xgb", "lgbm", "mlp", "stacked"])
def test_hp_search_smoke(head_name):
    """Run 3 Optuna trials per head, verify artifacts get written.

    Uses mkdtemp + manual cleanup because Optuna's SQLite storage leaves
    the connection open until GC, and Windows file locks make
    TemporaryDirectory's rmtree fail during teardown.
    """
    train, val, test = _mk_features(n_train=120, n_val=40, n_test=40)

    td = tempfile.mkdtemp()
    try:
        out_dir = Path(td) / f"{head_name}-v1"
        with patch("stacking.hp_search.ds.build_all_splits",
                    return_value=(train, val, test)):
            summary = hp_search.run_hp_search(
                head_name=head_name,
                variant="v1",
                trials=3,
                seeds=[42, 43],
                search_seed=42,
                in_splits=Path("."),
                extraction_dir=Path("."),
                out_dir=out_dir,
                class_weight_mode="auto",
            )

        assert (out_dir / "best_params.json").exists()
        assert (out_dir / "trials.jsonl").exists()
        assert (out_dir / "seed_results.jsonl").exists()
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "study.db").exists()

        lines = (out_dir / "trials.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert obj["status"] == "ok"
            assert "val_macro_f1" in obj

        seed_rows = (out_dir / "seed_results.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(seed_rows) == 2
        for r in seed_rows:
            obj = json.loads(r)
            assert obj.get("status") == "ok", f"seed run failed: {obj}"
            assert 0.0 <= obj["test_accuracy"] <= 1.0

        assert "test_acc_mean" in summary
        assert summary["n_seeds"] == 2
        assert summary["head"] == head_name
        assert summary["variant"] == "v1"
    finally:
        # Release Optuna's SQLAlchemy engine before removing the dir
        # (Windows holds a file lock on study.db otherwise).
        gc.collect()
        shutil.rmtree(td, ignore_errors=True)


def test_hp_search_aggregate_writes_summary_md():
    """Given two cells of summary.json, aggregate() writes HP_SUMMARY.md + CSV."""
    td = tempfile.mkdtemp()
    try:
        root = Path(td)
        for head, f1 in (("xgb", 0.91), ("mlp", 0.89)):
            cell = root / f"{head}-v1"
            cell.mkdir()
            (cell / "summary.json").write_text(json.dumps({
                "head": head, "variant": "v1",
                "best_val_macro_f1": f1,
                "test_acc_mean": f1 - 0.005,
                "test_acc_std": 0.003,
                "test_macro_f1_mean": f1,
                "test_macro_f1_std": 0.002,
                "n_seeds": 3,
                "best_params": {"max_depth": 5},
                "search_seconds": 120.0,
            }), encoding="utf-8")

        hp_search.aggregate(root)
        md = (root / "HP_SUMMARY.md").read_text(encoding="utf-8")
        assert "xgb" in md and "mlp" in md
        # xgb should appear before mlp in the ranking (higher f1)
        assert md.index("xgb") < md.index("mlp")
        # CSV exists
        assert (root / "HP_SUMMARY.csv").exists()
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_space_hp_from_best_roundtrip():
    """space_hp_from_best rebuilds a fit-ready HP dict from Optuna best_params."""
    # MLP: should inject fixed fields
    best = {
        "hidden_layers": 3, "hidden_dim": 256, "activation": "gelu",
        "dropout": 0.2, "layer_norm": True, "optimizer": "adamw",
        "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 128,
    }
    hp = hp_search.space_hp_from_best("mlp", best, seed=99)
    assert hp["seed"] == 99
    assert hp["epochs"] == 40
    assert hp["patience"] == 5
    assert hp["hidden_layers"] == 3

    # XGB: 1:1 pass-through + seed injection
    best = {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 800,
            "min_child_weight": 2.0, "subsample": 0.8, "colsample_bytree": 0.7,
            "gamma": 0.5, "reg_alpha": 0.1, "reg_lambda": 2.0}
    hp = hp_search.space_hp_from_best("xgb", best, seed=7)
    assert hp["seed"] == 7
    assert hp["max_depth"] == 8

    # Stacked: use_* flags expand to bases list
    best = {"use_xgb": True, "use_lgbm": True, "use_mlp": False,
            "use_logreg": False, "use_rf": False, "meta": "logreg"}
    hp = hp_search.space_hp_from_best("stacked", best, seed=3)
    assert hp["bases"] == ["xgb", "lgbm"]
    assert hp["meta"] == "logreg"

    # Stacked degenerate: only one base flagged => fallback expands to >=2
    best = {"use_xgb": False, "use_lgbm": False, "use_mlp": True,
            "use_logreg": False, "use_rf": False, "meta": "mlp"}
    hp = hp_search.space_hp_from_best("stacked", best, seed=3)
    assert len(hp["bases"]) >= 2
    assert "mlp" in hp["bases"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
