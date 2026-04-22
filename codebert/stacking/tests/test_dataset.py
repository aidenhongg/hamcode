"""Tests for dataset.py: filter, label construction, schema stability,
scaler discipline."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stacking.dataset import (
    binary_labels,
    filter_b_ge_a,
    log_imbalance,
)


def _mk_table(ternaries: list[str]) -> pa.Table:
    return pa.table({
        "pair_id": [f"p{i:04d}" for i in range(len(ternaries))],
        "ternary": ternaries,
        "code_a": ["def a():\n pass"] * len(ternaries),
        "code_b": ["def b():\n pass"] * len(ternaries),
    })


def test_filter_drops_b_faster():
    tbl = _mk_table(["A_faster", "same", "B_faster", "A_faster", "B_faster"])
    kept = filter_b_ge_a(tbl)
    assert kept.num_rows == 3
    remaining = set(kept.column("ternary").to_pylist())
    assert "B_faster" not in remaining


def test_filter_preserves_order_within_kept():
    tbl = _mk_table(["A_faster", "B_faster", "same", "B_faster", "A_faster"])
    kept = filter_b_ge_a(tbl)
    kept_pids = kept.column("pair_id").to_pylist()
    # p0000, p0002, p0004 survive (indices 0, 2, 4)
    assert kept_pids == ["p0000", "p0002", "p0004"]


def test_binary_labels_same_is_zero():
    tbl = _mk_table(["A_faster", "same", "A_faster"])
    y = binary_labels(tbl)
    np.testing.assert_array_equal(y, [1, 0, 1])


def test_binary_labels_only_a_faster():
    tbl = _mk_table(["A_faster"] * 5)
    y = binary_labels(tbl)
    np.testing.assert_array_equal(y, [1] * 5)


def test_log_imbalance_warns_on_skew(capsys):
    y = np.array([1] * 900 + [0] * 100, dtype=np.int64)
    with pytest.warns(UserWarning, match="imbalance ratio"):
        log_imbalance(y, "train")


def test_log_imbalance_no_warning_on_balanced():
    y = np.array([0] * 100 + [1] * 100, dtype=np.int64)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Should not raise
        log_imbalance(y, "train")


def test_filter_then_label_roundtrip():
    tbl = _mk_table(["A_faster", "same", "B_faster", "same", "A_faster"])
    kept = filter_b_ge_a(tbl)
    y = binary_labels(kept)
    # After filter, kept has [A_faster, same, same, A_faster] -> [1,0,0,1]
    np.testing.assert_array_equal(y, [1, 0, 0, 1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
