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
    FeatureMatrix,
    _carve_val_from_train,
    binary_labels,
    filter_b_ge_a,
    filter_to_language,
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


# -----------------------------------------------------------------------------
# v1-only invariants (after the v2 / variant axis was retired)
# -----------------------------------------------------------------------------

def test_feature_matrix_has_no_variant_field():
    """Sanity-check that FeatureMatrix dropped the v1/v2 axis cleanly."""
    fm = _mk_fm(["python"])
    assert not hasattr(fm, "variant")


def test_dataset_has_no_variant_descriptions():
    """The v2 variant table should be gone from the module."""
    from stacking import dataset as ds
    assert not hasattr(ds, "Variant")
    assert not hasattr(ds, "VARIANT_DESCRIPTIONS")


def test_join_point_logits_emits_44d_block():
    """v1 logit block: 11 A + 11 B + 11 diff + 11 abs_diff = 44 cols, all scaled."""
    import hashlib
    import pyarrow as pa

    from stacking.dataset import _join_point_logits

    # Two pair rows pointing at two distinct snippets
    code_a, code_b = "def a(): pass", "def b(): pass"
    sha_a = hashlib.sha256(code_a.encode("utf-8")).hexdigest()
    sha_b = hashlib.sha256(code_b.encode("utf-8")).hexdigest()
    pair_tbl = pa.table({
        "pair_id": ["p0", "p1"],
        "ternary": ["same", "A_faster"],
        "code_a": [code_a, code_a],
        "code_b": [code_b, code_b],
    })
    point_tbl = pa.table({
        "id": ["a-id", "b-id"],
        "code_sha256": [sha_a, sha_b],
    })

    import tempfile
    import pyarrow.parquet as pq
    with tempfile.TemporaryDirectory() as td:
        ppt = Path(td) / "point.parquet"
        pq.write_table(point_tbl, ppt)

        # Synthetic pointwise logits: zero except for one nonzero per row
        logit_cols = {f"point_logit_{k}": [float(k), float(k * 2)]
                      for k in range(11)}
        logit_tbl = pa.table({"id": ["a-id", "b-id"], **logit_cols})

        X, cols, scaled = _join_point_logits(pair_tbl, ppt, logit_tbl)

    assert X.shape == (2, 44)
    assert len(cols) == 44
    assert all(scaled), "all v1 logit cols are scaled"
    # Block ordering
    assert cols[:11] == [f"point_A_logit_{k}" for k in range(11)]
    assert cols[11:22] == [f"point_B_logit_{k}" for k in range(11)]
    assert cols[22:33] == [f"point_diff_logit_{k}" for k in range(11)]
    assert cols[33:44] == [f"point_abs_diff_logit_{k}" for k in range(11)]
    # diff = B - A; abs_diff = |B - A|
    np.testing.assert_allclose(X[:, 22:33], X[:, 11:22] - X[:, :11])
    np.testing.assert_allclose(X[:, 33:44], np.abs(X[:, 11:22] - X[:, :11]))


# -----------------------------------------------------------------------------
# Per-language slicing
# -----------------------------------------------------------------------------

def _mk_fm(languages: list[str], y: list[int] | None = None,
           seed: int = 0, d: int = 8) -> FeatureMatrix:
    rng = np.random.default_rng(seed)
    n = len(languages)
    X = rng.standard_normal((n, d)).astype(np.float32)
    if y is None:
        y = (rng.random(n) > 0.5).astype(np.int64)
    return FeatureMatrix(
        pair_ids=[f"p{i:04d}" for i in range(n)],
        X=X,
        y=np.asarray(y, dtype=np.int64),
        columns=[f"f{i}" for i in range(d)],
        scaled_mask=np.ones(d, dtype=bool),
        languages=list(languages),
    )


def test_filter_to_language_row_count_matches():
    fm = _mk_fm(["python"] * 5 + ["java"] * 3 + ["ruby"] * 2)
    py = filter_to_language(fm, "python")
    java = filter_to_language(fm, "java")
    ruby = filter_to_language(fm, "ruby")
    assert py.X.shape[0] == 5
    assert java.X.shape[0] == 3
    assert ruby.X.shape[0] == 2
    assert all(l == "python" for l in py.languages)
    assert all(l == "java" for l in java.languages)


def test_filter_to_language_preserves_schema():
    fm = _mk_fm(["python"] * 4 + ["java"] * 2)
    py = filter_to_language(fm, "python")
    assert py.columns == fm.columns
    np.testing.assert_array_equal(py.scaled_mask, fm.scaled_mask)


def test_filter_to_language_missing_returns_empty():
    fm = _mk_fm(["python"] * 3 + ["java"] * 2)
    rust = filter_to_language(fm, "rust")
    assert rust.X.shape[0] == 0
    assert rust.y.shape[0] == 0
    assert rust.languages == []
    # Schema stable even when empty
    assert rust.columns == fm.columns


def test_filter_to_language_raises_when_no_languages():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 3)).astype(np.float32)
    fm = FeatureMatrix(
        pair_ids=[f"p{i}" for i in range(4)],
        X=X, y=np.zeros(4, dtype=np.int64),
        columns=["a","b","c"], scaled_mask=np.ones(3, dtype=bool),
        languages=[],   # missing
    )
    with pytest.raises(ValueError, match="no languages"):
        filter_to_language(fm, "python")


def test_carve_val_deterministic_across_calls():
    fm = _mk_fm(["ruby"] * 100, seed=7)
    a_train, a_val = _carve_val_from_train(fm, fraction=0.10, seed=1234)
    b_train, b_val = _carve_val_from_train(fm, fraction=0.10, seed=1234)
    # Same partition under same seed
    assert a_train.pair_ids == b_train.pair_ids
    assert a_val.pair_ids == b_val.pair_ids


def test_carve_val_disjoint_with_train():
    fm = _mk_fm(["ruby"] * 50, seed=11)
    train_p, val_p = _carve_val_from_train(fm, fraction=0.20, seed=42)
    train_set = set(train_p.pair_ids)
    val_set = set(val_p.pair_ids)
    assert train_set.isdisjoint(val_set)
    # Together cover the whole input
    assert train_set | val_set == set(fm.pair_ids)


def test_carve_val_target_size():
    fm = _mk_fm(["ruby"] * 100, seed=2)
    _, val_p = _carve_val_from_train(fm, fraction=0.10, seed=42)
    assert val_p.X.shape[0] == 10  # 10% of 100


def test_carve_val_seed_changes_partition():
    fm = _mk_fm(["ruby"] * 100, seed=2)
    _, val_a = _carve_val_from_train(fm, fraction=0.10, seed=1)
    _, val_b = _carve_val_from_train(fm, fraction=0.10, seed=2)
    assert set(val_a.pair_ids) != set(val_b.pair_ids)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
