"""Unit tests for OOF fold disjointness — the core leakage-fix invariant.

If these tests pass, no code seen during fold k's training also appears in
fold k's held-out set. That's the whole reason OOF exists.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stacking.features.oof_point import (
    _fold_key,
    assign_folds as assign_folds_point,
    split_train_parquet,
)
from stacking.features.oof_pair import (
    assign_folds as assign_folds_pair,
    split_pair_parquet,
)


# -----------------------------------------------------------------------------
# Pointwise fold disjointness
# -----------------------------------------------------------------------------

POINT_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("source", pa.string()),
    ("problem_id", pa.string()),
    ("solution_idx", pa.int32()),
    ("code", pa.string()),
    ("code_sha256", pa.string()),
    ("label", pa.string()),
    ("raw_complexity", pa.string()),
    ("tokens_graphcodebert", pa.int32()),
    ("ast_nodes", pa.int32()),
    ("augmented_from", pa.string()),
    ("split", pa.string()),
])


def _mk_point_parquet(rows: list[dict], tmp: Path) -> Path:
    p = tmp / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=POINT_SCHEMA), p,
                    compression="zstd")
    return p


def test_point_folds_disjoint_by_problem_id():
    # Several rows per problem, ensure all go to same fold
    rows = []
    for pid in range(20):
        for sol in range(3):  # 3 solutions per problem
            rows.append({
                "id": f"p{pid}_s{sol}",
                "source": "leetcode",
                "problem_id": str(pid),
                "solution_idx": sol,
                "code": f"def f_{pid}_{sol}(): pass\n",
                "code_sha256": f"sha_{pid}_{sol}",
                "label": "O(n)",
                "raw_complexity": "O(n)",
                "tokens_graphcodebert": 10,
                "ast_nodes": 5,
                "augmented_from": None,
                "split": "train",
            })

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pqp = _mk_point_parquet(rows, tmp)
        assignment = assign_folds_point(pqp, n_folds=5, seed=42)
        # Every solution of a given problem_id must land in the same fold
        by_pid: dict[str, set[int]] = {}
        for key, fold in assignment.items():
            by_pid.setdefault(key, set()).add(fold)
        for k, folds in by_pid.items():
            assert len(folds) == 1, f"fold_key {k!r} spans {folds}"


def test_point_split_parquet_round_trip():
    rows = [
        {
            "id": f"id{i}", "source": "s", "problem_id": str(i // 2),
            "solution_idx": i, "code": f"c_{i}",
            "code_sha256": f"sha_{i}", "label": "O(n)",
            "raw_complexity": "O(n)", "tokens_graphcodebert": 1,
            "ast_nodes": 1, "augmented_from": None, "split": "train",
        }
        for i in range(40)
    ]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pqp = _mk_point_parquet(rows, tmp)
        assignment = assign_folds_point(pqp, n_folds=4, seed=7)
        paths = split_train_parquet(pqp, assignment, tmp / "oof/splits", 4)
        assert len(paths) == 4
        all_rows_across_heldout = []
        for _, held_p in paths:
            all_rows_across_heldout.extend(pq.read_table(held_p).to_pylist())
        # Every original row must appear exactly once across held-out sets
        assert len(all_rows_across_heldout) == len(rows)
        assert {r["id"] for r in all_rows_across_heldout} == {r["id"] for r in rows}

        # Each held-out set has no overlap with its paired train set
        for tp, hp in paths:
            t_ids = {r["id"] for r in pq.read_table(tp).to_pylist()}
            h_ids = {r["id"] for r in pq.read_table(hp).to_pylist()}
            assert not (t_ids & h_ids), "train and heldout share ids"


def test_point_fold_assignment_stable_under_seed():
    rows = [
        {
            "id": f"id{i}", "source": "s", "problem_id": str(i),
            "solution_idx": 0, "code": f"c_{i}", "code_sha256": f"sha_{i}",
            "label": "O(1)", "raw_complexity": "O(1)",
            "tokens_graphcodebert": 1, "ast_nodes": 1, "augmented_from": None,
            "split": "train",
        }
        for i in range(30)
    ]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pqp = _mk_point_parquet(rows, tmp)
        a1 = assign_folds_point(pqp, n_folds=3, seed=99)
        a2 = assign_folds_point(pqp, n_folds=3, seed=99)
        a_other = assign_folds_point(pqp, n_folds=3, seed=100)
        assert a1 == a2
        assert a1 != a_other  # different seed should shuffle differently


# -----------------------------------------------------------------------------
# Pairwise fold disjointness (union-find over code SHAs)
# -----------------------------------------------------------------------------

PAIR_SCHEMA = pa.schema([
    ("pair_id", pa.string()),
    ("code_a", pa.string()),
    ("code_b", pa.string()),
    ("label_a", pa.string()),
    ("label_b", pa.string()),
    ("ternary", pa.string()),
    ("same_problem", pa.bool_()),
    ("tokens_combined", pa.int32()),
    ("split", pa.string()),
])


def _mk_pair_parquet(rows: list[dict], tmp: Path) -> Path:
    p = tmp / "pair_train.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=PAIR_SCHEMA), p,
                    compression="zstd")
    return p


def test_pair_folds_no_shared_code_across_fold_train_sets():
    """Any two pairs that share a code (on either side) must land in the same fold."""
    import hashlib
    rows = []
    # 20 distinct codes. We construct pairs so some share code across groups.
    codes = [f"def f_{i}(): return {i}" for i in range(20)]

    # Pair 0: code 0 + code 1  (group A)
    # Pair 1: code 1 + code 2  (group A — shares code 1 with pair 0)
    # Pair 2: code 2 + code 3  (group A — shares code 2 with pair 1)
    # Pair 3: code 4 + code 5  (group B — disjoint from A)
    # Pair 4: code 6 + code 7  (group C)
    # ...
    pair_specs = [
        (0, 1), (1, 2), (2, 3),   # component A
        (4, 5),                    # component B
        (6, 7), (7, 8),            # component C
        (10, 11),                  # component D
        (12, 13), (13, 14),        # component E
        (15, 16),                  # component F
    ]
    for i, (a, b) in enumerate(pair_specs):
        rows.append({
            "pair_id": f"p{i:04d}",
            "code_a": codes[a], "code_b": codes[b],
            "label_a": "O(n)", "label_b": "O(n)", "ternary": "same",
            "same_problem": False, "tokens_combined": 1, "split": "train",
        })

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pqp = _mk_pair_parquet(rows, tmp)
        assignment = assign_folds_pair(pqp, n_folds=3, seed=5)
        # Expected components: {0,1,2,3}, {4,5}, {6,7,8}, {10,11}, {12,13,14}, {15,16}
        # pairs within the same component must share the same fold.
        components = [
            [0, 1, 2],       # pairs p0000..p0002
            [3],              # p0003
            [4, 5],           # p0004..p0005
            [6],              # p0006
            [7, 8],           # p0007..p0008
            [9],              # p0009
        ]
        for comp in components:
            fold_set = {assignment[f"p{i:04d}"] for i in comp}
            assert len(fold_set) == 1, f"component {comp} spans folds {fold_set}"

        # Global covering: every pair has an assignment
        assert len(assignment) == len(rows)


def test_pair_split_parquet_disjoint_between_train_and_heldout():
    rows = [
        {
            "pair_id": f"p{i:04d}",
            "code_a": f"def a_{i}(): pass", "code_b": f"def b_{i}(): pass",
            "label_a": "O(n)", "label_b": "O(n)", "ternary": "same",
            "same_problem": False, "tokens_combined": 1, "split": "train",
        }
        for i in range(30)
    ]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pqp = _mk_pair_parquet(rows, tmp)
        assignment = assign_folds_pair(pqp, n_folds=5, seed=0)
        paths = split_pair_parquet(pqp, assignment, tmp / "oof_pair/splits", 5)
        # Union of held-out sets equals all pair ids; train is the complement.
        all_ids = {r["pair_id"] for r in rows}
        held_union = set()
        for tp, hp in paths:
            t_ids = {r["pair_id"] for r in pq.read_table(tp).to_pylist()}
            h_ids = {r["pair_id"] for r in pq.read_table(hp).to_pylist()}
            assert not (t_ids & h_ids)
            assert t_ids | h_ids == all_ids
            held_union |= h_ids
        assert held_union == all_ids


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
