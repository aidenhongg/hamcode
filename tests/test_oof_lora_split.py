"""Smoke tests for stacking.features.oof_lora fold-splitting correctness.

These tests don't run real LoRA training — they only exercise the pure-data
logic (fold assignment, per-fold parquet writes, language filtering). The
actual GPU training path is covered by Runpod-side integration runs.

What we assert:
  T1. assign_folds_per_language is deterministic under (seed, n_folds, rows).
  T2. Two rows with the same problem_id always land in the same fold.
  T3. write_per_fold_parquets builds disjoint, exhaustive heldout splits
      across all K folds (every train row appears in exactly one heldout).
  T4. For each fold k, the per-fold train.parquet has exactly the rows whose
      fold_assignment != k.
  T5. filter_table_by_lang preserves schema and only emits rows for the
      requested language.
  T6. The CLI parses (--help works, required args declared correctly).

Run from codebert/:
    python tests/test_oof_lora_split.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
REPO = THIS.parents[1]
sys.path.insert(0, str(REPO))

import pyarrow as pa
import pyarrow.parquet as pq

from common.schemas import POINT_SCHEMA
from stacking.features.oof_lora import (
    _fold_key,
    assign_folds_per_language,
    filter_table_by_lang,
    write_per_fold_parquets,
)


def _make_rows(n_problems: int, language: str, n_solutions_per_problem: int = 2,
               start_idx: int = 0) -> list[dict]:
    """Synthesize valid PointRecord rows with stable problem_ids."""
    rows: list[dict] = []
    for p in range(n_problems):
        pid = f"{language}-prob-{p:04d}"
        for s in range(n_solutions_per_problem):
            i = start_idx + p * n_solutions_per_problem + s
            code = f"# {language} sample {p}-{s}\nfn solve_{p}_{s}() {{}}"
            rows.append({
                "id": f"{language}-{i:06d}",
                "source": "test",
                "language": language,
                "problem_id": pid,
                "solution_idx": s,
                "code": code,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "label": "O(n)",
                "raw_complexity": "O(n)",
                "tokens_bpe": 20,
                "ast_nodes": 10,
                "augmented_from": None,
                "split": "train",
            })
    return rows


# ---------------------------------------------------------------------------
# T1: determinism
# ---------------------------------------------------------------------------

def test_t1_determinism() -> None:
    rows = _make_rows(50, "python")
    a = assign_folds_per_language(rows, n_folds=5, seed=42)
    b = assign_folds_per_language(rows, n_folds=5, seed=42)
    assert a == b, "assignment changed with same seed"
    c = assign_folds_per_language(rows, n_folds=5, seed=43)
    # Different seed should typically produce a different assignment.
    assert a != c, "different seeds gave identical assignment"


# ---------------------------------------------------------------------------
# T2: same problem_id -> same fold
# ---------------------------------------------------------------------------

def test_t2_same_pid_same_fold() -> None:
    rows = _make_rows(30, "java", n_solutions_per_problem=3)
    fold_assign = assign_folds_per_language(rows, n_folds=4, seed=42)
    by_pid: dict[str, set[int]] = {}
    for r in rows:
        pid = _fold_key(r)
        # Each row's effective fold is fold_assign[pid].
        by_pid.setdefault(pid, set()).add(fold_assign[pid])
    for pid, folds in by_pid.items():
        assert len(folds) == 1, f"pid {pid!r} ended up in folds {folds}"


# ---------------------------------------------------------------------------
# T3: heldout disjoint + exhaustive across folds
# T4: per-fold train.parquet = complement of heldout
# ---------------------------------------------------------------------------

def test_t3_t4_split_exhaustive(tmp_path: Path | None = None) -> None:
    import tempfile
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp(prefix="oof-test-"))

    # Build a multi-language pool but only fold one language at a time.
    rows_lang = _make_rows(40, "cpp", n_solutions_per_problem=2)  # 80 rows
    val_table = pa.Table.from_pylist(_make_rows(5, "cpp", 1, start_idx=10000),
                                       schema=POINT_SCHEMA)
    test_table = pa.Table.from_pylist(_make_rows(5, "cpp", 1, start_idx=20000),
                                        schema=POINT_SCHEMA)

    fold_assignment = assign_folds_per_language(rows_lang, n_folds=5, seed=42)
    paths = write_per_fold_parquets(
        rows_lang, val_table, test_table,
        fold_assignment, n_folds=5, schema=POINT_SCHEMA,
        lang_root=tmp_path,
    )
    assert len(paths) == 5

    all_heldout_ids: set[str] = set()
    all_train_rows = {r["id"] for r in rows_lang}

    for k, (fold_dir, fold_data_dir, heldout_pq) in enumerate(paths):
        # Files exist
        assert heldout_pq.exists(), heldout_pq
        assert (fold_data_dir / "train.parquet").exists()
        assert (fold_data_dir / "val.parquet").exists()
        assert (fold_data_dir / "test.parquet").exists()

        held_tbl = pq.read_table(heldout_pq)
        train_tbl = pq.read_table(fold_data_dir / "train.parquet")

        held_ids = set(held_tbl.column("id").to_pylist())
        train_ids = set(train_tbl.column("id").to_pylist())

        # T3a: heldout for fold k = exactly rows with fold_assignment[pid]==k
        expected_held = {r["id"] for r in rows_lang
                         if fold_assignment[_fold_key(r)] == k}
        assert held_ids == expected_held, (
            f"fold {k} heldout mismatch: got {len(held_ids)} expected "
            f"{len(expected_held)} (sym diff: "
            f"{(held_ids ^ expected_held)})"
        )

        # T4: per-fold train.parquet = complement of heldout
        expected_train = all_train_rows - expected_held
        assert train_ids == expected_train, (
            f"fold {k} train mismatch: got {len(train_ids)} expected "
            f"{len(expected_train)}"
        )

        # T3b: held + train together cover all train rows, with no overlap
        assert held_ids.isdisjoint(train_ids), (
            f"fold {k}: heldout and train overlap on {held_ids & train_ids}"
        )
        assert held_ids | train_ids == all_train_rows

        # val and test parquet contents preserved
        val_tbl = pq.read_table(fold_data_dir / "val.parquet")
        assert val_tbl.num_rows == val_table.num_rows
        test_tbl = pq.read_table(fold_data_dir / "test.parquet")
        assert test_tbl.num_rows == test_table.num_rows

        all_heldout_ids |= held_ids

    # T3c: union of heldout across folds = full L-train (every row appears in
    # exactly one fold).
    assert all_heldout_ids == all_train_rows, (
        f"union of heldout != full train. missing={all_train_rows - all_heldout_ids}"
        f"  extra={all_heldout_ids - all_train_rows}"
    )


# ---------------------------------------------------------------------------
# T5: filter_table_by_lang
# ---------------------------------------------------------------------------

def test_t5_filter_table_by_lang() -> None:
    rows = _make_rows(10, "python") + _make_rows(10, "java", start_idx=1000) + \
           _make_rows(5, "rust", start_idx=2000)
    tbl = pa.Table.from_pylist(rows, schema=POINT_SCHEMA)
    py = filter_table_by_lang(tbl, "python")
    assert py.num_rows == 20  # 10 problems * 2 solutions
    assert all(l == "python" for l in py.column("language").to_pylist())
    rs = filter_table_by_lang(tbl, "rust")
    assert rs.num_rows == 10
    swift = filter_table_by_lang(tbl, "swift")
    assert swift.num_rows == 0


# ---------------------------------------------------------------------------
# T6: CLI --help responds
# ---------------------------------------------------------------------------

def test_t6_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "stacking.features.oof_lora", "--help"],
        cwd=str(REPO), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"--help failed: {proc.stderr[-400:]}"
    out = proc.stdout
    for needed in ("--base_run", "--full_lora_root", "--n_folds", "--resume",
                   "--languages"):
        assert needed in out, f"--help missing {needed}"


# ---------------------------------------------------------------------------
# Boundary: n_folds == 2 (smallest valid)
# ---------------------------------------------------------------------------

def test_t7_min_folds() -> None:
    rows = _make_rows(8, "go", n_solutions_per_problem=1)
    fa = assign_folds_per_language(rows, n_folds=2, seed=42)
    folds = set(fa.values())
    assert folds <= {0, 1}
    # All 2 folds should be populated for 8 problems.
    assert folds == {0, 1}


def test_t8_invalid_folds() -> None:
    rows = _make_rows(5, "ts")
    try:
        assign_folds_per_language(rows, n_folds=1, seed=42)
    except ValueError:
        return
    raise AssertionError("expected ValueError for n_folds<2")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        ("T1 determinism",                test_t1_determinism),
        ("T2 same pid same fold",         test_t2_same_pid_same_fold),
        ("T3+T4 split exhaustive",        test_t3_t4_split_exhaustive),
        ("T5 filter_table_by_lang",       test_t5_filter_table_by_lang),
        ("T6 CLI --help",                 test_t6_cli_help),
        ("T7 min folds=2",                test_t7_min_folds),
        ("T8 invalid folds<2 raises",     test_t8_invalid_folds),
    ]
    fails: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
            print(f"  OK  {name}")
        except AssertionError as e:
            fails.append((name, "AssertionError: " + (str(e) or "<no msg>")))
            print(f"  FAIL {name}: {e}")
        except Exception as e:
            fails.append((name, f"{type(e).__name__}: {str(e)[:300]}"))
            print(f"  ERR  {name}: {type(e).__name__}: {e}")
    if fails:
        print(f"\n{len(fails)} test(s) failed")
        return 1
    print(f"\nall {len(tests)} oof_lora smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
