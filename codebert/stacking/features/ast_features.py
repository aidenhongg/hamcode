"""AST + cyclomatic feature extraction for Python code snippets.

Uses the stdlib `ast` module (reliable, deterministic) plus `radon` for
McCabe cyclomatic complexity. The fallback on SyntaxError mirrors the
existing `data.py:extract_dataflow` contract: return zero vector.

The per-snippet feature vector has a fixed schema (see FEATURES below).
`extract_differenced(code_a, code_b)` returns a pair-level vector combining
raw A, raw B, signed diff, and |diff| for each feature — giving 4x the
per-snippet dimensionality (minus the redundant |diff| for strict booleans,
which we still emit for schema stability).

CLI:
    python -m stacking.features.ast_features \
        --in_splits data/processed \
        --out_dir runs/heads/extraction
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=SyntaxWarning)

try:
    from radon.complexity import cc_visit
except ImportError as e:  # pragma: no cover
    raise RuntimeError("radon is required: pip install radon") from e


# Ordered feature schema. Must stay stable across runs (downstream scaler
# and head expect a fixed column order).
#
# Categories:
#   count:   non-negative integer — will be log1p'd + z-scored downstream
#   bool:    0/1 — kept raw, differencing yields {-1, 0, 1}
#   cont:    continuous float — z-scored downstream
FEATURES: tuple[tuple[str, str], ...] = (
    ("no_of_ifs",               "count"),
    ("no_of_switches",          "count"),   # Python 3.10 `match` counted here
    ("no_of_loop",              "count"),
    ("no_of_break",             "count"),
    ("nested_loop_depth",       "count"),   # strongest signal (user notes 62% alone)
    ("noOfMethods",             "count"),
    ("noOfVariables",           "count"),
    ("noOfStatements",          "count"),
    ("noOfJumps",               "count"),
    ("recursion_present",       "bool"),
    ("priority_queue_present",  "bool"),
    ("hash_set_present",        "bool"),
    ("hash_map_present",        "bool"),
    ("no_of_sort",              "count"),
    ("cond_in_loop_freq",       "count"),
    ("loop_in_cond_freq",       "count"),
    ("loop_in_loop_freq",       "count"),
    ("cond_in_cond_freq",       "count"),
    ("cyclomatic_max",          "cont"),
    ("cyclomatic_sum",          "cont"),
    ("cyclomatic_mean",         "cont"),
)
FEATURE_NAMES: tuple[str, ...] = tuple(name for name, _ in FEATURES)
FEATURE_KIND: dict[str, str] = dict(FEATURES)
N_FEATURES = len(FEATURES)

# Identifiers that imply specific data structures / ops when imported/called.
_PQ_HINTS = {"heapq", "heappush", "heappop", "heapify", "heappushpop", "PriorityQueue"}
_HSET_CALLS = {"set", "frozenset"}
_HMAP_CALLS = {"dict", "defaultdict", "OrderedDict", "Counter", "ChainMap"}
_SORT_CALLS = {"sorted", "sort"}  # .sort method on lists also counted


def _is_loop(node: ast.AST) -> bool:
    return isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.comprehension))


def _is_cond(node: ast.AST) -> bool:
    # `match` goes here too — we count it under no_of_switches *and* treat it
    # structurally like a multi-way branch for the nesting heuristics.
    if isinstance(node, ast.If):
        return True
    if hasattr(ast, "Match") and isinstance(node, ast.Match):
        return True
    return False


def _nested_loop_depth(root: ast.AST) -> int:
    """Max depth of nested loops in the AST."""
    max_depth = 0

    def walk(node: ast.AST, depth: int) -> None:
        nonlocal max_depth
        new_depth = depth + 1 if _is_loop(node) else depth
        max_depth = max(max_depth, new_depth)
        for child in ast.iter_child_nodes(node):
            walk(child, new_depth)

    walk(root, 0)
    return max_depth


def _count_nested_cooccurrence(root: ast.AST, outer_pred, inner_pred) -> int:
    """Count occurrences of `inner_pred` nodes inside `outer_pred` ancestors."""
    count = 0

    def walk(node: ast.AST, inside_outer: bool) -> None:
        nonlocal count
        is_outer = outer_pred(node)
        is_inner = inner_pred(node)
        # Count this node if it's an inner and an outer ancestor exists.
        # For outer==inner (e.g., loop_in_loop), only count when INSIDE an existing outer.
        if is_inner and inside_outer and not (outer_pred is inner_pred and node is _ROOT):
            count += 1
        new_inside = inside_outer or is_outer
        for child in ast.iter_child_nodes(node):
            walk(child, new_inside)

    walk(root, False)
    return count


_ROOT = object()  # sentinel — never actually reached


def _extract_identifier(node: ast.AST) -> str | None:
    """Pull a readable identifier from Attribute/Name for call-target analysis."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _collect_call_names(root: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(root):
        if isinstance(node, ast.Call):
            nm = _extract_identifier(node.func)
            if nm:
                names.append(nm)
    return names


def _collect_import_names(root: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(root):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
                if alias.asname:
                    out.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
            for alias in node.names:
                out.add(alias.name)
                if alias.asname:
                    out.add(alias.asname)
    return out


def _detects_recursion(root: ast.AST) -> bool:
    """True if any function body calls a function with its own name.

    Catches direct recursion. Does not catch mutual recursion (e.g., a calls b
    which calls a) — that's a known limitation; in practice direct is >95%.
    """
    for node in ast.walk(root):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_name = node.name
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    nm = _extract_identifier(sub.func)
                    if nm == fn_name:
                        return True
    return False


def _count_variables(root: ast.AST) -> int:
    """Unique assignment targets (Name ids only — skips destructured tuples, etc.)."""
    names: set[str] = set()
    for node in ast.walk(root):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
                elif isinstance(tgt, (ast.Tuple, ast.List)):
                    for elt in tgt.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            tgt = node.target
            if isinstance(tgt, ast.Name):
                names.add(tgt.id)
    return len(names)


def _count_switch_like(root: ast.AST) -> int:
    """Python 3.10+ `match` statements. Always 0 on <3.10."""
    if not hasattr(ast, "Match"):
        return 0
    return sum(1 for n in ast.walk(root) if isinstance(n, ast.Match))


def _cyclomatic(code: str) -> tuple[float, float, float]:
    """Return (max, sum, mean) of McCabe cyclomatic complexity across fns.

    Returns (0.0, 0.0, 0.0) if no functions.
    """
    try:
        blocks = cc_visit(code)
    except (SyntaxError, ValueError):
        return 0.0, 0.0, 0.0
    vals = [float(b.complexity) for b in blocks]
    if not vals:
        return 0.0, 0.0, 0.0
    return max(vals), sum(vals), float(np.mean(vals))


@dataclass
class ASTFeatures:
    values: np.ndarray   # shape (N_FEATURES,), dtype float32

    @classmethod
    def zero(cls) -> "ASTFeatures":
        return cls(values=np.zeros(N_FEATURES, dtype=np.float32))

    def to_dict(self) -> dict[str, float]:
        return {name: float(self.values[i]) for i, name in enumerate(FEATURE_NAMES)}


def extract_features(code: str) -> ASTFeatures:
    """Extract the fixed-schema AST feature vector from Python source.

    On SyntaxError: return zeros (same fallback semantics as data.py).
    """
    if not code or not code.strip():
        return ASTFeatures.zero()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ASTFeatures.zero()

    # Structural counts via single walk
    n_if = 0
    n_loop = 0
    n_break = 0
    n_methods = 0
    n_statements = 0
    n_jumps = 0

    for node in ast.walk(tree):
        if _is_cond(node):
            n_if += 1
        if _is_loop(node):
            n_loop += 1
        if isinstance(node, ast.Break):
            n_break += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n_methods += 1
        if isinstance(node, ast.stmt):
            n_statements += 1
        if isinstance(node, (ast.Break, ast.Continue, ast.Return, ast.Raise)):
            n_jumps += 1

    n_switches = _count_switch_like(tree)
    n_vars = _count_variables(tree)
    depth = _nested_loop_depth(tree)
    recursion = 1 if _detects_recursion(tree) else 0

    # Presence indicators from imports + call names
    imports = _collect_import_names(tree)
    calls = _collect_call_names(tree)
    call_set = set(calls)

    pq_present = 1 if (imports & _PQ_HINTS) or (call_set & _PQ_HINTS) else 0
    # Also detect dict/set literals directly.
    hset_present = 0
    hmap_present = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Set) or isinstance(node, ast.SetComp):
            hset_present = 1
        if isinstance(node, ast.Dict) or isinstance(node, ast.DictComp):
            hmap_present = 1
    if (call_set & _HSET_CALLS) or ("set" in imports):
        hset_present = 1
    if (call_set & _HMAP_CALLS):
        hmap_present = 1

    # Sort: count both sorted() calls AND .sort() method calls.
    n_sort = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            nm = _extract_identifier(node.func)
            if nm in _SORT_CALLS:
                n_sort += 1
            # .sort() method
            if isinstance(node.func, ast.Attribute) and node.func.attr == "sort":
                n_sort += 1

    cond_in_loop = _count_nested_cooccurrence(tree, _is_loop, _is_cond)
    loop_in_cond = _count_nested_cooccurrence(tree, _is_cond, _is_loop)
    loop_in_loop = _count_nested_cooccurrence(tree, _is_loop, _is_loop)
    cond_in_cond = _count_nested_cooccurrence(tree, _is_cond, _is_cond)

    cc_max, cc_sum, cc_mean = _cyclomatic(code)

    vec = np.array([
        n_if,
        n_switches,
        n_loop,
        n_break,
        depth,
        n_methods,
        n_vars,
        n_statements,
        n_jumps,
        recursion,
        pq_present,
        hset_present,
        hmap_present,
        n_sort,
        cond_in_loop,
        loop_in_cond,
        loop_in_loop,
        cond_in_cond,
        cc_max,
        cc_sum,
        cc_mean,
    ], dtype=np.float32)
    return ASTFeatures(values=vec)


# -----------------------------------------------------------------------------
# Pair differencing: produce raw A, raw B, B-A, |B-A| for each feature.
# -----------------------------------------------------------------------------

def diff_columns() -> list[str]:
    """Column names for the differenced pair feature matrix."""
    out: list[str] = []
    for name, kind in FEATURES:
        out.extend([f"ast_a__{name}", f"ast_b__{name}", f"ast_diff__{name}",
                    f"ast_abs_diff__{name}"])
    return out


def extract_differenced(code_a: str, code_b: str) -> np.ndarray:
    """Return a pair feature row: 4*N_FEATURES floats."""
    fa = extract_features(code_a).values
    fb = extract_features(code_b).values
    diff = fb - fa
    absd = np.abs(diff)
    # Interleave: [a0, b0, d0, |d0|, a1, b1, d1, |d1|, ...]
    out = np.empty(4 * N_FEATURES, dtype=np.float32)
    out[0::4] = fa
    out[1::4] = fb
    out[2::4] = diff
    out[3::4] = absd
    return out


# -----------------------------------------------------------------------------
# CLI: extract for all splits
# -----------------------------------------------------------------------------

def _extract_pointwise_for_split(split_path: Path) -> tuple[list[str], np.ndarray]:
    """Extract per-snippet features for a pointwise parquet."""
    tbl = pq.read_table(split_path)
    codes: list[str] = tbl.column("code").to_pylist()
    ids: list[str] = tbl.column("id").to_pylist()
    mat = np.stack(
        [extract_features(c).values for c in tqdm(codes, desc=f"ast:{split_path.stem}")],
        axis=0,
    )
    return ids, mat


def _extract_pairwise_for_split(split_path: Path) -> tuple[list[str], np.ndarray]:
    """Extract differenced pair features for a pairwise parquet."""
    tbl = pq.read_table(split_path)
    codes_a: list[str] = tbl.column("code_a").to_pylist()
    codes_b: list[str] = tbl.column("code_b").to_pylist()
    pair_ids: list[str] = tbl.column("pair_id").to_pylist()
    mat = np.stack(
        [extract_differenced(a, b) for a, b in tqdm(
            list(zip(codes_a, codes_b)), desc=f"ast:{split_path.stem}")],
        axis=0,
    )
    return pair_ids, mat


def _write_pointwise(ids: list[str], mat: np.ndarray, out: Path) -> None:
    cols = {"id": ids}
    for i, name in enumerate(FEATURE_NAMES):
        cols[name] = mat[:, i].astype(np.float32)
    pq.write_table(pa.table(cols), out, compression="zstd")


def _write_pairwise(pair_ids: list[str], mat: np.ndarray, out: Path) -> None:
    cols = {"pair_id": pair_ids}
    names = diff_columns()
    for i, name in enumerate(names):
        cols[name] = mat[:, i].astype(np.float32)
    pq.write_table(pa.table(cols), out, compression="zstd")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_splits", default="data/processed",
                    help="directory with {pair_,}{train,val,test}.parquet")
    ap.add_argument("--out_dir", default="runs/heads/extraction",
                    help="where to write ast_{pair,point}_{split}.parquet")
    ap.add_argument("--pairwise_only", action="store_true",
                    help="skip pointwise (only pair features)")
    ap.add_argument("--pointwise_only", action="store_true",
                    help="skip pairwise (only per-snippet features)")
    args = ap.parse_args()

    in_dir = Path(args.in_splits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.pairwise_only:
        for sp in ("train", "val", "test"):
            src = in_dir / f"{sp}.parquet"
            if not src.exists():
                print(f"[ast] skip missing {src}", flush=True)
                continue
            ids, mat = _extract_pointwise_for_split(src)
            dst = out_dir / f"ast_point_{sp}.parquet"
            _write_pointwise(ids, mat, dst)
            print(f"[ast] wrote {dst}: {mat.shape}", flush=True)

    if not args.pointwise_only:
        for sp in ("train", "val", "test"):
            src = in_dir / f"pair_{sp}.parquet"
            if not src.exists():
                print(f"[ast] skip missing {src}", flush=True)
                continue
            pair_ids, mat = _extract_pairwise_for_split(src)
            dst = out_dir / f"ast_pair_{sp}.parquet"
            _write_pairwise(pair_ids, mat, dst)
            print(f"[ast] wrote {dst}: {mat.shape}", flush=True)

    # Also write a schema file so downstream code can re-derive column order.
    (out_dir / "ast_schema.json").write_text(json.dumps({
        "feature_names": list(FEATURE_NAMES),
        "feature_kind": FEATURE_KIND,
        "n_features_per_snippet": N_FEATURES,
        "pair_columns": diff_columns(),
    }, indent=2), encoding="utf-8")
    print(f"[ast] wrote {out_dir / 'ast_schema.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
