"""AST + cyclomatic feature extraction for code snippets, multi-language.

Uses tree-sitter per language (via common.parsers) instead of stdlib `ast`.
The 21-feature schema is unchanged from the Python-only version; per-language
visitors map each canonical feature to the right grammar's node-kinds.

CLI:
    python -m stacking.features.ast_features \\
        --in_splits data/processed \\
        --out_dir runs/heads/extraction
"""

from __future__ import annotations

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.parsers import (
    parse, walk,
    LOOP_NODE_KINDS, IF_NODE_KINDS, SWITCH_NODE_KINDS,
    DECISION_NODE_KINDS, FUNCTION_NODE_KINDS, BREAK_NODE_KINDS,
    JUMP_NODE_KINDS,
)


# Ordered feature schema. Stays stable across runs and across languages
# (same names, same dimensionality; per-language visitors compute each).
FEATURES: tuple[tuple[str, str], ...] = (
    ("no_of_ifs",               "count"),
    ("no_of_switches",          "count"),
    ("no_of_loop",              "count"),
    ("no_of_break",             "count"),
    ("nested_loop_depth",       "count"),
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


# ---------------------------------------------------------------------------
# Per-language regex/text-level hint sets for the "presence" features. We
# don't try to walk every grammar's import tree; we just regex the source for
# distinctive identifier names that imply use of priority queues / hash sets /
# hash maps / sorts.
# ---------------------------------------------------------------------------

_PQ_TOKENS: dict[str, tuple[str, ...]] = {
    "python":     ("heapq", "heappush", "heappop", "heapify", "PriorityQueue"),
    "java":       ("PriorityQueue", "Heap"),
    "cpp":        ("priority_queue", "make_heap", "push_heap", "pop_heap"),
    "c":          ("heap_push", "heap_pop", "heapify"),
    "csharp":     ("PriorityQueue", "Heap"),
    "go":         ("heap.Push", "heap.Pop", "heap.Init", "container/heap"),
    "javascript": ("PriorityQueue", "MinHeap", "MaxHeap", "Heap"),
    "typescript": ("PriorityQueue", "MinHeap", "MaxHeap", "Heap"),
    "php":        ("SplPriorityQueue", "SplHeap"),
    "ruby":       ("PriorityQueue", "Heap"),
    "rust":       ("BinaryHeap",),
    "swift":      ("Heap", "PriorityQueue"),
}

_HSET_TOKENS: dict[str, tuple[str, ...]] = {
    "python":     ("set(", "frozenset(", "{ "),  # set literals are ambiguous; light hint
    "java":       ("HashSet", "TreeSet", "LinkedHashSet"),
    "cpp":        ("unordered_set", "set<"),
    "c":          ("hashset", "set_t"),
    "csharp":     ("HashSet", "SortedSet"),
    "go":         ("map[", "set", "struct{}"),  # Go uses map[T]struct{} idiom
    "javascript": ("new Set(",),
    "typescript": ("new Set(",),
    "php":        ("array_unique", "SplObjectStorage"),
    "ruby":       ("Set.new", "require 'set'"),
    "rust":       ("HashSet", "BTreeSet"),
    "swift":      ("Set<", "Set("),
}

_HMAP_TOKENS: dict[str, tuple[str, ...]] = {
    "python":     ("dict(", "defaultdict", "OrderedDict", "Counter", "{}"),
    "java":       ("HashMap", "TreeMap", "LinkedHashMap"),
    "cpp":        ("unordered_map", "map<"),
    "c":          ("hashmap", "map_t"),
    "csharp":     ("Dictionary<", "Hashtable"),
    "go":         ("map[",),
    "javascript": ("new Map(", "Object.create", "{}"),
    "typescript": ("new Map(", "Record<"),
    "php":        ("array(",),
    "ruby":       ("{}", "Hash.new"),
    "rust":       ("HashMap", "BTreeMap"),
    "swift":      ("[", ":"),  # weak hint; Swift dict literals use [k: v]
}

_SORT_TOKENS: dict[str, tuple[str, ...]] = {
    "python":     ("sorted(", ".sort(", "sort("),
    "java":       ("Collections.sort", "Arrays.sort", ".sort("),
    "cpp":        ("std::sort", "sort("),
    "c":          ("qsort",),
    "csharp":     ("Array.Sort", "OrderBy", ".Sort("),
    "go":         ("sort.Sort", "sort.Slice", "sort.Ints", "sort.Strings"),
    "javascript": (".sort(",),
    "typescript": (".sort(",),
    "php":        ("sort(", "asort(", "ksort(", "usort("),
    "ruby":       (".sort", "sort_by"),
    "rust":       (".sort(", ".sort_by(", ".sort_unstable("),
    "swift":      (".sort(", ".sorted(", "sorted("),
}


def _has_token(code: str, tokens: tuple[str, ...]) -> bool:
    return any(t in code for t in tokens)


def _count_token(code: str, tokens: tuple[str, ...]) -> int:
    return sum(code.count(t) for t in tokens)


# ---------------------------------------------------------------------------
# Tree-sitter helpers
# ---------------------------------------------------------------------------

def _is_node_in(node, kinds: frozenset[str]) -> bool:
    return node.type in kinds


def _nested_depth(root, kinds: frozenset[str]) -> int:
    """Max stack depth of nodes in `kinds` along any root-to-leaf path."""
    if not kinds:
        return 0
    max_depth = 0

    def visit(node, depth: int) -> None:
        nonlocal max_depth
        new_depth = depth + 1 if node.type in kinds else depth
        if new_depth > max_depth:
            max_depth = new_depth
        for c in node.children:
            visit(c, new_depth)

    visit(root, 0)
    return max_depth


def _count_nested_cooccurrence(root, outer_kinds: frozenset[str],
                               inner_kinds: frozenset[str]) -> int:
    """Count occurrences of inner-kind nodes inside an outer-kind ancestor."""
    if not outer_kinds or not inner_kinds:
        return 0
    count = 0

    def visit(node, depth_outer: int) -> None:
        nonlocal count
        is_outer = node.type in outer_kinds
        is_inner = node.type in inner_kinds
        # Self-nesting (loop_in_loop, cond_in_cond): only count when an outer
        # ancestor already exists, i.e. depth_outer > 0.
        if is_inner and depth_outer > 0:
            count += 1
        new_depth = depth_outer + 1 if is_outer else depth_outer
        for c in node.children:
            visit(c, new_depth)

    visit(root, 0)
    return count


def _count_methods(root, kinds: frozenset[str]) -> int:
    return sum(1 for n in walk(root) if n.type in kinds)


def _count_statements(root) -> int:
    """Crude statement count: any node ending in `_statement` or named like a stmt."""
    n = 0
    for node in walk(root):
        t = node.type
        if t.endswith("_statement") or t in (
            "expression_statement", "assignment", "assignment_expression",
            "let_declaration", "var_declaration", "field_declaration",
        ):
            n += 1
    return n


def _detects_recursion(root, fn_kinds: frozenset[str], src: bytes) -> bool:
    """True if any function body references its own name."""
    if not fn_kinds:
        return False
    for node in walk(root):
        if node.type not in fn_kinds:
            continue
        # Find the identifier name within the function header
        fname: str | None = None
        for child in node.children:
            if child.type in ("identifier", "name", "field_identifier",
                              "property_identifier", "type_identifier"):
                fname = src[child.start_byte:child.end_byte].decode("utf-8", "replace")
                break
        if not fname or len(fname) < 2:
            continue
        body = src[node.start_byte:node.end_byte].decode("utf-8", "replace")
        # `fname` appears at least once for the declaration; recursion -> >= 2.
        if body.count(fname) >= 2:
            return True
    return False


def _count_variables(root, src: bytes) -> int:
    """Approximate count of distinct identifier names that appear as
    assignment targets. Uses a regex over identifier tokens — works across
    languages without per-grammar AST surgery."""
    import re
    names: set[str] = set()
    text = src.decode("utf-8", "replace")
    # Patterns approximate per common syntax: `let x =`, `var x =`, `int x =`,
    # `x =`, `x:`. We collect candidate identifiers preceding `=`.
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z_0-9]{0,30})\b\s*=(?!=)", text):
        names.add(m.group(1))
    # Reject obvious operator/keyword-like tokens; keep this list small.
    names -= {"let", "var", "const", "if", "while", "for", "fn", "func",
              "def", "return", "import", "from", "true", "false", "null",
              "None", "True", "False", "this", "self"}
    return len(names)


def _cyclomatic(root, decision_kinds: frozenset[str],
                fn_kinds: frozenset[str]) -> tuple[float, float, float]:
    """McCabe-style cyclomatic complexity, computed per-function and aggregated.

    Per-function CC = 1 + (decision-point nodes inside the function body).
    Returns (max, sum, mean). Returns (0, 0, 0) if no functions.
    """
    if not decision_kinds or not fn_kinds:
        return 0.0, 0.0, 0.0
    per_fn: list[int] = []

    def count_decisions(node) -> int:
        n = 0
        for c in walk(node):
            if c is node:
                continue
            if c.type in decision_kinds:
                n += 1
        return n

    for node in walk(root):
        if node.type not in fn_kinds:
            continue
        per_fn.append(1 + count_decisions(node))
    if not per_fn:
        return 0.0, 0.0, 0.0
    arr = np.asarray(per_fn, dtype=np.float64)
    return float(arr.max()), float(arr.sum()), float(arr.mean())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ASTFeatures:
    values: np.ndarray   # shape (N_FEATURES,), dtype float32

    @classmethod
    def zero(cls) -> "ASTFeatures":
        return cls(values=np.zeros(N_FEATURES, dtype=np.float32))

    def to_dict(self) -> dict[str, float]:
        return {name: float(self.values[i]) for i, name in enumerate(FEATURE_NAMES)}


def extract_features(code: str, language: str = "python") -> ASTFeatures:
    """Extract the fixed-schema AST feature vector from `code` in `language`.

    On parse failure: zero vector (consistent with old fallback semantics).
    """
    if not code or not code.strip():
        return ASTFeatures.zero()

    try:
        tree = parse(language, code)
    except Exception:
        return ASTFeatures.zero()
    root = tree.root_node
    src = code.encode("utf-8")

    loop_k = LOOP_NODE_KINDS.get(language, frozenset())
    if_k = IF_NODE_KINDS.get(language, frozenset())
    switch_k = SWITCH_NODE_KINDS.get(language, frozenset())
    dec_k = DECISION_NODE_KINDS.get(language, frozenset())
    fn_k = FUNCTION_NODE_KINDS.get(language, frozenset())
    brk_k = BREAK_NODE_KINDS.get(language, frozenset())
    jmp_k = JUMP_NODE_KINDS.get(language, frozenset())

    # Single-pass counts
    n_if = n_switch = n_loop = n_break = n_jump = 0
    for node in walk(root):
        t = node.type
        if t in if_k:
            n_if += 1
        if t in switch_k:
            n_switch += 1
        if t in loop_k:
            n_loop += 1
        if t in brk_k:
            n_break += 1
        if t in jmp_k:
            n_jump += 1

    n_methods = _count_methods(root, fn_k)
    n_statements = _count_statements(root)
    n_vars = _count_variables(root, src)
    depth = _nested_depth(root, loop_k)
    recursion = 1 if _detects_recursion(root, fn_k, src) else 0

    pq_present = 1 if _has_token(code, _PQ_TOKENS.get(language, ())) else 0
    hset_present = 1 if _has_token(code, _HSET_TOKENS.get(language, ())) else 0
    hmap_present = 1 if _has_token(code, _HMAP_TOKENS.get(language, ())) else 0
    n_sort = _count_token(code, _SORT_TOKENS.get(language, ()))

    # Cond-aware sets for nesting heuristics: union of if + switch
    cond_k = if_k | switch_k

    cond_in_loop = _count_nested_cooccurrence(root, loop_k, cond_k)
    loop_in_cond = _count_nested_cooccurrence(root, cond_k, loop_k)
    loop_in_loop = _count_nested_cooccurrence(root, loop_k, loop_k)
    cond_in_cond = _count_nested_cooccurrence(root, cond_k, cond_k)

    cc_max, cc_sum, cc_mean = _cyclomatic(root, dec_k, fn_k)

    vec = np.array([
        n_if,
        n_switch,
        n_loop,
        n_break,
        depth,
        n_methods,
        n_vars,
        n_statements,
        n_jump,
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


def diff_columns() -> list[str]:
    out: list[str] = []
    for name, _kind in FEATURES:
        out.extend([f"ast_a__{name}", f"ast_b__{name}", f"ast_diff__{name}",
                    f"ast_abs_diff__{name}"])
    return out


def extract_differenced(code_a: str, code_b: str, language: str = "python") -> np.ndarray:
    """Pair feature row: 4*N_FEATURES floats. Both sides must share `language`
    (within-language pairs only).
    """
    fa = extract_features(code_a, language).values
    fb = extract_features(code_b, language).values
    diff = fb - fa
    absd = np.abs(diff)
    out = np.empty(4 * N_FEATURES, dtype=np.float32)
    out[0::4] = fa
    out[1::4] = fb
    out[2::4] = diff
    out[3::4] = absd
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _extract_pointwise_for_split(split_path: Path) -> tuple[list[str], np.ndarray]:
    tbl = pq.read_table(split_path)
    codes = tbl.column("code").to_pylist()
    ids = tbl.column("id").to_pylist()
    if "language" in tbl.column_names:
        langs = tbl.column("language").to_pylist()
    else:
        langs = ["python"] * len(codes)
    rows: list[np.ndarray] = []
    for code, lang in tqdm(list(zip(codes, langs)), desc=f"ast:{split_path.stem}"):
        rows.append(extract_features(code, lang).values)
    mat = np.stack(rows, axis=0) if rows else np.zeros((0, N_FEATURES), dtype=np.float32)
    return ids, mat


def _extract_pairwise_for_split(split_path: Path) -> tuple[list[str], np.ndarray]:
    tbl = pq.read_table(split_path)
    codes_a = tbl.column("code_a").to_pylist()
    codes_b = tbl.column("code_b").to_pylist()
    pair_ids = tbl.column("pair_id").to_pylist()
    if "language" in tbl.column_names:
        langs = tbl.column("language").to_pylist()
    else:
        langs = ["python"] * len(codes_a)
    rows: list[np.ndarray] = []
    for a, b, lang in tqdm(list(zip(codes_a, codes_b, langs)),
                            desc=f"ast:{split_path.stem}"):
        rows.append(extract_differenced(a, b, lang))
    mat = np.stack(rows, axis=0) if rows else np.zeros((0, 4 * N_FEATURES), dtype=np.float32)
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
    ap.add_argument("--pairwise_only", action="store_true")
    ap.add_argument("--pointwise_only", action="store_true")
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

    (out_dir / "ast_schema.json").write_text(json.dumps({
        "n_features": N_FEATURES,
        "feature_names": list(FEATURE_NAMES),
        "kinds": {n: k for n, k in FEATURES},
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
