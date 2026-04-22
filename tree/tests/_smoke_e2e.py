"""End-to-end smoke test: synthetic data -> extract features -> train -> predict.

Not a pytest test; run manually:
    PYTHONPATH=. python tests/_smoke_e2e.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

from complexity.schemas import POINT_SCHEMA, PointRecord
from complexity.ingest.utils import write_points, sha256_str


# Small curated set of Python snippets with known complexity classes.
SNIPPETS: list[tuple[str, str]] = [
    # O(1)
    ("def add(a, b): return a + b", "O(1)"),
    ("def first(xs): return xs[0]", "O(1)"),
    ("def magic(): return 42", "O(1)"),
    ("def max2(a, b): return a if a > b else b", "O(1)"),

    # O(log n)
    ("""
def bsearch(xs, t):
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        m = (lo + hi) // 2
        if xs[m] == t: return m
        if xs[m] < t: lo = m + 1
        else: hi = m - 1
    return -1
""", "O(log n)"),
    ("""
def count_bits(n):
    c = 0
    while n > 0:
        c += n & 1
        n >>= 1
    return c
""", "O(log n)"),

    # O(n)
    ("""
def sum_list(xs):
    s = 0
    for x in xs:
        s += x
    return s
""", "O(n)"),
    ("""
def count(xs, t):
    c = 0
    for x in xs:
        if x == t:
            c += 1
    return c
""", "O(n)"),
    ("""
def reverse(xs):
    out = []
    for x in xs:
        out.append(x)
    return out[::-1]
""", "O(n)"),

    # O(n log n)
    ("""
def sort_wrap(xs):
    return sorted(xs)
""", "O(n log n)"),
    ("""
def merge_sort(xs):
    if len(xs) <= 1: return xs
    m = len(xs) // 2
    a = merge_sort(xs[:m])
    b = merge_sort(xs[m:])
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    return out + a[i:] + b[j:]
""", "O(n log n)"),

    # O(n^2)
    ("""
def bubble(xs):
    n = len(xs)
    for i in range(n):
        for j in range(n - i - 1):
            if xs[j] > xs[j+1]:
                xs[j], xs[j+1] = xs[j+1], xs[j]
    return xs
""", "O(n^2)"),
    ("""
def pairs(xs):
    result = []
    for i in range(len(xs)):
        for j in range(i+1, len(xs)):
            result.append((xs[i], xs[j]))
    return result
""", "O(n^2)"),

    # O(n^3)
    ("""
def triples(xs):
    r = []
    n = len(xs)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                r.append((xs[i], xs[j], xs[k]))
    return r
""", "O(n^3)"),

    # exponential
    ("""
def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)
""", "exponential"),
    ("""
def subsets(xs):
    if not xs: return [[]]
    rest = subsets(xs[1:])
    return rest + [[xs[0]] + s for s in rest]
""", "exponential"),

    # O(m+n)
    ("""
def merge(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:]); out.extend(b[j:])
    return out
""", "O(m+n)"),
    ("""
def common(s, t):
    seen = set(s)
    out = []
    for ch in t:
        if ch in seen:
            out.append(ch)
    return out
""", "O(m+n)"),

    # O(m*n)
    ("""
def sum_matrix(matrix):
    total = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            total += matrix[i][j]
    return total
""", "O(m*n)"),
    ("""
def edit_distance(s, t):
    m, n = len(s), len(t)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
""", "O(m*n)"),

    # O(m log n)
    ("""
import heapq
def k_smallest_per_row(matrix, k):
    h = []
    for row in matrix:
        for v in row:
            heapq.heappush(h, v)
    return [heapq.heappop(h) for _ in range(k)]
""", "O(m log n)"),

    # O((m+n) log(m+n))
    ("""
def sort_combined(a, b):
    return sorted(list(a) + list(b))
""", "O((m+n) log(m+n))"),
]


def main():
    # Duplicate each snippet 4 times (with different "problem_ids") for enough
    # per-class data to train a tree on.
    records = []
    for dup in range(4):
        for i, (code, label) in enumerate(SNIPPETS):
            code_v = code if dup == 0 else code + f"\n# variant {dup}\n"
            sha = sha256_str(code_v)
            records.append(PointRecord(
                id=f"synth::{i:02d}::{dup}::{sha[:8]}",
                source="synthetic",
                problem_id=f"synth_p_{i:02d}_{dup}",
                solution_idx=dup,
                code=code_v,
                code_sha256=sha,
                label=label,
                raw_complexity=label,
                origin="dataset",
                ast_nodes=0,
            ))
    print(f"[smoke] synthesized {len(records)} records")

    out = Path("data/interim/all.parquet")
    write_points(records, out)
    print(f"[smoke] wrote {out}")

    # Run feature extraction
    rc = subprocess.call([sys.executable, "scripts/02_extract_features.py"])
    if rc != 0:
        sys.exit(f"02_extract_features failed with code {rc}")

    # Show class distribution
    df = pd.read_parquet("data/features/all.parquet")
    print("\n[smoke] class distribution in feature table:")
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0))

    # Train
    rc = subprocess.call([sys.executable, "scripts/03_train.py"])
    if rc != 0:
        sys.exit(f"03_train failed with code {rc}")

    # Predict on a fresh snippet
    test_code = """
def foo(a, b):
    for i in range(len(a)):
        for j in range(len(b)):
            pass
"""
    from complexity.predict import ComplexityPredictor
    predictor = ComplexityPredictor(Path("models/point_rf.joblib"))
    r = predictor.predict_one(test_code)
    print(f"\n[smoke] prediction for nested O(m*n) snippet:")
    print(f"  label = {r['label']}")
    print(f"  conf  = {r['confidence']:.3f}")
    top = sorted(r["probabilities"].items(), key=lambda kv: -kv[1])[:5]
    for lab, p in top:
        print(f"  {lab:22s} {p:.3f}")


if __name__ == "__main__":
    main()
