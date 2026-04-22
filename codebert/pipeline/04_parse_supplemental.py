"""Generate supplemental synthetic samples for rare complexity classes.

Hand-crafted templates per class; each expanded via deterministic variable
renaming + harmless literal tweaks (loop-var swap, list vs tuple, rename
helper fn) to produce K variants. Tagged `source=synthetic` so we can
ablation-test the model.

Also accepts an optional CodeNet-style jsonl via --codenet_path; IBM CodeNet
access is gated, so we leave the path as a hook rather than auto-fetching.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)

# -----------------------------------------------------------------------------
# Templates. Dedent is handled; keep them parse-clean.
# -----------------------------------------------------------------------------

TEMPLATES: dict[str, list[str]] = {
    "exponential": [
        # subset-sum recursion (O(2^n))
        """\
def solve(arr, target):
    def rec(i, remain):
        if remain == 0:
            return True
        if i == len(arr) or remain < 0:
            return False
        return rec(i + 1, remain - arr[i]) or rec(i + 1, remain)
    return rec(0, target)
""",
        # n-queens (backtracking)
        """\
def n_queens(n):
    result = []
    def backtrack(row, cols, d1, d2, state):
        if row == n:
            result.append(state[:])
            return
        for col in range(n):
            if col in cols or (row - col) in d1 or (row + col) in d2:
                continue
            state.append(col)
            cols.add(col); d1.add(row - col); d2.add(row + col)
            backtrack(row + 1, cols, d1, d2, state)
            state.pop()
            cols.remove(col); d1.remove(row - col); d2.remove(row + col)
    backtrack(0, set(), set(), set(), [])
    return result
""",
        # permutations (O(n!))
        """\
def permute(nums):
    out = []
    def back(path, rem):
        if not rem:
            out.append(path[:])
            return
        for i, x in enumerate(rem):
            path.append(x)
            back(path, rem[:i] + rem[i+1:])
            path.pop()
    back([], list(nums))
    return out
""",
        # TSP brute-force
        """\
from itertools import permutations
def tsp(dist, n):
    best = float("inf")
    for perm in permutations(range(1, n)):
        c = dist[0][perm[0]]
        for i in range(len(perm) - 1):
            c += dist[perm[i]][perm[i+1]]
        c += dist[perm[-1]][0]
        if c < best:
            best = c
    return best
""",
        # power set (O(2^n))
        """\
def power_set(nums):
    out = [[]]
    for x in nums:
        out = out + [s + [x] for s in out]
    return out
""",
        # combinations recursion
        """\
def combinations(arr, k):
    res = []
    def back(start, path):
        if len(path) == k:
            res.append(path[:])
            return
        for i in range(start, len(arr)):
            path.append(arr[i])
            back(i + 1, path)
            path.pop()
    back(0, [])
    return res
""",
        # hamiltonian path check
        """\
def hamiltonian(adj, n):
    def back(v, visited):
        if len(visited) == n:
            return True
        for u in adj[v]:
            if u not in visited:
                visited.add(u)
                if back(u, visited):
                    return True
                visited.remove(u)
        return False
    return back(0, {0})
""",
    ],
    "O(n^3)": [
        # Floyd-Warshall
        """\
def floyd_warshall(dist, n):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist
""",
        # 3SUM naive
        """\
def three_sum(nums):
    n = len(nums)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if nums[i] + nums[j] + nums[k] == 0:
                    out.append((nums[i], nums[j], nums[k]))
    return out
""",
        # matrix multiplication
        """\
def matmul(A, B, n):
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = 0
            for k in range(n):
                acc += A[i][k] * B[k][j]
            C[i][j] = acc
    return C
""",
        # interval DP (matrix chain)
        """\
def matrix_chain(dims):
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float("inf")
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
    return dp[0][n-1]
""",
    ],
    "O(m+n)": [
        # merge two sorted arrays
        """\
def merge(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
""",
        # two-pointer symmetric difference
        """\
def sym_diff(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1; j += 1
        elif a[i] < b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:]); out.extend(b[j:])
    return out
""",
        # graph DFS on adjacency list (V + E)
        """\
def dfs_all(adj, n_nodes):
    visited = [False] * n_nodes
    order = []
    def go(u):
        visited[u] = True
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                go(v)
    for u in range(n_nodes):
        if not visited[u]:
            go(u)
    return order
""",
        # BFS shortest path (V + E)
        """\
from collections import deque
def bfs(adj, source, n):
    dist = [-1] * n
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist
""",
        # histogram union
        """\
def union_counts(a, b):
    ca = {}
    for x in a: ca[x] = ca.get(x, 0) + 1
    cb = {}
    for x in b: cb[x] = cb.get(x, 0) + 1
    out = {}
    for k, v in ca.items(): out[k] = v
    for k, v in cb.items(): out[k] = out.get(k, 0) + v
    return out
""",
    ],
    "O(m*n)": [
        # LCS DP
        """\
def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
""",
        # edit distance
        """\
def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
""",
        # grid BFS
        """\
from collections import deque
def grid_bfs(grid):
    m, n = len(grid), len(grid[0])
    INF = float("inf")
    dist = [[INF] * n for _ in range(m)]
    q = deque()
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dist[i][j] = 0
                q.append((i, j))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and dist[nr][nc] == INF:
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
    return dist
""",
        # knapsack 0/1
        """\
def knapsack(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                cand = dp[i-1][w - weights[i-1]] + values[i-1]
                if cand > dp[i][w]:
                    dp[i][w] = cand
    return dp[n][W]
""",
    ],
    "O(m log n)": [
        # m binary searches in a sorted array of length n
        """\
import bisect
def search_many(queries, sorted_arr):
    return [bisect.bisect_left(sorted_arr, q) for q in queries]
""",
        # event-driven heap
        """\
import heapq
def process(events, n_initial):
    h = list(range(n_initial))
    heapq.heapify(h)
    out = []
    for e in events:
        heapq.heappush(h, e)
        out.append(heapq.heappop(h))
    return out
""",
        # bulk insort
        """\
from bisect import insort
def insert_all(values, sorted_init):
    arr = list(sorted_init)
    for v in values:
        insort(arr, v)
    return arr
""",
        # m top-k queries over a sorted window
        """\
def top_k_per_step(events, window_sorted, k):
    out = []
    for e in events:
        window_sorted.append(e)
        window_sorted.sort()
        out.append(window_sorted[-k:])
    return out
""",
    ],
    "O((m+n) log(m+n))": [
        # sort a concatenation
        """\
def combined_sort(a, b):
    merged = list(a) + list(b)
    merged.sort()
    return merged
""",
        # sort an event stream
        """\
def schedule(a, b):
    events = []
    for t in a: events.append((t, "a"))
    for t in b: events.append((t, "b"))
    events.sort()
    return events
""",
        # mergesort the combined list
        """\
def merge_sort_combined(a, b):
    def msort(arr):
        if len(arr) <= 1: return arr
        mid = len(arr) // 2
        l = msort(arr[:mid])
        r = msort(arr[mid:])
        out = []
        i = j = 0
        while i < len(l) and j < len(r):
            if l[i] <= r[j]: out.append(l[i]); i += 1
            else:           out.append(r[j]); j += 1
        out.extend(l[i:]); out.extend(r[j:])
        return out
    return msort(list(a) + list(b))
""",
    ],
}

# Pairs for variable renaming augmentation. Keep semantics identical.
_RENAME_SCHEMES: tuple[dict[str, str], ...] = (
    {},
    {"i": "p", "j": "q"},
    {"x": "u", "y": "v"},
    {"out": "result"},
    {"arr": "nums", "a": "xs", "b": "ys"},
    {"dp": "table"},
    {"dist": "distances"},
    {"path": "trail", "back": "walk"},
    {"perm": "permutation"},
    {"solve": "calc"},
    {"rec": "helper"},
)


def _rename_identifiers(src: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return src
    result = src
    for old, new in mapping.items():
        result = re.sub(rf"\b{re.escape(old)}\b", new, result)
    return result


def _validate_python(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def expand_template(label: str, idx: int, src: str, max_variants: int) -> list[dict]:
    """Yield variants of a template — original + up to (max_variants-1) renames."""
    out: list[dict] = []
    for variant_i, scheme in enumerate(_RENAME_SCHEMES[:max_variants]):
        renamed = _rename_identifiers(src, scheme)
        if not _validate_python(renamed):
            continue
        pid = "syn-" + hashlib.sha1(renamed.encode("utf-8")).hexdigest()[:12]
        out.append({
            "source": "synthetic",
            "problem_id": pid,
            "solution_idx": variant_i,
            "code": renamed,
            "raw_complexity": label,       # already canonical
            "pre_label": label,
            "template": f"{label}#{idx}",
            "variant": variant_i,
        })
    return out


def read_codenet(path: Path | None) -> list[dict]:
    """Read an optional CodeNet-derived jsonl (schema: {code, label})."""
    if path is None or not path.exists() or not path.is_file():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = obj.get("code") or obj.get("src")
            label = obj.get("label") or obj.get("complexity")
            if not code or not label:
                continue
            pid = "cn-" + hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]
            out.append({
                "source": "codenet",
                "problem_id": pid,
                "solution_idx": 0,
                "code": code,
                "raw_complexity": str(label),
                "pre_label": str(label),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/interim/parsed/supplemental.jsonl")
    ap.add_argument("--max_variants_per_template", type=int, default=8)
    ap.add_argument("--codenet_path", default="", help="optional path to CodeNet jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_emit = 0
    per_class: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as fout:
        for label, templates in TEMPLATES.items():
            for idx, src in enumerate(templates):
                for rec in expand_template(label, idx, src, args.max_variants_per_template):
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    per_class[label] = per_class.get(label, 0) + 1
                    n_emit += 1

        for rec in read_codenet(Path(args.codenet_path) if args.codenet_path else None):
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            per_class[rec["pre_label"]] = per_class.get(rec["pre_label"], 0) + 1
            n_emit += 1

    print(f"[04] synthetic+codenet records: {n_emit}", flush=True)
    for cls in sorted(per_class):
        print(f"[04]   {cls:<24} {per_class[cls]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
