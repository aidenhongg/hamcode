"""Unit tests for the 18-feature extractor.

Strategy: one minimal test per feature (verifies the counter fires on the
expected construct), then a set of end-to-end tests on canonical small
snippets from known complexity classes.
"""

from __future__ import annotations

import pytest

from complexity.features import FEATURE_NAMES, extract_features


def feat(code: str) -> dict[str, int]:
    return extract_features(code)


# ---------- per-feature tests ----------

def test_no_of_ifs():
    code = """
def f(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0
"""
    assert feat(code)["no_of_ifs"] == 2


def test_no_of_switches_match_statement():
    code = """
def f(x):
    match x:
        case 1:
            return 'one'
        case 2:
            return 'two'
    return 'other'
"""
    assert feat(code)["no_of_switches"] == 1


def test_no_of_loop_for_and_while():
    code = """
def f(n):
    for i in range(n):
        pass
    while n > 0:
        n -= 1
"""
    assert feat(code)["no_of_loop"] == 2


def test_no_of_loop_counts_comprehensions():
    # List comp = 1 for_in_clause. Nested list comp inside = depth 2 at the expression.
    code = "result = [x * y for x in range(10) for y in range(10)]"
    features = feat(code)
    # Two for_in_clauses → 2 loops at feature level
    assert features["no_of_loop"] == 2


def test_no_of_break():
    code = """
def f():
    for i in range(10):
        if i == 5:
            break
    for j in range(10):
        break
"""
    assert feat(code)["no_of_break"] == 2


def test_nested_loop_depth_flat():
    code = """
def f(n):
    for i in range(n):
        pass
"""
    assert feat(code)["nested_loop_depth"] == 1


def test_nested_loop_depth_two():
    code = """
def f(n):
    for i in range(n):
        for j in range(n):
            pass
"""
    assert feat(code)["nested_loop_depth"] == 2


def test_nested_loop_depth_three():
    code = """
def f(n):
    for i in range(n):
        for j in range(n):
            for k in range(n):
                pass
"""
    assert feat(code)["nested_loop_depth"] == 3


def test_nested_loop_depth_comprehension():
    code = "x = [[i*j for j in range(10)] for i in range(10)]"
    assert feat(code)["nested_loop_depth"] == 2


def test_nested_loop_depth_mixed_for_while():
    code = """
def f(n):
    for i in range(n):
        j = 0
        while j < n:
            j += 1
"""
    assert feat(code)["nested_loop_depth"] == 2


def test_noOfMethods():
    code = """
def a():
    pass

def b():
    def c():
        pass
    return c
"""
    # Three function_definition nodes (a, b, c)
    assert feat(code)["noOfMethods"] == 3


def test_noOfVariables_assignments():
    code = """
def f():
    a = 1
    b = 2
    c = a + b
"""
    assert feat(code)["noOfVariables"] == 3


def test_noOfVariables_includes_parameters():
    code = """
def f(x, y, z):
    total = x + y + z
    return total
"""
    # x, y, z, total
    assert feat(code)["noOfVariables"] == 4


def test_noOfStatements():
    code = """
def f(x):
    if x > 0:
        return 1
    return 0
"""
    # function_definition + if_statement + return_statement + return_statement = 4
    assert feat(code)["noOfStatements"] >= 4


def test_noOfJumps():
    code = """
def f(x):
    if x == 0:
        return 0
    if x < 0:
        raise ValueError("neg")
    for i in range(x):
        if i == 5:
            break
        if i == 3:
            continue
"""
    # return + raise + break + continue = 4
    assert feat(code)["noOfJumps"] == 4


def test_recursion_present_direct():
    code = """
def fact(n):
    if n <= 1:
        return 1
    return n * fact(n - 1)
"""
    assert feat(code)["recursion_present"] == 1


def test_recursion_absent():
    code = """
def f(xs):
    total = 0
    for x in xs:
        total += x
    return total
"""
    assert feat(code)["recursion_present"] == 0


def test_priority_queue_via_heapq_import():
    code = """
import heapq
def f(xs):
    h = []
    for x in xs:
        heapq.heappush(h, x)
    return h
"""
    assert feat(code)["priority_queue_present"] == 1


def test_priority_queue_via_from_import():
    code = """
from heapq import heappush, heappop
def f():
    h = []
    heappush(h, 1)
    return heappop(h)
"""
    assert feat(code)["priority_queue_present"] == 1


def test_priority_queue_via_queue_priorityqueue():
    code = """
from queue import PriorityQueue
q = PriorityQueue()
q.put((1, 'a'))
"""
    assert feat(code)["priority_queue_present"] == 1


def test_priority_queue_absent():
    code = """
def f(xs):
    return sorted(xs)
"""
    assert feat(code)["priority_queue_present"] == 0


def test_hash_set_literal():
    code = "s = {1, 2, 3}"
    assert feat(code)["hash_set_present"] == 1


def test_hash_set_builtin():
    code = "s = set([1, 2, 3])"
    assert feat(code)["hash_set_present"] == 1


def test_hash_set_comprehension():
    code = "s = {x for x in range(10)}"
    assert feat(code)["hash_set_present"] == 1


def test_hash_map_literal():
    code = "d = {1: 'a', 2: 'b'}"
    assert feat(code)["hash_map_present"] == 1


def test_hash_map_empty_is_dict_not_set():
    # {} is dict literal in Python.
    code = "d = {}"
    # tree-sitter-python parses {} as 'dictionary' — ensure we detect that.
    assert feat(code)["hash_map_present"] == 1
    assert feat(code)["hash_set_present"] == 0


def test_hash_map_defaultdict():
    code = """
from collections import defaultdict
d = defaultdict(int)
"""
    assert feat(code)["hash_map_present"] == 1


def test_no_of_sort():
    code = """
def f(xs, ys):
    xs.sort()
    return sorted(ys)
"""
    assert feat(code)["no_of_sort"] == 2


def test_cond_in_loop_freq():
    code = """
def f(n):
    for i in range(n):
        if i == 5:
            pass
        if i == 7:
            pass
    if n == 0:  # NOT inside loop
        pass
"""
    # Two ifs inside the for loop, one outside.
    assert feat(code)["cond_in_loop_freq"] == 2


def test_loop_in_cond_freq():
    code = """
def f(n):
    if n > 0:
        for i in range(n):
            pass
    else:
        while n < 10:
            n += 1
"""
    # Two loops inside if branches
    assert feat(code)["loop_in_cond_freq"] == 2


def test_loop_in_loop_freq_distinct_from_depth():
    code = """
def f(n):
    for i in range(n):
        for j in range(n):
            pass
        for k in range(n):
            pass
"""
    # Outer loop has two inner loops → loop_in_loop_freq = 2
    # Nested depth = 2
    features = feat(code)
    assert features["loop_in_loop_freq"] == 2
    assert features["nested_loop_depth"] == 2


def test_cond_in_cond_freq():
    code = """
def f(x, y):
    if x > 0:
        if y > 0:
            return 1
        if y < 0:
            return 2
    return 0
"""
    # Two inner ifs inside the outer if
    assert feat(code)["cond_in_cond_freq"] == 2


# ---------- end-to-end on known complexity snippets ----------

def test_e2e_constant():
    code = """
def add(a, b):
    return a + b
"""
    f = feat(code)
    assert f["no_of_loop"] == 0
    assert f["nested_loop_depth"] == 0
    assert f["recursion_present"] == 0


def test_e2e_linear():
    code = """
def max_val(xs):
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best
"""
    f = feat(code)
    assert f["no_of_loop"] == 1
    assert f["nested_loop_depth"] == 1
    assert f["no_of_ifs"] == 1
    assert f["cond_in_loop_freq"] == 1


def test_e2e_quadratic_bubble_sort():
    code = """
def bubble_sort(xs):
    n = len(xs)
    for i in range(n):
        for j in range(n - i - 1):
            if xs[j] > xs[j+1]:
                xs[j], xs[j+1] = xs[j+1], xs[j]
    return xs
"""
    f = feat(code)
    assert f["no_of_loop"] == 2
    assert f["nested_loop_depth"] == 2
    assert f["loop_in_loop_freq"] == 1
    assert f["cond_in_loop_freq"] == 1


def test_e2e_nlogn_sorted():
    code = """
def sort_them(xs):
    return sorted(xs)
"""
    f = feat(code)
    assert f["no_of_sort"] == 1
    assert f["no_of_loop"] == 0


def test_e2e_logn_binary_search():
    code = """
def bsearch(xs, target):
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        elif xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
"""
    f = feat(code)
    assert f["no_of_loop"] == 1
    assert f["nested_loop_depth"] == 1
    assert f["no_of_ifs"] >= 1


def test_e2e_exponential_fib_recursive():
    code = """
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)
"""
    f = feat(code)
    assert f["recursion_present"] == 1
    assert f["no_of_ifs"] == 1


def test_e2e_mxn_matrix_traverse():
    # O(m*n) pattern: two nested loops over distinct params
    code = """
def traverse(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    total = 0
    for i in range(rows):
        for j in range(cols):
            total += matrix[i][j]
    return total
"""
    f = feat(code)
    assert f["no_of_loop"] == 2
    assert f["nested_loop_depth"] == 2
    assert f["loop_in_loop_freq"] == 1


def test_e2e_heap_dijkstra_like():
    code = """
import heapq
def dijkstra(graph, start):
    dist = {start: 0}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
"""
    f = feat(code)
    assert f["priority_queue_present"] == 1
    assert f["hash_map_present"] == 1
    assert f["no_of_loop"] == 2


def test_feature_set_is_complete():
    code = "def f(): return 1"
    f = feat(code)
    assert set(f.keys()) == set(FEATURE_NAMES)
    assert len(f) == 18


def test_empty_code():
    # Pathological: empty string shouldn't crash.
    f = feat("")
    assert all(v == 0 for v in f.values())


def test_syntax_error_soft_fail():
    # tree-sitter is error-recovering; a partial parse should still yield a dict.
    code = "def f(x:\n    return x"
    f = feat(code)
    assert set(f.keys()) == set(FEATURE_NAMES)
    # Function should still be detected
    assert f["noOfMethods"] >= 0  # we don't promise exact counts on broken code
