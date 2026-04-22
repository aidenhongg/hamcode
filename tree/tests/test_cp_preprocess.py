"""Tests for the competitive-programming boilerplate stripper."""

from __future__ import annotations

from complexity.cp_preprocess import preprocess


def normalize_ws(s: str) -> str:
    return "\n".join(line for line in s.split("\n") if line.strip())


def test_strips_imports():
    code = """
import sys
from heapq import heappush, heappop
import os, io

def solve(n):
    return n * 2
"""
    out = preprocess(code)
    assert "import" not in out
    assert "from " not in out
    assert "def solve" in out


def test_strips_io_helper_defs():
    code = """
import sys
def rs(): return sys.stdin.readline().rstrip()
def ri(): return int(sys.stdin.readline())
def ria(): return list(map(int, sys.stdin.readline().split()))
def ws(s): sys.stdout.write(s + '\\n')

n = int(input())
for i in range(n):
    print(i)
"""
    out = preprocess(code)
    assert "def rs" not in out
    assert "def ri" not in out
    assert "def ria" not in out
    assert "def ws" not in out
    assert "for i in range(n)" in out


def test_strips_library_aliases():
    code = """
from collections import defaultdict, Counter, deque

D = defaultdict
C = Counter
Q = deque
enum = enumerate
BL = bisect_left

def solve():
    d = D(int)
    return d
"""
    out = preprocess(code)
    assert "D = defaultdict" not in out
    assert "C = Counter" not in out
    assert "enum = enumerate" not in out
    assert "def solve" in out


def test_strips_t_harness():
    code = """
t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    print(sum(a))
"""
    out = preprocess(code)
    assert "for _ in range(t)" not in out
    assert "print(sum(a))" in out


def test_strips_inline_t_harness():
    code = """
for _ in range(int(input())):
    n = int(input())
    print(n * 2)
"""
    out = preprocess(code)
    assert "for _ in" not in out
    assert "print(n * 2)" in out


def test_keeps_algorithmic_loops():
    """Algorithmic for-loops must survive."""
    code = """
def solve(a):
    total = 0
    for x in a:
        for y in a:
            total += x * y
    return total
"""
    out = preprocess(code)
    assert "for x in a" in out
    assert "for y in a" in out
    assert "def solve" in out


def test_preserves_algorithm_after_strip():
    """An O(n^2) snippet should retain its nested-loop structure after stripping."""
    code = """
import sys
from collections import defaultdict
D = defaultdict

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    seen = D(int)
    for i in range(n):
        for j in range(n):
            if a[i] == a[j]:
                seen[a[i]] += 1
    print(len(seen))
"""
    out = preprocess(code)
    # Harness stripped
    assert "import" not in out
    assert "D = defaultdict" not in out
    assert "for _ in" not in out
    # Algorithmic core preserved
    assert "for i in range(n)" in out
    assert "for j in range(n)" in out


def test_empty_code():
    assert preprocess("") == ""


def test_no_boilerplate_unchanged():
    code = "def f(x):\n    return x + 1\n"
    out = preprocess(code)
    assert "def f" in out
    assert "return x + 1" in out


def test_falls_back_on_parse_error():
    """If the code doesn't parse, don't crash — return whatever lines remain."""
    code = """
import sys
def foo(x:
    return x  # broken syntax
"""
    # Should not raise
    out = preprocess(code)
    # Imports are still line-level strippable regardless of parse
    assert "import sys" not in out
