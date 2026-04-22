"""Tests for the raw-complexity-string normalizer and comment miner."""

from __future__ import annotations

import pytest

from complexity.normalizer import normalize, normalize_any
from complexity.ingest.utils import find_complexity_in_text


@pytest.mark.parametrize("raw, expected", [
    ("O(1)", "O(1)"),
    ("O(n)", "O(n)"),
    ("O(log n)", "O(log n)"),
    ("O(n log n)", "O(n log n)"),
    ("O(n*log n)", "O(n log n)"),
    ("O(n^2)", "O(n^2)"),
    ("O(n^3)", "O(n^3)"),
    ("O(2^n)", "exponential"),
    ("O(n!)", "exponential"),
    ("O(m + n)", "O(m+n)"),
    ("O(n + m)", "O(m+n)"),
    ("O(m * n)", "O(m*n)"),
    ("O(m log n)", "O(m log n)"),
    ("O((m + n) log(m + n))", "O((m+n) log(m+n))"),
    ("O((m+n) log(m+n))", "O((m+n) log(m+n))"),
    ("O(rows * cols)", "O(m*n)"),
    ("O(V + E)", "O(m+n)"),
    ("O(k * n)", "O(n)"),
    ("O(n + k)", "O(n)"),
    (r"O(\log n)", "O(log n)"),
])
def test_normalize_o_expressions(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw", [
    r"O(\sqrt{n})",
    "O(a * b * c)",         # 3+ distinct variables
    "O(m + n * log n)",     # straddles tiers — reject
    "",
])
def test_normalize_rejects(raw):
    assert normalize(raw) is None


@pytest.mark.parametrize("raw, expected", [
    ("constant", "O(1)"),
    ("linear", "O(n)"),
    ("quadratic", "O(n^2)"),
    ("cubic", "O(n^3)"),
    ("logn", "O(log n)"),
    ("nlogn", "O(n log n)"),
    ("np", "exponential"),
    ("np-hard", "exponential"),
    ("exponential", "exponential"),
])
def test_normalize_any_bare_words(raw, expected):
    assert normalize_any(raw) == expected


@pytest.mark.parametrize("text, expected", [
    ("# Time complexity: O(n log n)", "O(n log n)"),
    ('"""Time: O(n)"""', "O(n)"),
    ("// Time Complexity: O(m*n)", "O(m*n)"),
    ("# O(m + n) time", "O(m + n)"),
    ("// A lovely solution: O(2^n) time", "O(2^n)"),
    ("Time Complexity: O((m + n) log(m + n))", "O((m + n) log(m + n))"),
    ("Time - O(log n)", "O(log n)"),
])
def test_find_complexity(text, expected):
    assert find_complexity_in_text(text) == expected


def test_find_complexity_none():
    assert find_complexity_in_text("no complexity discussion here") is None
    assert find_complexity_in_text("") is None
