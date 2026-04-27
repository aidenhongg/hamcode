"""Unit tests for AST feature extractor.

Checks known feature values on curated Python snippets — one per complexity
class — plus edge cases (empty code, syntax error, unicode).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stacking.features.ast_features import (
    FEATURE_NAMES,
    N_FEATURES,
    extract_differenced,
    extract_features,
)


LINEAR_CODE = """
def sum_array(nums):
    total = 0
    for x in nums:
        total += x
    return total
"""

QUADRATIC_CODE = """
def pair_sum(nums):
    result = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == 0:
                result.append((i, j))
    return result
"""

RECURSIVE_CODE = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

HEAP_CODE = """
import heapq
def top_k(nums, k):
    heap = []
    for x in nums:
        heapq.heappush(heap, x)
        if len(heap) > k:
            heapq.heappop(heap)
    return sorted(heap)
"""

SET_MAP_CODE = """
def unique_count(arr):
    seen = set()
    counts = {}
    for x in arr:
        seen.add(x)
        counts[x] = counts.get(x, 0) + 1
    return sorted(seen), counts
"""


def _idx(name: str) -> int:
    return FEATURE_NAMES.index(name)


def test_schema_size():
    assert N_FEATURES == len(FEATURE_NAMES) == 21


def test_linear_feature_counts():
    f = extract_features(LINEAR_CODE).values
    assert f[_idx("no_of_loop")] == 1
    assert f[_idx("nested_loop_depth")] == 1
    assert f[_idx("no_of_ifs")] == 0
    assert f[_idx("noOfMethods")] == 1
    assert f[_idx("recursion_present")] == 0
    assert f[_idx("priority_queue_present")] == 0
    assert f[_idx("hash_set_present")] == 0
    assert f[_idx("hash_map_present")] == 0
    assert f[_idx("no_of_sort")] == 0


def test_quadratic_nested_depth():
    f = extract_features(QUADRATIC_CODE).values
    assert f[_idx("no_of_loop")] >= 2
    assert f[_idx("nested_loop_depth")] == 2
    assert f[_idx("no_of_ifs")] >= 1
    # Nested structure: at least one cond inside a loop
    assert f[_idx("cond_in_loop_freq")] >= 1
    # Two loops nested → loop_in_loop_freq counts the inner occurrence
    assert f[_idx("loop_in_loop_freq")] >= 1


def test_recursive_detection():
    f = extract_features(RECURSIVE_CODE).values
    assert f[_idx("recursion_present")] == 1
    assert f[_idx("no_of_ifs")] >= 1


def test_heap_detection():
    f = extract_features(HEAP_CODE).values
    assert f[_idx("priority_queue_present")] == 1
    assert f[_idx("no_of_sort")] >= 1  # sorted(heap)
    # heap.sort() would also be caught — confirm no false positive
    assert f[_idx("no_of_loop")] >= 1


def test_set_map_detection():
    f = extract_features(SET_MAP_CODE).values
    assert f[_idx("hash_set_present")] == 1
    assert f[_idx("hash_map_present")] == 1
    assert f[_idx("no_of_sort")] >= 1


def test_empty_code():
    f = extract_features("").values
    assert f.shape == (N_FEATURES,)
    assert np.all(f == 0)


def test_whitespace_only():
    f = extract_features("   \n\n\t").values
    assert np.all(f == 0)


def test_syntax_error():
    f = extract_features("def foo(: ::").values
    assert f.shape == (N_FEATURES,)
    assert np.all(f == 0)


def test_unicode_identifiers():
    code = "def 日本語(x): return x + 1"
    f = extract_features(code).values
    assert f[_idx("noOfMethods")] == 1


def test_differenced_shape():
    row = extract_differenced(LINEAR_CODE, QUADRATIC_CODE)
    assert row.shape == (4 * N_FEATURES,)


def test_differenced_diff_symmetry():
    # diff(a,b) + diff(b,a) = 0 elementwise
    ab = extract_differenced(LINEAR_CODE, QUADRATIC_CODE)
    ba = extract_differenced(QUADRATIC_CODE, LINEAR_CODE)
    # layout per feature: [a, b, b-a, |b-a|] — so index 2 of each block is signed diff
    signed_ab = ab[2::4]
    signed_ba = ba[2::4]
    np.testing.assert_allclose(signed_ab + signed_ba, 0.0, atol=1e-6)


def test_differenced_abs_invariance():
    # |diff| must be invariant under swap
    ab = extract_differenced(LINEAR_CODE, QUADRATIC_CODE)
    ba = extract_differenced(QUADRATIC_CODE, LINEAR_CODE)
    abs_ab = ab[3::4]
    abs_ba = ba[3::4]
    np.testing.assert_allclose(abs_ab, abs_ba, atol=1e-6)


def test_differenced_self_pair_is_zero_diff():
    # diff(a, a) == 0 for the diff channels; raw A/B channels should be equal
    aa = extract_differenced(LINEAR_CODE, LINEAR_CODE)
    signed = aa[2::4]
    absd = aa[3::4]
    np.testing.assert_allclose(signed, 0.0, atol=1e-6)
    np.testing.assert_allclose(absd, 0.0, atol=1e-6)
    # a[0::4] == a[1::4] (raw A == raw B when code is identical)
    np.testing.assert_allclose(aa[0::4], aa[1::4], atol=1e-6)


def test_nested_loop_depth_three():
    code = """
def cubed(n):
    s = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s += 1
    return s
"""
    f = extract_features(code).values
    assert f[_idx("nested_loop_depth")] == 3


def test_cyclomatic_nonzero_for_branching():
    f = extract_features(QUADRATIC_CODE).values
    assert f[_idx("cyclomatic_max")] > 1
    assert f[_idx("cyclomatic_sum")] >= f[_idx("cyclomatic_max")]


def test_cyclomatic_zero_for_no_functions():
    f = extract_features("x = 1\ny = 2").values
    assert f[_idx("cyclomatic_max")] == 0
    assert f[_idx("cyclomatic_sum")] == 0
    assert f[_idx("cyclomatic_mean")] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
