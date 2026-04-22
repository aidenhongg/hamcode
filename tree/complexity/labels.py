"""Canonical complexity labels.

Mirrors codebert/common/labels.py so the two projects classify into the same
label space. Only the POINT side is needed here (no pair task).
"""

from __future__ import annotations

# 11 canonical pointwise classes. Order is stable — used as the label index.
POINT_LABELS: tuple[str, ...] = (
    "O(1)",
    "O(log n)",
    "O(n)",
    "O(n log n)",
    "O(n^2)",
    "O(n^3)",
    "exponential",
    "O(m+n)",
    "O(m*n)",
    "O(m log n)",
    "O((m+n) log(m+n))",
)
NUM_POINT_LABELS = len(POINT_LABELS)

LABEL_TO_IDX: dict[str, int] = {lab: i for i, lab in enumerate(POINT_LABELS)}
IDX_TO_LABEL: dict[int, str] = {i: lab for i, lab in enumerate(POINT_LABELS)}

# Ordinal tier used for diagnostics and class-weight heuristics. Assumes m ≈ n
# for multi-var classes when collapsed to a single axis.
TIER: dict[str, int] = {
    "O(1)": 0,
    "O(log n)": 1,
    "O(n)": 2,
    "O(m+n)": 2,
    "O(n log n)": 3,
    "O(m log n)": 3,
    "O((m+n) log(m+n))": 3,
    "O(n^2)": 4,
    "O(m*n)": 4,
    "O(n^3)": 5,
    "exponential": 6,
}
assert set(TIER) == set(POINT_LABELS), "TIER must cover all 11 classes"

MULTIVAR_LABELS: frozenset[str] = frozenset(
    {"O(m+n)", "O(m*n)", "O(m log n)", "O((m+n) log(m+n))"}
)
