"""Single source of truth for complexity labels, tiers, and pairwise classes."""

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

# Ordinal tier for pairwise ranking. Assumes m ≈ n for multi-var.
# Equal tier -> "same".
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

# Ternary pairwise labels. Stable order for the classification head.
PAIR_LABELS: tuple[str, ...] = ("A_faster", "same", "B_faster")
NUM_PAIR_LABELS = len(PAIR_LABELS)
PAIR_LABEL_TO_IDX: dict[str, int] = {lab: i for i, lab in enumerate(PAIR_LABELS)}


def pair_label_from_labels(label_a: str, label_b: str) -> str:
    ta, tb = TIER[label_a], TIER[label_b]
    if ta < tb:
        return "A_faster"
    if ta > tb:
        return "B_faster"
    return "same"
