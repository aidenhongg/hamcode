"""Single source of truth for complexity labels, ordinal tiers, and the
binary pair-label derivation used by head training data generation."""

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

# Ordinal tier mapping (used by within-1-tier accuracy). Assumes m ≈ n for
# multi-variable classes. Equal tier => same complexity bucket.
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

# Binary pair labels emitted by `pipeline/10_make_pairwise.py` and consumed by
# the stacking heads (which classify pairs as same / A_faster). Pairs with
# tier_A > tier_B are canonicalized via swap upstream so only two outcomes
# survive into the head training data.
#   - "same"      : tier_A == tier_B
#   - "A_faster"  : tier_A <  tier_B
PAIR_LABELS: tuple[str, ...] = ("same", "A_faster")
PAIR_LABEL_TO_IDX: dict[str, int] = {lab: i for i, lab in enumerate(PAIR_LABELS)}


def pair_label_from_labels(label_a: str, label_b: str) -> str:
    """Return the binary pair label, or 'B_faster' if the pair violates the
    canonical ordering (tier_A > tier_B). Callers must filter/swap such pairs
    before emitting them into the dataset.
    """
    ta, tb = TIER[label_a], TIER[label_b]
    if ta < tb:
        return "A_faster"
    if ta > tb:
        return "B_faster"   # signal: swap or drop upstream
    return "same"
