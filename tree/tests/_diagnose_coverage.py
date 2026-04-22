"""Check which LeetCode problems we have labels for vs which we're missing.

Our hypothesis: the O(n) skew might reflect that we're losing harder problems
(which tend to be O(n^2) / O(n^3) / exponential) because their READMEs are
either structured differently or in Chinese-only form.
"""
from pathlib import Path
from collections import Counter
import pandas as pd
import re

REPO = Path("data/raw/doocs-leetcode")
ingested = pd.read_parquet("data/interim/doocs_leetcode.parquet")

_NUM_RE = re.compile(r"^(\d{4})\.")

# All doocs problems from main solution/ directory (ignore lcci/lcof which have
# different ID formats)
all_problems = set()
for p in REPO.glob("solution/*/*"):
    if p.is_dir():
        m = _NUM_RE.match(p.name)
        if m:
            all_problems.add(m.group(1))

labeled_problems = set(
    str(pid) for pid in ingested["problem_id"].unique()
    if isinstance(pid, str) and pid.isdigit() and len(pid) == 4
)

print(f"Total LeetCode problems in doocs (numeric IDs): {len(all_problems)}")
print(f"Labeled (captured complexity):                  {len(labeled_problems)}")
print(f"Missing:                                        {len(all_problems - labeled_problems)}")
print(f"Coverage: {100*len(labeled_problems)/len(all_problems):.1f}%")
print()

def bucket_of(pid: str) -> str:
    n = int(pid)
    return f"{(n // 500) * 500:04d}-{(n // 500) * 500 + 499:04d}"

all_by_bucket = Counter(bucket_of(p) for p in all_problems)
labeled_by_bucket = Counter(bucket_of(p) for p in labeled_problems)
print("Coverage by problem-number bucket (later buckets = harder problems):")
print(f"  {'bucket':<12s} {'total':>6s} {'labeled':>8s} {'pct':>6s}")
for b in sorted(all_by_bucket):
    t = all_by_bucket[b]
    l = labeled_by_bucket.get(b, 0)
    pct = 100 * l / t if t else 0
    bar = "#" * int(pct / 2)
    print(f"  {b:<12s} {t:>6d} {l:>8d} {pct:>5.1f}% {bar}")
print()

# Class distribution in ingested
print("Ingested class distribution:")
for label, cnt in ingested["label"].value_counts().items():
    pct = 100 * cnt / len(ingested)
    print(f"  {label:22s} {cnt:>5d} ({pct:>5.1f}%)")
