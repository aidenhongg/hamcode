"""Show what raw_complexity strings are failing to normalize."""
import json
from collections import Counter
from pathlib import Path

rejects = [json.loads(l) for l in Path("data/audit/doocs_leetcode_rejects.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]

normalize_fails = [r for r in rejects if r.get("reason") == "normalize_fail"]
raws = [r.get("raw", "") for r in normalize_fails]

# Deduplicate and count unique raw strings
counts = Counter(raws)
print(f"Total normalize_fail: {len(normalize_fails)}")
print(f"Unique raw strings:   {len(counts)}")
print()
print("Top 40 failing raw strings:")
for raw, cnt in counts.most_common(40):
    # Truncate long ones
    r = raw if len(raw) < 80 else raw[:77] + "..."
    print(f"  {cnt:4d}  {r!r}")
