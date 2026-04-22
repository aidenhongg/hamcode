"""Inspect the downloaded python_data.jsonl schema."""
import json
from collections import Counter
from pathlib import Path

p = Path("data/raw/codecomplex_python/python_data.jsonl")
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"Total Python rows: {len(rows)}")
print()
print("First row keys:", list(rows[0].keys()))
print()
print("First row (with src truncated):")
for k, v in rows[0].items():
    if isinstance(v, str) and len(v) > 200:
        print(f"  {k}: {v[:200]}...")
    else:
        print(f"  {k}: {v!r}")
print()
print("Unique 'complexity' values (with counts):")
for label, cnt in Counter(r.get("complexity", "?") for r in rows).most_common():
    print(f"  {label!r:20s} {cnt}")
print()
print("Unique 'from' values:")
for v, cnt in Counter(r.get("from", "?") for r in rows).most_common():
    print(f"  {v!r:20s} {cnt}")
