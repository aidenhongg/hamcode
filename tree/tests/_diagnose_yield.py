"""Trace where records are being lost in the doocs/leetcode pipeline."""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path("data/raw/doocs-leetcode")

# Stage 1: total problem directories
problem_dirs = set()
for root in ("solution", "lcof", "lcof2", "lcci", "lcp"):
    base = REPO / root
    if not base.exists(): continue
    # problem dirs have README_EN.md or README.md
    for readme in base.rglob("README*.md"):
        if readme.name in ("README.md", "README_EN.md"):
            problem_dirs.add(readme.parent)

print(f"Total problem dirs (with any README): {len(problem_dirs)}")

# Stage 2: problems with README_EN.md
with_en = [d for d in problem_dirs if (d / "README_EN.md").exists()]
with_zh_only = [d for d in problem_dirs if not (d / "README_EN.md").exists() and (d / "README.md").exists()]
print(f"  with README_EN.md: {len(with_en)}")
print(f"  with README.md only (Chinese): {len(with_zh_only)}")

# Stage 3: problems with at least one Python solution
has_py = lambda d: any(p.is_file() and re.match(r"^Solution\d*\.py$", p.name) for p in d.iterdir())
en_with_py = [d for d in with_en if has_py(d)]
zh_with_py = [d for d in with_zh_only if has_py(d)]
print(f"  with EN README + Python: {len(en_with_py)}")
print(f"  with ZH README only + Python: {len(zh_with_py)}")

# Stage 4: rejects from last ingest
rejects = []
rej_path = Path("data/audit/doocs_leetcode_rejects.jsonl")
if rej_path.exists():
    rejects = [json.loads(l) for l in rej_path.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"\nReject reasons from last run:")
for reason, count in Counter(r["reason"] for r in rejects).most_common():
    print(f"  {reason:30s} {count}")

# Stage 5: per-source counts in final features
features = pd.read_parquet("data/features/all.parquet") if Path("data/features/all.parquet").exists() else None
if features is not None:
    print(f"\nFinal records per source in data/features/all.parquet:")
    for src, cnt in features["source"].value_counts().items():
        print(f"  {src:30s} {cnt}")

# Stage 6: for 'no_complexity_in_readme' rejects, check how many DO have mineable complexity in a ZH-only form
no_comp_rejects = [r for r in rejects if r.get("reason") == "no_complexity_in_readme"][:50]
print(f"\nSample of first 5 'no_complexity' rejects — what their READMEs actually contain:")
for r in no_comp_rejects[:5]:
    p = Path(r["dir"])
    for name in ("README_EN.md", "README.md"):
        readme = p / name
        if readme.exists():
            text = readme.read_text(encoding="utf-8", errors="replace")
            # wide net for any O(...) or complexity mention
            o_matches = re.findall(r"O\s*\([^)\n]{1,80}\)", text[:8000])
            cn_time = "时间复杂度" in text
            print(f"  {name[:11]} dir={str(p)[-55:]}")
            print(f"    O(...) occurrences: {len(o_matches)}; first 3: {o_matches[:3]}")
            print(f"    Chinese '时间复杂度' (time complexity) present: {cn_time}")
            break
