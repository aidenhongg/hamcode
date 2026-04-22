"""Inspect doocs-leetcode rejects to find patterns we're missing."""
import json
import re
from collections import Counter
from pathlib import Path

rejects = [json.loads(l) for l in Path("data/audit/doocs_leetcode_rejects.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
print("reject reasons:", Counter(r["reason"] for r in rejects))
print()

no_comp = [r for r in rejects if r.get("reason") == "no_complexity_in_readme"]
print(f"sampling from {len(no_comp)} no_complexity READMEs to find complexity mentions:")
print()

found_but_missed = 0
for r in no_comp[:30]:
    p = Path(r["dir"]) / "README_EN.md"
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8", errors="replace")[:6000]
    # Cast a wide net to find any complexity notation
    candidates = re.findall(
        r"(?i)(?:time[^.\n]{0,20}|space[^.\n]{0,20})?(?:\$\$?)?O\s*\([^\n]{1,80}\)(?:\$\$?)?",
        text,
    )
    if candidates:
        found_but_missed += 1
        print(f"  {str(p)[-70:]}")
        for c in candidates[:3]:
            print(f"    -> {c!r}")
        print()

print(f"\nRejected READMEs that DO contain O(...) somewhere: {found_but_missed} / 30 sampled")
