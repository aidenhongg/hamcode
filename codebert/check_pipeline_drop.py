"""Diagnose where samples are lost at each pipeline stage.

Reads the audit files + intermediate jsonl + final parquet and prints a funnel.
Run after `bash run_pipeline.sh` to see what dropped where.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open("r", encoding="utf-8") if _.strip())


def count_parquet(path: Path) -> int:
    if not path.exists():
        return 0
    return pq.read_table(path).num_rows


def top_reject_reasons(path: Path, field: str = "reject_reason", limit: int = 10) -> list:
    if not path.exists():
        return []
    counts: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts[r.get(field, "unknown")[:100]] += 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:limit]


def main() -> int:
    print("=== DATA PIPELINE FUNNEL ===\n")

    # Sources
    leetcode_parsed = count_jsonl(Path("data/interim/parsed/leetcode.jsonl"))
    codeforces_parsed = count_jsonl(Path("data/interim/parsed/codeforces.jsonl"))
    codecomplex_parsed = count_jsonl(Path("data/interim/parsed/codecomplex.jsonl"))
    supplemental_parsed = count_jsonl(Path("data/interim/parsed/supplemental.jsonl"))
    total_parsed = (leetcode_parsed + codeforces_parsed
                    + codecomplex_parsed + supplemental_parsed)

    print(f"  [02]  leetcode parsed:        {leetcode_parsed:>6}")
    print(f"  [02b] codeforces parsed:      {codeforces_parsed:>6}")
    print(f"  [03]  codecomplex parsed:     {codecomplex_parsed:>6}")
    print(f"  [04]  supplemental parsed:    {supplemental_parsed:>6}")
    print(f"        TOTAL PARSED:           {total_parsed:>6}")

    # Parse failures
    lc_fail = count_jsonl(Path("data/audit/leetcode_parse_failures.jsonl"))
    print(f"\n  [02] leetcode parse failures (no python/no complexity): {lc_fail}")

    # Normalization
    normalized = count_jsonl(Path("data/interim/normalized/combined.jsonl"))
    rejected = count_jsonl(Path("data/audit/normalize_rejects.jsonl"))
    print(f"\n  [05] after normalize:        {normalized:>6}"
          f"   (rejected {rejected} -> audit/normalize_rejects.jsonl)")
    for reason, n in top_reject_reasons(Path("data/audit/normalize_rejects.jsonl")):
        print(f"          {n:>5} x  {reason}")

    # Dedupe / filter
    filtered = count_jsonl(Path("data/interim/filtered.jsonl"))
    drop_filter = normalized - filtered if normalized and filtered else 0
    print(f"\n  [06] after dedupe+filter:    {filtered:>6}"
          f"   (-{drop_filter} to minhash/ast/token-length)")

    # Balance
    balanced = count_jsonl(Path("data/interim/balanced.jsonl"))
    print(f"  [07] after balance+augment:  {balanced:>6}   (augmentations added if any)")

    # Split
    split = count_jsonl(Path("data/interim/split.jsonl"))
    print(f"  [08] after split:            {split:>6}   (same records, split field added)")

    # Final parquet
    pw_total = count_parquet(Path("data/processed/pointwise.parquet"))
    pw_train = count_parquet(Path("data/processed/train.parquet"))
    pw_val = count_parquet(Path("data/processed/val.parquet"))
    pw_test = count_parquet(Path("data/processed/test.parquet"))
    print(f"  [09] pointwise.parquet:      {pw_total:>6}"
          f"   (train={pw_train} val={pw_val} test={pw_test})")

    pair_total = count_parquet(Path("data/processed/pairwise.parquet"))
    print(f"  [10] pairwise.parquet:       {pair_total:>6}")

    # Funnel summary
    if total_parsed > 0 and pw_total > 0:
        retention = 100 * pw_total / total_parsed
        print(f"\n  overall retention: {pw_total}/{total_parsed} = {retention:.1f}%")
        if retention < 50:
            print("  [WARN] less than half of parsed records made it through.")
            print("         Main culprits are usually: aggressive MinHash dedup (06),")
            print("         normalizer rejects (05), per-class cap (07).")
            print("         Try:  python pipeline/06_dedupe_filter.py --threshold 0.95")
            print("         Or:   python pipeline/07_balance_augment.py --cap_per_class 2000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
