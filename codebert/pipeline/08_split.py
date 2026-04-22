"""Split balanced data into train/val/test by problem_id, stratified by label.

Each problem_id goes entirely into one split (no leakage). Records without a
problem_id default to the train split.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_path", default="data/interim/balanced.jsonl")
    ap.add_argument("--out", default="data/interim/split.jsonl")
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    records: list[dict] = []
    with open(args.in_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # Derive a stable "problem label" for each pid (first label seen).
    pid_label: dict[str, str] = {}
    for r in records:
        pid = r.get("problem_id")
        if pid and pid not in pid_label:
            pid_label[pid] = r["label"]

    # Bucket pids by label, shuffle, assign splits.
    pid_buckets: dict[str, list[str]] = defaultdict(list)
    for pid, lab in pid_label.items():
        pid_buckets[lab].append(pid)

    assignment: dict[str, str] = {}
    for lab, pids in pid_buckets.items():
        rng.shuffle(pids)
        n = len(pids)
        n_tr = int(n * args.train)
        n_val = int(n * args.val)
        for i, pid in enumerate(pids):
            if i < n_tr:
                assignment[pid] = "train"
            elif i < n_tr + n_val:
                assignment[pid] = "val"
            else:
                assignment[pid] = "test"

    n_split: dict[str, int] = defaultdict(int)
    per_split_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with open(args.out, "w", encoding="utf-8") as fout:
        for r in records:
            pid = r.get("problem_id")
            split = assignment.get(pid, "train") if pid else "train"
            r["split"] = split
            n_split[split] += 1
            per_split_class[split][r["label"]] += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[08] split counts: {dict(n_split)}", flush=True)
    for sp in ("train", "val", "test"):
        print(f"[08]   {sp}: {dict(per_split_class[sp])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
