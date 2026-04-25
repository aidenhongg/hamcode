"""Split balanced data into train/val/test by problem_id, stratified by (language, label).

Each problem_id (across ALL languages — same LeetCode 0001 in Python and Java
shares one problem_id) goes entirely into one split (no leakage). Records
without a problem_id get a synthetic per-record id so they don't collide with
each other and default to `train` allocation rules.

The stratification key for split assignment is (language, label_for_pid):
  - For each problem_id, derive a "primary" label from the first record we see.
  - Bucket pids by (language, primary_label); shuffle within each bucket;
    assign 80/10/10.
"""

from __future__ import annotations

import argparse
import json
import random
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

    # Synthesize a unique pid for records lacking one (avoids "all None pid -> all-train").
    for i, r in enumerate(records):
        if not r.get("problem_id"):
            r["problem_id"] = f"_anon-{r.get('code_sha256', i)[:12]}-{i}"

    # First (language, label) per pid — pids are language-tagged because in our
    # ingest each (lang) fence in one block becomes its own per-language record;
    # however the SAME LeetCode number 0001 is the SAME problem_id across
    # languages, and we want it to land in the same split for ALL languages to
    # prevent cross-language leakage. So we key the split assignment by pid
    # alone and stratify by *one* representative (language, label).
    pid_label: dict[str, tuple[str, str]] = {}
    pid_records: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        pid = r["problem_id"]
        pid_records[pid].append(r)
        if pid not in pid_label:
            pid_label[pid] = (r.get("language", "python"), r["label"])

    # Bucket pids by (language, label); stratify-assign within each bucket.
    pid_buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pid, key in pid_label.items():
        pid_buckets[key].append(pid)

    assignment: dict[str, str] = {}
    for key, pids in pid_buckets.items():
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

    # Apply assignment to every record (every record sharing pid pins to same split)
    n_split: dict[str, int] = defaultdict(int)
    per_split_lang_class: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
    with open(args.out, "w", encoding="utf-8") as fout:
        for r in records:
            split = assignment.get(r["problem_id"], "train")
            r["split"] = split
            n_split[split] += 1
            per_split_lang_class[split][(r.get("language", "python"), r["label"])] += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[08] split counts: {dict(n_split)}", flush=True)
    for sp in ("train", "val", "test"):
        per_lang = defaultdict(int)
        for (lang, _lab), n in per_split_lang_class[sp].items():
            per_lang[lang] += n
        per_lang_str = " ".join(f"{lang}={n}" for lang, n in sorted(per_lang.items()))
        print(f"[08]   {sp:<5s} per-lang totals: {per_lang_str}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
