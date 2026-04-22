"""Generate pairwise pairs: same-problem (gold) + cross-problem (synthetic).

Pairs respect split boundaries — a train pair uses only train snippets.
Ordinal tier map from common/labels is used to derive the ternary label.
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from collections import defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common.labels import POINT_LABELS, pair_label_from_labels
from common.schemas import PAIR_SCHEMA


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_path", default="data/processed/pointwise.parquet")
    ap.add_argument("--out", default="data/processed/pairwise.parquet")
    ap.add_argument("--per_cell_cap", type=int, default=3000)
    ap.add_argument("--target_total", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    table = pq.read_table(args.in_path)
    df = table.to_pylist()

    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in df:
        by_split[r["split"]].append(r)

    all_rows: list[dict] = []
    pair_idx = 0

    # target per-cell scales by split fraction
    split_fractions = {"train": 0.80, "val": 0.10, "test": 0.10}

    for sp, rows in by_split.items():
        share = split_fractions.get(sp, 0.80)
        split_target = int(args.target_total * share)
        cells = [(la, lb) for i, la in enumerate(POINT_LABELS) for lb in POINT_LABELS[i:]]
        per_cell = max(1, split_target // max(1, len(cells)))

        # same-problem pairs
        by_pid: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["problem_id"]:
                by_pid[r["problem_id"]].append(r)
        sp_pairs: list[tuple[dict, dict, bool]] = []
        for _pid, items in by_pid.items():
            for a, b in itertools.combinations(items, 2):
                sp_pairs.append((a, b, True))

        # cross-problem pairs, uniform per (class_a, class_b) cell
        by_label: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_label[r["label"]].append(r)

        cross_pairs: list[tuple[dict, dict, bool]] = []
        for la, lb in cells:
            la_rows = by_label.get(la, [])
            lb_rows = by_label.get(lb, [])
            if not la_rows or not lb_rows:
                continue
            cap = min(args.per_cell_cap, per_cell)
            attempts = 0
            added = 0
            while added < cap and attempts < cap * 4:
                a = rng.choice(la_rows)
                b = rng.choice(lb_rows)
                attempts += 1
                if a["code_sha256"] == b["code_sha256"]:
                    continue
                if a["problem_id"] and a["problem_id"] == b["problem_id"]:
                    continue
                cross_pairs.append((a, b, False))
                added += 1

        rng.shuffle(sp_pairs)
        rng.shuffle(cross_pairs)

        for a, b, same_prob in sp_pairs + cross_pairs:
            ternary = pair_label_from_labels(a["label"], b["label"])
            all_rows.append({
                "pair_id": f"p{pair_idx:07d}",
                "code_a": a["code"],
                "code_b": b["code"],
                "label_a": a["label"],
                "label_b": b["label"],
                "ternary": ternary,
                "same_problem": same_prob,
                "tokens_combined": int(a.get("tokens_graphcodebert") or 0)
                                   + int(b.get("tokens_graphcodebert") or 0),
                "split": sp,
            })
            pair_idx += 1

    table = pa.Table.from_pylist(all_rows, schema=PAIR_SCHEMA)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")
    for sp in ("train", "val", "test"):
        mask = pc.equal(table["split"], sp)
        pq.write_table(table.filter(mask), out.parent / f"pair_{sp}.parquet", compression="zstd")

    by_t: dict[str, int] = defaultdict(int)
    by_sp: dict[str, int] = defaultdict(int)
    for r in all_rows:
        by_t[r["ternary"]] += 1
        by_sp[r["split"]] += 1
    print(f"[10] total pairs={len(all_rows)}", flush=True)
    print(f"[10] by ternary: {dict(by_t)}", flush=True)
    print(f"[10] by split:   {dict(by_sp)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
