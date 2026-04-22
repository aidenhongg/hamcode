"""Emit a human-readable + JSON audit report for the processed dataset.

Writes data/audit/stats.json and prints a summary. Flags thin classes (<200).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

import pyarrow.parquet as pq

from common.labels import POINT_LABELS


def _pointwise_stats(path: Path, samples_per_class: int) -> dict:
    if not path.exists():
        return {"found": False}
    rows = pq.read_table(path).to_pylist()
    by_class: dict[str, int] = defaultdict(int)
    by_split: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    by_split_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    examples: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["label"]] += 1
        by_split[r["split"]] += 1
        by_source[r["source"]] += 1
        by_split_class[r["split"]][r["label"]] += 1
        if len(examples[r["label"]]) < samples_per_class:
            examples[r["label"]].append({
                "source": r["source"],
                "problem_id": r["problem_id"],
                "code_prefix": (r["code"] or "")[:220],
            })
    return {
        "found": True,
        "total": len(rows),
        "by_class": dict(by_class),
        "by_split": dict(by_split),
        "by_source": dict(by_source),
        "by_split_class": {s: dict(c) for s, c in by_split_class.items()},
        "examples": dict(examples),
    }


def _pairwise_stats(path: Path) -> dict:
    if not path.exists():
        return {"found": False}
    rows = pq.read_table(path).to_pylist()
    by_ternary: dict[str, int] = defaultdict(int)
    by_split: dict[str, int] = defaultdict(int)
    by_cell: dict[str, int] = defaultdict(int)
    n_same_problem = 0
    for r in rows:
        by_ternary[r["ternary"]] += 1
        by_split[r["split"]] += 1
        by_cell[f"{r['label_a']} | {r['label_b']}"] += 1
        if r["same_problem"]:
            n_same_problem += 1
    top_cells = dict(sorted(by_cell.items(), key=lambda kv: -kv[1])[:20])
    return {
        "found": True,
        "total": len(rows),
        "by_ternary": dict(by_ternary),
        "by_split": dict(by_split),
        "same_problem_pairs": n_same_problem,
        "top_cells": top_cells,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pointwise", default="data/processed/pointwise.parquet")
    ap.add_argument("--pairwise", default="data/processed/pairwise.parquet")
    ap.add_argument("--out", default="data/audit/stats.json")
    ap.add_argument("--samples_per_class", type=int, default=3)
    args = ap.parse_args()

    stats = {
        "pointwise": _pointwise_stats(Path(args.pointwise), args.samples_per_class),
        "pairwise": _pairwise_stats(Path(args.pairwise)),
    }

    # Thin-class alert
    thin: list[dict] = []
    p = stats["pointwise"]
    if p.get("found"):
        for lab in POINT_LABELS:
            n = p["by_class"].get(lab, 0)
            if n < 200:
                thin.append({"label": lab, "count": n})
    stats["thin_classes_warning"] = thin

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # Pretty print
    print("=== AUDIT STATS ===", flush=True)
    if p.get("found"):
        print(f"Pointwise total: {p['total']}", flush=True)
        print("Pointwise by class:")
        for lab in POINT_LABELS:
            mark = " [THIN]" if p["by_class"].get(lab, 0) < 200 else ""
            print(f"  {lab:<24} {p['by_class'].get(lab, 0):>5}{mark}", flush=True)
        print(f"Pointwise by split: {p['by_split']}", flush=True)
        print(f"Pointwise by source: {p['by_source']}", flush=True)
    pw = stats["pairwise"]
    if pw.get("found"):
        print(f"\nPairwise total: {pw['total']}", flush=True)
        print(f"Pairwise by ternary: {pw['by_ternary']}", flush=True)
        print(f"Pairwise by split:   {pw['by_split']}", flush=True)
        print(f"Same-problem pairs:  {pw['same_problem_pairs']}", flush=True)
    if thin:
        print(f"\n[WARN] thin classes (<200): {[t['label'] for t in thin]}", flush=True)
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
