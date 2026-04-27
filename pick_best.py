"""Scan multiple run directories and pick the one with the best test macro-F1.

Copies the winning checkpoint to --out_dir/best/ and writes a summary across
all seeds. Use after running train.py with several --seed values.

Example:
    for s in 42 43 44 45; do
        python train.py --point --seed $s --output_dir runs/point-s$s
    done
    python pick_best.py --globs 'runs/point-s*' --out_dir runs/point-best
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path


def scan(globs: list[str]) -> list[dict]:
    candidates: list[dict] = []
    for pat in globs:
        for p in glob.glob(pat):
            rd = Path(p)
            mp = rd / "test_metrics.json"
            cfg = rd / "config.json"
            if not mp.exists() or not cfg.exists():
                continue
            met = json.loads(mp.read_text(encoding="utf-8"))
            c = json.loads(cfg.read_text(encoding="utf-8"))
            candidates.append({
                "run_dir": str(rd),
                "seed": c.get("seed"),
                "macro_f1": met.get("macro_f1", 0.0),
                "accuracy": met.get("accuracy", 0.0),
                "within_1_tier_accuracy": met.get("within_1_tier_accuracy"),
            })
    return candidates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--globs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--metric", default="macro_f1")
    args = ap.parse_args()

    cands = scan(args.globs)
    if not cands:
        print("[pick_best] no candidates found", file=sys.stderr)
        return 1
    cands.sort(key=lambda r: r[args.metric], reverse=True)

    print("\n=== SEED LEADERBOARD ===")
    for c in cands:
        print(f"seed={c['seed']:>5}  {args.metric}={c[args.metric]:.4f}  "
              f"acc={c['accuracy']:.4f}  w1t={c.get('within_1_tier_accuracy')}  {c['run_dir']}")

    best = cands[0]
    src = Path(best["run_dir"]) / "best"
    if not src.exists():
        print(f"[pick_best] WARNING: winner has no best/ dir at {src}")
        return 2

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dst = out / "best"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    (out / "leaderboard.json").write_text(json.dumps(cands, indent=2), encoding="utf-8")
    (out / "winner.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"\n[pick_best] winner: seed={best['seed']} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
