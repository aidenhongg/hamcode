"""Cartesian sweep: (head x variant x seed) with fail-soft aggregation.

Writes per-experiment directories to <out_dir>/<head>-<variant>-s<seed>/
plus aggregate SUMMARY.md and SUMMARY.csv.

Fail-soft: a single experiment failure is logged to <out_dir>/_failures.jsonl
and the sweep continues.

Usage:
    python -m stacking.sweep \
        --config stacking/configs/sweep.yaml \
        --in_splits data/processed \
        --extraction_dir runs/heads/extraction \
        --out_dir runs/heads

    # Smoke: 1 head x 1 variant x 1 seed on whatever data is present
    python -m stacking.sweep --smoke
"""

from __future__ import annotations

# Load torch FIRST for Windows DLL search path reasons; see train_head.py.
import torch  # noqa: F401

import argparse
import itertools
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stacking import train_head as th


@dataclass
class SweepConfig:
    heads: list[str]
    variants: list[str]
    seeds: list[int]
    class_weight: str = "auto"

    @classmethod
    def load(cls, path: Path) -> "SweepConfig":
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            heads=list(cfg["heads"]),
            variants=list(cfg["variants"]),
            seeds=list(cfg["seeds"]),
            class_weight=cfg.get("class_weight", "auto"),
        )


def _run_one(head: str, variant: str, seed: int, in_splits: Path,
             extraction_dir: Path, out_dir: Path, class_weight: str,
             failures: Path) -> dict | None:
    exp_dir = out_dir / f"{head}-{variant}-s{seed}"
    try:
        met = th.run(
            head_name=head, variant=variant, seed=seed,
            in_splits=in_splits, extraction_dir=extraction_dir,
            out_dir=exp_dir, class_weight_mode=class_weight,
        )
        row = {
            "head": head, "variant": variant, "seed": seed,
            "test_acc": met["accuracy"],
            "balanced_acc": met["balanced_accuracy"],
            "macro_f1": met["macro_f1"],
            "roc_auc": met["roc_auc"],
            "brier": met["brier_score"],
            "ece": met["ece"],
            "per_class_same_f1": met["per_class"]["same"]["f1"],
            "per_class_A_faster_f1": met["per_class"]["A_faster"]["f1"],
            "dir": str(exp_dir),
        }
        if "bert_pairwise_baseline_comparison" in met:
            b = met["bert_pairwise_baseline_comparison"]
            row["vs_bert_delta"] = b["delta"]
            row["mcnemar_p"] = b["mcnemar_p"]
        return row
    except Exception as e:  # fail-soft
        msg = {
            "head": head, "variant": variant, "seed": seed,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        with failures.open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg) + "\n")
        print(f"[sweep] FAIL {head}/{variant}/s{seed}: {e}", flush=True)
        return None


def _write_summary(rows: list[dict], out_dir: Path) -> None:
    # Sort by test_acc desc as the primary ranking
    rows_sorted = sorted(rows, key=lambda r: -r.get("test_acc", 0.0))

    # CSV
    import csv
    csv_path = out_dir / "SUMMARY.csv"
    if rows_sorted:
        keys = list(rows_sorted[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows_sorted:
                w.writerow(r)

    # Markdown
    md_path = out_dir / "SUMMARY.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Stacking Sweep Results\n\n")
        f.write(f"Total experiments: {len(rows)}\n\n")
        if not rows_sorted:
            f.write("(no successful runs)\n"); return

        cols = ["head", "variant", "seed", "test_acc", "macro_f1", "roc_auc",
                "brier", "ece", "vs_bert_delta", "mcnemar_p"]
        # header
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join("---" for _ in cols) + "|\n")
        for r in rows_sorted:
            def _fmt(k):
                v = r.get(k)
                if v is None:
                    return "—"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)
            f.write("| " + " | ".join(_fmt(c) for c in cols) + " |\n")

        # Best per (head, variant) pivot
        f.write("\n## Best seed per (head, variant)\n\n")
        best: dict[tuple[str, str], dict] = {}
        for r in rows_sorted:
            key = (r["head"], r["variant"])
            if key not in best or r["test_acc"] > best[key]["test_acc"]:
                best[key] = r
        pcols = ["head", "variant", "best_seed", "test_acc", "macro_f1",
                 "roc_auc", "vs_bert_delta", "mcnemar_p"]
        f.write("| " + " | ".join(pcols) + " |\n")
        f.write("|" + "|".join("---" for _ in pcols) + "|\n")
        for (head, variant), r in sorted(best.items()):
            row = {
                "head": head, "variant": variant,
                "best_seed": r["seed"],
                "test_acc": r["test_acc"],
                "macro_f1": r["macro_f1"],
                "roc_auc": r.get("roc_auc"),
                "vs_bert_delta": r.get("vs_bert_delta"),
                "mcnemar_p": r.get("mcnemar_p"),
            }
            def _fmt(k):
                v = row.get(k)
                if v is None:
                    return "—"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)
            f.write("| " + " | ".join(_fmt(c) for c in pcols) + " |\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="stacking/configs/sweep.yaml")
    ap.add_argument("--in_splits", default="data/processed")
    ap.add_argument("--extraction_dir", default="runs/heads/extraction")
    ap.add_argument("--out_dir", default="runs/heads")
    ap.add_argument("--smoke", action="store_true",
                    help="run xgb/v3/seed 42 only (quick E2E check)")
    ap.add_argument("--head", default=None, help="override heads from config")
    ap.add_argument("--variant", default=None, help="override variants from config")
    ap.add_argument("--seed", type=int, default=None, help="override seeds from config")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = out_dir / "_failures.jsonl"
    if failures.exists():
        failures.unlink()

    if args.smoke:
        cfg = SweepConfig(heads=["xgb"], variants=["v3"], seeds=[42])
    else:
        cfg = SweepConfig.load(Path(args.config))

    if args.head:
        cfg.heads = [args.head]
    if args.variant:
        cfg.variants = [args.variant]
    if args.seed is not None:
        cfg.seeds = [args.seed]

    all_combos = list(itertools.product(cfg.heads, cfg.variants, cfg.seeds))
    print(f"[sweep] running {len(all_combos)} experiments", flush=True)

    rows: list[dict] = []
    for head, variant, seed in all_combos:
        print(f"\n=== {head} / {variant} / s{seed} ===", flush=True)
        row = _run_one(head, variant, seed,
                        Path(args.in_splits), Path(args.extraction_dir),
                        out_dir, cfg.class_weight, failures)
        if row is not None:
            rows.append(row)
            # Incremental summary so we can peek mid-sweep
            _write_summary(rows, out_dir)

    _write_summary(rows, out_dir)
    print(f"\n[sweep] done. {len(rows)}/{len(all_combos)} successful.", flush=True)
    print(f"[sweep] summary: {out_dir / 'SUMMARY.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
