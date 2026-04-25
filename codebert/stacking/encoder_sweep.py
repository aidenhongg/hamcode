"""Run a sweep over encoder finetuning recipes.

Each recipe runs an end-to-end encoder pipeline (optional activation cache
prewarm -> OOF pointwise -> CLS similarity -> head grid sweep) into its own
output directory under runs/heads/<recipe>/. After all recipes finish, the
best test macro-F1 per recipe is summarized in runs/heads/ENCODER_SUMMARY.md
and a winner is picked (highest test macro-F1, ties broken on lower
wallclock).

Usage:
    python -m stacking.encoder_sweep \
        --config stacking/configs/encoder_sweep.yaml \
        --data_dir data/processed \
        --out_root runs/heads

Resume-aware: a recipe whose runs/heads/<name>/SUMMARY.md already exists is
skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[encoder-sweep] $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (exit={proc.returncode}): {cmd[:3]}...")


def _resolve_path(p: str | Path, root: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else root / pp


def _summarize_recipe(out_dir: Path) -> dict:
    """Read the head sweep's SUMMARY.csv and return the best row by test_acc."""
    csv_path = out_dir / "SUMMARY.csv"
    if not csv_path.exists():
        return {"status": "no_summary"}
    import csv as csvmod
    rows = list(csvmod.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        return {"status": "empty"}
    def _f(r, k):
        try: return float(r.get(k) or "nan")
        except ValueError: return float("nan")
    best = max(rows, key=lambda r: _f(r, "test_macro_f1"))
    return {
        "status": "ok",
        "best_head": best.get("head"),
        "best_seed": best.get("seed"),
        "test_macro_f1": _f(best, "test_macro_f1"),
        "test_accuracy": _f(best, "test_accuracy"),
        "val_macro_f1": _f(best, "val_macro_f1"),
        "n_rows": len(rows),
    }


def _flag(name: str, value) -> list[str]:
    """Convert a bool/scalar to CLI flags. None and False emit nothing."""
    if value is None or value is False:
        return []
    if value is True:
        return [f"--{name}"]
    return [f"--{name}", str(value)]


def _run_recipe(
    recipe: dict,
    data_dir: Path,
    out_root: Path,
    extraction_root: Path,
    head_sweep_config: Path,
    force: bool,
    skip_data: bool,
) -> dict:
    name = recipe["name"]
    recipe_extraction = extraction_root / name
    recipe_out = out_root / name
    summary_path = recipe_out / "SUMMARY.md"

    if summary_path.exists() and not force:
        print(f"[encoder-sweep] {name}: SUMMARY.md present, skipping (use --force to re-run)",
              flush=True)
        return {"name": name, "skipped": True, "wallclock_s": 0.0,
                **_summarize_recipe(recipe_out)}

    print(f"[encoder-sweep] === recipe: {name} ===", flush=True)
    print(f"[encoder-sweep] note: {recipe.get('note', '')}", flush=True)
    t0 = time.time()

    if recipe.get("activation_cache"):
        if recipe.get("lora_freeze_depth", 0) <= 0:
            raise ValueError(
                f"recipe {name}: activation_cache requires lora_freeze_depth > 0"
            )
        cache_cmd = [
            sys.executable, "cache_activations.py",
            "--data_dir", str(data_dir),
            "--model_name", recipe.get("model_name", "microsoft/longcoder-base"),
            "--freeze_depth", str(recipe["lora_freeze_depth"]),
            "--max_seq_len", str(recipe["max_seq_len"]),
            "--bridge_stride", str(recipe["bridge_stride"]),
            "--batch_size", str(recipe.get("extract_batch", 4)),
        ]
        _run(cache_cmd)

    oof_cmd = [
        sys.executable, "-m", "stacking.features.oof_point",
        "--data_dir", str(data_dir),
        "--out_dir", str(recipe_extraction),
        "--n_folds", str(recipe.get("n_folds", 5)),
        "--epochs", str(recipe["epochs"]),
        "--batch_size", str(recipe["batch_size"]),
        "--grad_accum", str(recipe["grad_accum"]),
        "--lr", str(recipe["lr"]),
        "--num_workers", str(recipe["num_workers"]),
        "--max_seq_len", str(recipe["max_seq_len"]),
        "--bridge_stride", str(recipe["bridge_stride"]),
        "--eval_every_steps", str(recipe["eval_every_steps"]),
        "--patience", str(recipe["patience"]),
        "--extract_batch", str(recipe["extract_batch"]),
        "--resume",
    ]
    oof_cmd += _flag("bf16", recipe.get("bf16", True))
    oof_cmd += _flag("lora", recipe.get("lora", False))
    if recipe.get("lora"):
        oof_cmd += [
            "--lora_r", str(recipe.get("lora_r", 16)),
            "--lora_alpha", str(recipe.get("lora_alpha", 32)),
            "--lora_dropout", str(recipe.get("lora_dropout", 0.05)),
            "--lora_target_modules", recipe.get(
                "lora_target_modules", "query,value,query_global,value_global"
            ),
            "--lora_freeze_depth", str(recipe.get("lora_freeze_depth", 0)),
        ]
    oof_cmd += _flag("activation_cache", recipe.get("activation_cache", False))
    _run(oof_cmd)

    sem_cmd = [
        sys.executable, "-m", "stacking.features.semantic",
        "--in_splits", str(data_dir),
        "--extraction_dir", str(recipe_extraction),
    ]
    _run(sem_cmd)

    sweep_cmd = [
        sys.executable, "-m", "stacking.sweep",
        "--config", str(head_sweep_config),
        "--in_splits", str(data_dir),
        "--extraction_dir", str(recipe_extraction),
        "--out_dir", str(recipe_out),
    ]
    _run(sweep_cmd)

    wallclock = time.time() - t0
    summary = {"name": name, "skipped": False, "wallclock_s": wallclock,
               **_summarize_recipe(recipe_out)}
    return summary


def _write_encoder_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Encoder recipe sweep",
        "",
        "Each row is the best test row from `<recipe>/SUMMARY.csv`.",
        "",
        "| recipe | best head | seed | val F1 | test F1 | test acc | wallclock | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        wallclock_h = r.get("wallclock_s", 0.0) / 3600
        lines.append(
            f"| `{r['name']}` | {r.get('best_head','-')} | {r.get('best_seed','-')} | "
            f"{r.get('val_macro_f1','-'):.4f} | {r.get('test_macro_f1','-'):.4f} | "
            f"{r.get('test_accuracy','-'):.4f} | {wallclock_h:.2f} h | {r.get('status','-')} |"
            if isinstance(r.get("test_macro_f1"), float) else
            f"| `{r['name']}` | - | - | - | - | - | {wallclock_h:.2f} h | {r.get('status','-')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pick_winner(rows: list[dict]) -> dict | None:
    valid = [r for r in rows if isinstance(r.get("test_macro_f1"), float)
             and not (r["test_macro_f1"] != r["test_macro_f1"])]  # filter NaN
    if not valid:
        return None
    valid.sort(key=lambda r: (-r["test_macro_f1"], r.get("wallclock_s", 0.0)))
    return valid[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="stacking/configs/encoder_sweep.yaml")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_root", default="runs/heads")
    ap.add_argument("--extraction_root", default="runs/heads/extraction")
    ap.add_argument("--force", action="store_true",
                    help="Re-run recipes even if SUMMARY.md exists.")
    ap.add_argument("--only", default=None,
                    help="csv list of recipe names to run; default = all.")
    ap.add_argument("--skip-data", action="store_true",
                    help="Reserved for run_runpod.sh — no-op here, dataset built upstream.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    recipes = cfg["recipes"]
    head_sweep_cfg = Path(cfg.get("head_sweep", {}).get("config",
                                                          "stacking/configs/sweep.yaml"))

    if args.only:
        keep = set(s.strip() for s in args.only.split(",") if s.strip())
        recipes = [r for r in recipes if r["name"] in keep]

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    extraction_root = Path(args.extraction_root)

    rows = []
    for recipe in recipes:
        try:
            row = _run_recipe(
                recipe, data_dir, out_root, extraction_root, head_sweep_cfg,
                force=args.force, skip_data=args.skip_data,
            )
        except Exception as e:
            print(f"[encoder-sweep] recipe {recipe['name']} FAILED: {e}", flush=True)
            row = {"name": recipe["name"], "skipped": False,
                   "wallclock_s": 0.0, "status": f"failed: {e}"}
        rows.append(row)

    summary_path = out_root / "ENCODER_SUMMARY.md"
    _write_encoder_summary(rows, summary_path)
    print(f"[encoder-sweep] wrote {summary_path}", flush=True)

    winner = _pick_winner(rows)
    if winner is not None:
        winner_path = out_root / "ENCODER_WINNER.json"
        winner_path.write_text(json.dumps(winner, indent=2), encoding="utf-8")
        print(f"[encoder-sweep] winner: {winner['name']} test_macro_f1={winner['test_macro_f1']:.4f}",
              flush=True)
        print(f"[encoder-sweep] wrote {winner_path}", flush=True)
    else:
        print("[encoder-sweep] no recipe produced a valid summary; no winner picked",
              flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
