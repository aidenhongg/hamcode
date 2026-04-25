"""Optuna HP search over LoRA encoder hyperparameters.

Run after `stacking.encoder_sweep` picks a winning recipe (A vs B). Each
trial:
  1. Trains ONE pointwise LongCoder model (full train split, not OOF) at the
     trial's LoRA HPs.
  2. Reports the best val macro-F1 from that run as the Optuna objective.
  3. MedianPruner truncates trials whose mid-training val F1 is below the
     running median.

OOF is skipped during HP search (1× cost per trial instead of 6×). Once the
winning HPs are known, kick off a final encoder_sweep run with --only
<recipe> and the new HPs baked in to produce the stacking artifacts.

Usage:
    python -m stacking.lora_hp_search \
        --base_recipe lora-r16-top6-cached \
        --data_dir data/processed \
        --out_root runs/heads/lora_hp \
        --trials 16 --seeds 42

The `--base_recipe` argument keys into stacking/configs/encoder_sweep.yaml
to recover the recipe's structural fields (freeze_depth band when fixed,
target modules, etc.). HPs that the search controls override the recipe
defaults per trial.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError as e:
    raise RuntimeError("pip install optuna") from e


def _read_recipe(config_path: Path, name: str) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for r in cfg["recipes"]:
        if r["name"] == name:
            return r
    raise KeyError(f"recipe {name!r} not in {config_path}")


def _best_val_f1(metrics_jsonl: Path) -> float:
    if not metrics_jsonl.exists():
        return float("nan")
    best = float("-inf")
    with metrics_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("split") != "val":
                continue
            f1 = rec.get("macro_f1")
            if isinstance(f1, (int, float)) and f1 > best:
                best = float(f1)
    return best if best != float("-inf") else float("nan")


def _train_one(
    args: argparse.Namespace,
    recipe: dict,
    trial_hps: dict,
    trial_dir: Path,
) -> float:
    """Run a single train.py invocation with trial HPs; return best val macro-F1."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "train.py",
        "--data_dir", str(args.data_dir),
        "--output_dir", str(trial_dir),
        "--epochs", str(args.epochs_per_trial),
        "--batch_size", str(recipe["batch_size"]),
        "--grad_accum", str(recipe["grad_accum"]),
        "--lr", str(trial_hps["lr"]),
        "--num_workers", str(recipe["num_workers"]),
        "--max_seq_len", str(recipe["max_seq_len"]),
        "--bridge_stride", str(recipe["bridge_stride"]),
        "--eval_every_steps", str(recipe["eval_every_steps"]),
        "--patience", str(recipe["patience"]),
        "--seed", str(args.seed),
        "--lora",
        "--lora_r", str(trial_hps["lora_r"]),
        "--lora_alpha", str(trial_hps["lora_alpha"]),
        "--lora_dropout", str(trial_hps["lora_dropout"]),
        "--lora_target_modules", trial_hps["lora_target_modules"],
        "--lora_freeze_depth", str(trial_hps["lora_freeze_depth"]),
    ]
    cmd.append("--bf16" if recipe.get("bf16", True) else "--no_bf16")
    if trial_hps.get("activation_cache"):
        cmd.append("--activation_cache")

    print(f"[lora-hp] $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        # Treat as a failed trial; Optuna takes nan and prunes.
        return float("nan")
    return _best_val_f1(trial_dir / "metrics.jsonl")


def _build_objective(args: argparse.Namespace, recipe: dict, out_root: Path):
    base_freeze_depth = int(recipe.get("lora_freeze_depth", 0))
    activation_cache = bool(recipe.get("activation_cache", False))
    base_target_modules = recipe.get(
        "lora_target_modules", "query,value,query_global,value_global"
    )

    def objective(trial: "optuna.Trial") -> float:
        hps = {
            "lora_r": trial.suggest_categorical("lora_r", [8, 16, 32]),
            "lora_alpha": trial.suggest_categorical("lora_alpha", [16, 32, 64]),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.2),
            "lr": trial.suggest_float("lr", 3e-5, 5e-4, log=True),
            "lora_target_modules": base_target_modules,
            "lora_freeze_depth": base_freeze_depth,
            "activation_cache": activation_cache,
        }
        # If the winning recipe is the cached/top-band variant, also sweep depth.
        if base_freeze_depth > 0:
            hps["lora_freeze_depth"] = trial.suggest_categorical(
                "lora_freeze_depth", [3, 6, 9]
            )
            if hps["lora_freeze_depth"] != base_freeze_depth and activation_cache:
                # The activation cache is keyed on freeze_depth — switching
                # depth invalidates the cache. Disable for this trial; we
                # eat the wallclock hit but stay correct.
                hps["activation_cache"] = False

        trial_dir = out_root / "trials" / f"trial_{trial.number:04d}"
        f1 = _train_one(args, recipe, hps, trial_dir)
        if f1 != f1:  # NaN
            raise optuna.TrialPruned()
        # Persist trial's HPs + result so we can reconstruct outside Optuna.
        (trial_dir / "trial_summary.json").write_text(
            json.dumps({"hps": hps, "best_val_macro_f1": f1}, indent=2),
            encoding="utf-8",
        )
        return f1

    return objective


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="stacking/configs/encoder_sweep.yaml",
                    help="Encoder-sweep YAML providing recipe defaults.")
    ap.add_argument("--base_recipe", required=True,
                    help="Recipe name from --config to start the LoRA HP search from.")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_root", default="runs/heads/lora_hp")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--epochs_per_trial", type=int, default=10,
                    help="Lower than the full-recipe budget; HP comparison only.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--study_name", default=None,
                    help="Optuna study name (defaults to lora-<base_recipe>).")
    ap.add_argument("--storage", default=None,
                    help="Optuna SQLite URL (defaults to sqlite under out_root).")
    args = ap.parse_args()

    recipe = _read_recipe(Path(args.config), args.base_recipe)
    out_root = Path(args.out_root) / args.base_recipe
    out_root.mkdir(parents=True, exist_ok=True)

    study_name = args.study_name or f"lora-{args.base_recipe}"
    storage = args.storage or f"sqlite:///{out_root / 'study.db'}"
    study = optuna.create_study(
        study_name=study_name, storage=storage,
        sampler=TPESampler(seed=args.seed, multivariate=True),
        pruner=MedianPruner(n_startup_trials=4),
        direction="maximize", load_if_exists=True,
    )
    print(f"[lora-hp] base_recipe={args.base_recipe} trials={args.trials} "
          f"epochs_per_trial={args.epochs_per_trial}", flush=True)

    t0 = time.time()
    study.optimize(
        _build_objective(args, recipe, out_root),
        n_trials=args.trials, gc_after_trial=True,
        show_progress_bar=False,
    )
    wallclock = time.time() - t0

    best_params: dict[str, Any] = dict(study.best_params)
    best_value = float(study.best_value)
    summary = {
        "base_recipe": args.base_recipe,
        "study_name": study_name,
        "trials": args.trials,
        "best_val_macro_f1": best_value,
        "best_params": best_params,
        "wallclock_s": wallclock,
    }
    (out_root / "best_params.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
    print(f"[lora-hp] best val macro-F1 = {best_value:.4f}", flush=True)
    print(f"[lora-hp] best params = {best_params}", flush=True)
    print(f"[lora-hp] wrote {out_root / 'best_params.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
