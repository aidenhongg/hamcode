"""Hyperparameter search via Optuna.

Each trial spawns a subprocess `python train.py --point ...` with sampled HPs
and parses the final dev macro-F1 out of the run's test_metrics.json. The user
supplies a separate `--test_metrics_field` if they want to optimize something
other than `macro_f1`.

Optuna's sampler persists in a sqlite DB at runs/<study>/optuna.db, so runs
are resumable across machines.

Example:
    python tune.py --point --n_trials 12 --study point-tune \\
        --data_dir data/processed --base_output_dir runs/tune/point
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import optuna
except ImportError:
    print("Optuna not installed. pip install optuna", file=sys.stderr)
    sys.exit(1)


def build_cmd(task: str, data_dir: str, output_dir: str, hp: dict) -> list[str]:
    cmd = [
        sys.executable, "train.py", f"--{task}",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--epochs", str(hp["epochs"]),
        "--lr", str(hp["lr"]),
        "--warmup_ratio", str(hp["warmup_ratio"]),
        "--weight_decay", str(hp["weight_decay"]),
        "--batch_size", str(hp["batch_size"]),
        "--grad_accum", str(hp["grad_accum"]),
        "--seed", str(hp["seed"]),
        "--label_smoothing", str(hp["label_smoothing"]),
    ]
    return cmd


def objective_factory(args):
    def objective(trial: optuna.Trial) -> float:
        hp = {
            "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 24]),
            "grad_accum": trial.suggest_categorical("grad_accum", [1, 2, 4]),
            "epochs": trial.suggest_int("epochs", 3, args.max_epochs),
            "seed": trial.suggest_int("seed", 1, 10_000),
        }
        out_dir = Path(args.base_output_dir) / f"trial-{trial.number:03d}"
        cmd = build_cmd(args.task, args.data_dir, str(out_dir), hp)
        print(f"[tune] trial {trial.number}: {hp}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[tune] trial {trial.number} FAILED:\n{proc.stderr[-2000:]}", flush=True)
            raise optuna.TrialPruned()
        met_path = out_dir / "test_metrics.json"
        if not met_path.exists():
            print(f"[tune] trial {trial.number}: no test_metrics.json", flush=True)
            raise optuna.TrialPruned()
        met = json.loads(met_path.read_text(encoding="utf-8"))
        score = float(met.get(args.test_metrics_field, 0.0))
        print(f"[tune] trial {trial.number}: {args.test_metrics_field}={score:.4f}", flush=True)
        trial.set_user_attr("run_dir", str(out_dir))
        return score
    return objective


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--point", dest="task", action="store_const", const="point")
    mode.add_argument("--pair", dest="task", action="store_const", const="pair")
    ap.add_argument("--n_trials", type=int, default=12)
    ap.add_argument("--max_epochs", type=int, default=6,
                    help="per-trial epoch cap during search (keep small for speed)")
    ap.add_argument("--study", default="tune")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--base_output_dir", default="runs/tune")
    ap.add_argument("--test_metrics_field", default="macro_f1")
    # Final run with best HPs (default ON — this is the "single-command" workflow)
    ap.add_argument("--no_final", action="store_true",
                    help="skip the final full training run with best HPs")
    ap.add_argument("--final_epochs", type=int, default=50,
                    help="epochs for the final extended run (patience stops earlier)")
    ap.add_argument("--final_patience", type=int, default=5,
                    help="patience for the final run")
    ap.add_argument("--final_output_dir", default="",
                    help="where to put the final run (default: <base_output_dir>/final)")
    ap.add_argument("--warm_start_from", default="",
                    help="pass-through to train.py for --pair warm-start")
    args = ap.parse_args()

    base = Path(args.base_output_dir); base.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{base}/optuna.db"
    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(objective_factory(args), n_trials=args.n_trials)

    print("\n=== BEST TRIAL ===")
    best = study.best_trial
    print(f"value = {best.value}")
    print("params = ")
    for k, v in best.params.items():
        print(f"  {k} = {v}")
    print(f"run_dir = {best.user_attrs.get('run_dir')}")
    (base / "best.json").write_text(
        json.dumps({
            "value": best.value,
            "params": best.params,
            "run_dir": best.user_attrs.get("run_dir"),
        }, indent=2),
        encoding="utf-8",
    )

    # --- Final extended training with best HPs ---
    if args.no_final:
        print("[tune] --no_final set; skipping final training run.")
        return 0

    final_out = Path(args.final_output_dir or (base / "final"))
    best_hp = dict(best.params)
    best_hp["epochs"] = args.final_epochs     # override search cap with a proper training run
    final_cmd = build_cmd(args.task, args.data_dir, str(final_out), best_hp) + [
        "--patience", str(args.final_patience),
    ]
    if args.task == "pair" and args.warm_start_from:
        final_cmd += ["--warm_start_from", args.warm_start_from]

    print("\n=== FINAL RUN (best HPs, extended epochs) ===")
    print("cmd:", " ".join(final_cmd), flush=True)
    proc = subprocess.run(final_cmd)
    if proc.returncode != 0:
        print(f"[tune] final run failed (returncode={proc.returncode})")
        return proc.returncode

    met_path = final_out / "test_metrics.json"
    if met_path.exists():
        met = json.loads(met_path.read_text(encoding="utf-8"))
        print("\n=== FINAL TEST METRICS ===")
        print(f"  macro_f1 = {met.get('macro_f1')}")
        print(f"  accuracy = {met.get('accuracy')}")
        within = met.get("within_1_tier_accuracy")
        if within is not None:
            print(f"  within_1_tier_accuracy = {within}")
        print(f"\nArtifacts written to: {final_out}")
        print(f"  best checkpoint:  {final_out}/best/")
        print(f"  plots:            {final_out}/*.png")
        print(f"  full test json:   {met_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
