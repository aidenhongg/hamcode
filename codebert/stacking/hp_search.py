"""Optuna HP search for the top heads (xgb, lgbm, mlp, stacked).

Per (head, variant) cell:
  1. Build train/val/test feature matrices once with the shared scaler.
  2. Run Optuna with TPE + MedianPruner over head-specific search space,
     objective = val macro-F1, with class_weight='auto'.
  3. Take best HPs, retrain on train with same val, evaluate on test with
     N_SEEDS different seeds; report mean ± std.
  4. Persist:
        runs/heads/hp/<head>-<variant>/
          study.db                 # SQLite Optuna storage, resumable
          best_params.json         # winning HPs
          trials.jsonl             # every trial (params + val score)
          seed_results.jsonl       # final seed-variance run on test
          summary.json             # headline numbers
  5. Aggregate to runs/heads/hp/HP_SUMMARY.md + HP_SUMMARY.csv.

Runs sequentially. Variants default to v1,v3 (v2 consistently loses —
skip unless you specifically want to include it via --variants v1,v2,v3).

CLI:
    python -m stacking.hp_search \
        --heads xgb,lgbm,mlp,stacked \
        --variants v1,v3 \
        --trials 40 --seeds 42,43,44 \
        --in_splits data/processed \
        --extraction_dir runs/heads/extraction \
        --out_root runs/heads/hp
"""

from __future__ import annotations

# Load torch first to dodge the Windows DLL-order issue with xgboost/lightgbm.
import torch  # noqa: F401

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError as e:
    raise RuntimeError("pip install optuna") from e

from stacking import dataset as ds
from stacking import metrics as M
from stacking.heads import get_head
from stacking.heads.base import compute_class_weight


# -----------------------------------------------------------------------------
# Per-head search spaces. Each returns a dict of HPs sampled for one trial.
# -----------------------------------------------------------------------------

def _space_xgb(trial: "optuna.Trial", seed: int) -> dict:
    return dict(
        seed=seed,
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
        min_child_weight=trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
    )


def _space_lgbm(trial: "optuna.Trial", seed: int) -> dict:
    return dict(
        seed=seed,
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        max_depth=trial.suggest_int("max_depth", -1, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
    )


def _space_mlp(trial: "optuna.Trial", seed: int, device: str | None = None) -> dict:
    hp = dict(
        seed=seed,
        hidden_layers=trial.suggest_int("hidden_layers", 1, 4),
        hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 192, 256, 384, 512]),
        activation=trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        layer_norm=trial.suggest_categorical("layer_norm", [False, True]),
        optimizer=trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        epochs=40,
        patience=5,
        grad_clip=1.0,
    )
    if device is not None:
        hp["device"] = device   # only pin if caller explicitly requested
    return hp


def _space_stacked(trial: "optuna.Trial", seed: int) -> dict:
    # Use binary flags to include each base head. Require at least 2 bases
    # (otherwise stacking degenerates to a wrapper around one head).
    candidates = ["xgb", "lgbm", "mlp", "logreg", "rf"]
    include = {name: trial.suggest_categorical(f"use_{name}", [True, False])
               for name in candidates}
    bases = [n for n in candidates if include[n]]
    if len(bases) < 2:
        # Inflate with cheapest sensible default to make the trial valid.
        for extra in ("xgb", "lgbm", "logreg"):
            if extra not in bases:
                bases.append(extra)
            if len(bases) >= 2:
                break
    meta = trial.suggest_categorical("meta", ["logreg", "mlp", "xgb"])
    return dict(
        seed=seed,
        bases=bases,
        meta=meta,
        # Keep base HPs at defaults — stacked HP search is over
        # composition, not individual base tuning.
        base_hp=None,
        meta_hp=None,
    )


# Spaces take (trial, seed) by default; _space_mlp accepts an optional device
# kwarg. _dispatch_space normalises the call so the outer loop doesn't care.
_SPACES: dict[str, Callable[..., dict]] = {
    "xgb": _space_xgb,
    "lgbm": _space_lgbm,
    "mlp": _space_mlp,
    "stacked": _space_stacked,
}


def _dispatch_space(head_name: str, trial, seed: int, mlp_device: str | None):
    fn = _SPACES[head_name]
    if head_name == "mlp":
        return fn(trial, seed, device=mlp_device)
    return fn(trial, seed)


# -----------------------------------------------------------------------------
# One HP search for a given (head, variant)
# -----------------------------------------------------------------------------

def _fit_and_score(head_name: str, hp: dict, train, val, test, class_weight):
    head = get_head(head_name, **hp)
    head.fit(train.X, train.y, val.X, val.y, class_weight=class_weight)
    val_pred = head.predict(val.X)
    val_prob = head.predict_proba(val.X)[:, 1]
    val_met = M.compute_all(val.y, val_pred, val_prob)
    test_pred = head.predict(test.X)
    test_prob = head.predict_proba(test.X)[:, 1]
    test_met = M.compute_all(test.y, test_pred, test_prob)
    return val_met, test_met, head


def run_hp_search(
    head_name: str,
    variant: str,
    trials: int,
    seeds: list[int],
    search_seed: int,
    in_splits: Path,
    extraction_dir: Path,
    out_dir: Path,
    class_weight_mode: str = "auto",
    timeout_seconds: int | None = None,
    mlp_device: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build features once — feature matrix + scaler are identical across trials.
    train, val, test = ds.build_all_splits(
        variant=variant, in_splits=in_splits,
        extraction_dir=extraction_dir, out_dir=out_dir,
    )
    cw = compute_class_weight(train.y) if class_weight_mode == "auto" else None

    trials_log = (out_dir / "trials.jsonl").open("a", encoding="utf-8")

    def objective(trial: "optuna.Trial") -> float:
        hp = _dispatch_space(head_name, trial, seed=search_seed, mlp_device=mlp_device)
        try:
            val_met, _test_met, _ = _fit_and_score(
                head_name, hp, train, val, test, class_weight=cw,
            )
        except Exception as e:
            trials_log.write(json.dumps({
                "trial": trial.number, "status": "error",
                "params": dict(trial.params), "error": str(e),
            }) + "\n")
            trials_log.flush()
            raise
        score = float(val_met["macro_f1"])
        trials_log.write(json.dumps({
            "trial": trial.number, "status": "ok",
            "params": dict(trial.params),
            "val_macro_f1": score,
            "val_accuracy": float(val_met["accuracy"]),
            "val_roc_auc": float(val_met["roc_auc"]),
        }) + "\n")
        trials_log.flush()
        return score

    storage = f"sqlite:///{out_dir / 'study.db'}"
    study = optuna.create_study(
        study_name=f"{head_name}-{variant}",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=TPESampler(seed=search_seed, multivariate=True),
        pruner=MedianPruner(n_warmup_steps=1),
    )

    t0 = time.time()
    study.optimize(objective, n_trials=trials, timeout=timeout_seconds)
    trials_log.close()

    best_params = dict(study.best_trial.params)
    (out_dir / "best_params.json").write_text(
        json.dumps({
            "head": head_name,
            "variant": variant,
            "best_val_macro_f1": study.best_value,
            "params": best_params,
            "n_trials_total": len(study.trials),
            "n_trials_complete": sum(1 for t in study.trials
                                      if t.state == optuna.trial.TrialState.COMPLETE),
            "search_seconds": time.time() - t0,
        }, indent=2),
        encoding="utf-8",
    )

    # ------------- seed variance on test -------------
    seed_results: list[dict] = []
    for s in seeds:
        # Re-construct the HP dict with this seed substituted.
        hp = {**space_hp_from_best(head_name, best_params, seed=s,
                                      mlp_device=mlp_device)}
        try:
            val_met, test_met, head = _fit_and_score(
                head_name, hp, train, val, test, class_weight=cw,
            )
        except Exception as e:
            seed_results.append({"seed": s, "status": "error", "error": str(e)})
            continue
        seed_results.append({
            "seed": s,
            "status": "ok",
            "val_accuracy": float(val_met["accuracy"]),
            "val_macro_f1": float(val_met["macro_f1"]),
            "test_accuracy": float(test_met["accuracy"]),
            "test_macro_f1": float(test_met["macro_f1"]),
            "test_roc_auc": float(test_met["roc_auc"]),
            "test_brier": float(test_met["brier_score"]),
            "test_ece": float(test_met["ece"]),
        })

    with (out_dir / "seed_results.jsonl").open("w", encoding="utf-8") as f:
        for r in seed_results:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in seed_results if r.get("status") == "ok"]
    if ok:
        accs = np.array([r["test_accuracy"] for r in ok])
        f1s = np.array([r["test_macro_f1"] for r in ok])
        summary = {
            "head": head_name,
            "variant": variant,
            "best_val_macro_f1": study.best_value,
            "test_acc_mean": float(accs.mean()),
            "test_acc_std": float(accs.std()),
            "test_macro_f1_mean": float(f1s.mean()),
            "test_macro_f1_std": float(f1s.std()),
            "n_seeds": len(ok),
            "best_params": best_params,
            "search_seconds": time.time() - t0,
        }
    else:
        summary = {
            "head": head_name, "variant": variant,
            "error": "all seed re-runs failed",
            "seed_results": seed_results,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def space_hp_from_best(
    head_name: str, best_params: dict, seed: int,
    mlp_device: str | None = None,
) -> dict:
    """Rebuild a fit-ready HP dict from Optuna best_params + a chosen seed.

    Optuna's best_params is the raw suggest output (str/float/int). Most
    map 1:1 to head constructor args. Special cases:
      - MLP: add the non-searched fixed fields (epochs/patience/grad_clip).
             Device is left to MLPHead's default (CUDA if available, else CPU)
             unless the caller passes an explicit override.
      - Stacked: expand use_<name> flags back into a bases list
    """
    if head_name in ("xgb", "lgbm"):
        return {**best_params, "seed": seed}
    if head_name == "mlp":
        hp = {
            **best_params,
            "seed": seed,
            "epochs": 40, "patience": 5, "grad_clip": 1.0,
        }
        if mlp_device is not None:
            hp["device"] = mlp_device
        return hp
    if head_name == "stacked":
        candidates = ["xgb", "lgbm", "mlp", "logreg", "rf"]
        bases = [n for n in candidates if best_params.get(f"use_{n}", False)]
        if len(bases) < 2:
            for extra in ("xgb", "lgbm", "logreg"):
                if extra not in bases:
                    bases.append(extra)
                if len(bases) >= 2:
                    break
        return {"seed": seed, "bases": bases, "meta": best_params["meta"],
                "base_hp": None, "meta_hp": None}
    raise ValueError(head_name)


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def aggregate(out_root: Path) -> None:
    rows: list[dict] = []
    for sub in sorted(out_root.glob("*/")):
        sp = sub / "summary.json"
        if not sp.exists():
            continue
        try:
            rows.append(json.loads(sp.read_text(encoding="utf-8")))
        except Exception:
            continue
    # Sort by test_macro_f1_mean desc, skipping errored rows
    def _key(r):
        return (-r.get("test_macro_f1_mean", -1.0),
                -r.get("test_acc_mean", -1.0))
    rows.sort(key=_key)

    import csv
    with (out_root / "HP_SUMMARY.csv").open("w", encoding="utf-8", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            for r in rows:
                w.writerow({k: _scalar(v) for k, v in r.items()})

    lines = ["# HP Search Results\n", f"Total (head, variant) cells: {len(rows)}\n"]
    lines.append("| head | variant | test_acc (mean ± std) | test_macro_f1 (mean ± std) | best_val_macro_f1 | seeds | search_s |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['head']} | {r['variant']} | ERROR | ERROR | — | 0 | — |")
            continue
        acc = f"{r['test_acc_mean']:.4f} ± {r['test_acc_std']:.4f}"
        f1 = f"{r['test_macro_f1_mean']:.4f} ± {r['test_macro_f1_std']:.4f}"
        lines.append(
            f"| {r['head']} | {r['variant']} | {acc} | {f1} | "
            f"{r['best_val_macro_f1']:.4f} | {r['n_seeds']} | "
            f"{r.get('search_seconds', 0):.0f} |"
        )
    (out_root / "HP_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scalar(v):
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, default=str)
    return v


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_seeds(s: str) -> list[int]:
    return [int(x) for x in _parse_list(s)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heads", default="xgb,lgbm,mlp,stacked",
                    help="Comma-separated subset of xgb,lgbm,mlp,stacked")
    ap.add_argument("--variants", default="v1,v3",
                    help="Comma-separated subset of v1,v2,v3")
    ap.add_argument("--trials", type=int, default=40,
                    help="Optuna trials per (head, variant) cell")
    ap.add_argument("--seeds", default="42,43,44",
                    help="Seeds for final test-set variance run")
    ap.add_argument("--search_seed", type=int, default=42,
                    help="Seed for the Optuna TPE sampler itself")
    ap.add_argument("--timeout_seconds", type=int, default=0,
                    help="Per-cell time limit (0 = no timeout)")
    ap.add_argument("--in_splits", default="data/processed")
    ap.add_argument("--extraction_dir", default="runs/heads/extraction")
    ap.add_argument("--out_root", default="runs/heads/hp")
    ap.add_argument("--class_weight", default="auto", choices=["auto", "none"])
    ap.add_argument("--mlp_device", default=None,
                    help="Force MLP training device ('cpu', 'cuda', 'cuda:0'). "
                         "Default: MLPHead auto-picks CUDA if available.")
    args = ap.parse_args()

    heads = _parse_list(args.heads)
    variants = _parse_list(args.variants)
    seeds = _parse_seeds(args.seeds)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    failures: list[dict] = []
    for h in heads:
        if h not in _SPACES:
            print(f"[hp] unknown head {h!r}; skipping", flush=True)
            continue
        for v in variants:
            cell_dir = out_root / f"{h}-{v}"
            print(f"\n=== HP search: {h} / {v} "
                  f"({args.trials} trials, seeds={seeds}) ===", flush=True)
            try:
                run_hp_search(
                    head_name=h, variant=v,
                    trials=args.trials,
                    seeds=seeds,
                    search_seed=args.search_seed,
                    in_splits=Path(args.in_splits),
                    extraction_dir=Path(args.extraction_dir),
                    out_dir=cell_dir,
                    class_weight_mode=args.class_weight,
                    timeout_seconds=args.timeout_seconds or None,
                    mlp_device=args.mlp_device,
                )
            except Exception as e:
                print(f"[hp] FAILED {h}/{v}: {e}", flush=True)
                failures.append({
                    "head": h, "variant": v,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

    if failures:
        (out_root / "_failures.jsonl").write_text(
            "\n".join(json.dumps(f) for f in failures) + "\n",
            encoding="utf-8",
        )

    aggregate(out_root)
    print(f"\n[hp] done. summary: {out_root / 'HP_SUMMARY.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
