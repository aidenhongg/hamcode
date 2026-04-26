"""Single-experiment CLI: train one head with one seed (variant is fixed at v1).

Usage:
    python -m stacking.train_head \
        --head xgb --seed 42 \
        --in_splits data/processed \
        --extraction_dir runs/heads/extraction \
        --out_dir runs/heads/xgb-v1-s42

Writes to out_dir:
    config.json           -- resolved config
    metrics.jsonl         -- per-eval or per-epoch metrics (as applicable)
    test_metrics.json     -- final test metrics + confusion matrix +
                              per_language breakdown
    predictions.parquet   -- per-pair predictions (incl. language) for
                              error analysis
    feature_importance.json (tree heads only)
    confusion_matrix.png  -- saved if matplotlib available
    scaler.joblib + schema.json + head artifacts

Per-language test perf is also printed at the end of the run, sorted by
support (largest language first).
"""

from __future__ import annotations

# Load torch FIRST so its DLLs are found before other native libs
# (xgboost, lightgbm) might perturb the Windows DLL search path.
import torch  # noqa: F401

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stacking import dataset as ds
from stacking.heads import get_head
from stacking.heads.base import HeadRegistry, compute_class_weight
from stacking import metrics as M


def _maybe_plot_confusion(cm: list[list[int]], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    cm_np = np.asarray(cm, dtype=np.float32)
    row_sums = cm_np.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
    norm = cm_np / row_sums
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["same", "A_faster"])
    ax.set_yticklabels(["same", "A_faster"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm_np[i,j])}\n{norm[i,j]:.2f}",
                     ha="center", va="center", fontsize=10,
                     color="black" if norm[i,j] < 0.5 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _maybe_plot_roc(y_true: np.ndarray, probs_pos: np.ndarray, out_path: Path) -> None:
    if len(set(y_true.tolist())) < 2:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
    except Exception:
        return
    fpr, tpr, _ = roc_curve(y_true, probs_pos)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(loc="lower right")
    ax.set_title("ROC")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def run(
    head_name: str,
    seed: int,
    in_splits: Path,
    extraction_dir: Path,
    out_dir: Path,
    class_weight_mode: str = "auto",
    head_hp: dict | None = None,
    variant: str = "v1",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build feature matrices with shared scaler (fit on train only)
    train, val, test = ds.build_all_splits(
        variant=variant, in_splits=in_splits,
        extraction_dir=extraction_dir, out_dir=out_dir,
    )

    # Class weight
    cw = None
    if class_weight_mode == "auto":
        cw = compute_class_weight(train.y)
    elif class_weight_mode == "none":
        cw = None

    # Build & fit head
    hp = head_hp or {}
    head = get_head(head_name, seed=seed, **hp)
    print(f"[train_head] fitting {head_name} (variant={variant} seed={seed}) "
          f"on X={train.X.shape} cw={cw}", flush=True)
    fit_summary = head.fit(train.X, train.y, val.X, val.y, class_weight=cw)

    # Eval
    val_pred = head.predict(val.X)
    val_prob = head.predict_proba(val.X)[:, 1]
    val_met = M.compute_all(val.y, val_pred, val_prob)
    val_met["split"] = "val"

    test_pred = head.predict(test.X)
    test_prob = head.predict_proba(test.X)[:, 1]

    test_met = M.compute_all(test.y, test_pred, test_prob)
    test_met["split"] = "test"
    if test.languages:
        test_met["per_language"] = M.compute_per_language(
            test.y, test_pred, test_prob, test.languages,
        )

    # Config + metrics
    config = {
        "head": head_name,
        "variant": variant,
        "variant_desc": ds.VARIANT_DESCRIPTIONS[variant],
        "seed": seed,
        "class_weight_mode": class_weight_mode,
        "class_weight": cw,
        "hp": head.hp,
        "fit_summary": _jsonable(fit_summary),
        "train_rows": int(train.X.shape[0]),
        "val_rows": int(val.X.shape[0]),
        "test_rows": int(test.X.shape[0]),
        "n_features": int(train.X.shape[1]),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=_jsonable),
                                          encoding="utf-8")

    with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable({"split": "val", **val_met})) + "\n")
        f.write(json.dumps(_jsonable({"split": "test", **test_met})) + "\n")
    (out_dir / "test_metrics.json").write_text(
        json.dumps(_jsonable(test_met), indent=2), encoding="utf-8",
    )

    # Predictions parquet
    pred_cols: dict[str, Any] = {
        "pair_id": test.pair_ids,
        "label_true": test.y.astype(np.int64),
        "label_pred": test_pred.astype(np.int64),
        "prob_same": 1.0 - test_prob.astype(np.float32),
        "prob_A_faster": test_prob.astype(np.float32),
    }
    if test.languages:
        pred_cols["language"] = list(test.languages)
    pred_table = pa.table(pred_cols)
    pq.write_table(pred_table, out_dir / "predictions.parquet", compression="zstd")

    # Feature importance
    fi = head.feature_importance()
    if fi is not None:
        # remap "f0" style XGB keys to column names if possible
        col_names = train.columns
        remapped = {}
        for k, v in fi.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if 0 <= idx < len(col_names):
                    remapped[col_names[idx]] = v
                    continue
            if k.startswith("feat_") and k[5:].isdigit():
                idx = int(k[5:])
                if 0 <= idx < len(col_names):
                    remapped[col_names[idx]] = v
                    continue
            remapped[k] = v
        (out_dir / "feature_importance.json").write_text(
            json.dumps(dict(sorted(remapped.items(), key=lambda kv: -kv[1])), indent=2),
            encoding="utf-8",
        )

    # Plots
    _maybe_plot_confusion(test_met["confusion_matrix"], out_dir / "confusion_matrix.png")
    _maybe_plot_roc(test.y, test_prob, out_dir / "roc_curve.png")

    # Save head artifacts
    head.save(out_dir / "head")

    print(f"[train_head] test acc={test_met['accuracy']:.4f} "
          f"macro_f1={test_met['macro_f1']:.4f} "
          f"roc_auc={test_met['roc_auc']:.4f}", flush=True)

    per_lang = test_met.get("per_language")
    if per_lang:
        _print_per_language_table(per_lang)

    return test_met


def _print_per_language_table(per_lang: dict[str, dict]) -> None:
    rows = sorted(
        per_lang.items(),
        key=lambda kv: (-kv[1].get("n", 0), kv[0]),
    )
    print("[train_head] per-language test perf:", flush=True)
    print(f"  {'language':<12} {'n':>6} {'acc':>7} {'macro_f1':>9} "
          f"{'roc_auc':>8} {'pr_auc':>7}", flush=True)
    for lang, m in rows:
        def _fmt(x: float) -> str:
            return "  nan  " if x != x else f"{x:.4f}"
        print(f"  {lang:<12} {m['n']:>6d} {_fmt(m['accuracy']):>7} "
              f"{_fmt(m['macro_f1']):>9} {_fmt(m['roc_auc']):>8} "
              f"{_fmt(m['pr_auc']):>7}", flush=True)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head", required=True, choices=HeadRegistry.all())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--in_splits", default="data/processed")
    ap.add_argument("--extraction_dir", default="runs/heads/extraction")
    ap.add_argument("--out_dir", default=None,
                    help="Default: runs/heads/{head}-v1-s{seed}")
    ap.add_argument("--class_weight", default="auto", choices=["auto", "none"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"runs/heads/{args.head}-v1-s{args.seed}"
    )
    run(
        head_name=args.head,
        seed=args.seed,
        in_splits=Path(args.in_splits),
        extraction_dir=Path(args.extraction_dir),
        out_dir=out_dir,
        class_weight_mode=args.class_weight,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
