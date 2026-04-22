"""Single-experiment CLI: train one head on one variant with one seed.

Usage:
    python -m stacking.train_head \
        --head xgb --variant v3 --seed 42 \
        --in_splits data/processed \
        --extraction_dir runs/heads/extraction \
        --out_dir runs/heads/xgb-v3-s42

Writes to out_dir:
    config.json           -- resolved config
    metrics.jsonl         -- per-eval or per-epoch metrics (as applicable)
    test_metrics.json     -- final test metrics + confusion matrix + McNemar
    predictions.parquet   -- per-pair predictions (for error analysis)
    feature_importance.json (tree heads only)
    confusion_matrix.png  -- saved if matplotlib available
    scaler.joblib + schema.json + head artifacts
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


# Map pair-BERT ternary label to our binary scheme, for baseline comparison.
# In the filtered subset, BERT can only output A_faster or same to be correct;
# if it outputs B_faster we treat that as "same" (wrong direction by definition
# in this subset — the most meaningful head baseline).
_TERNARY_TO_BINARY = {"A_faster": 1, "same": 0, "B_faster": 0}


def _bert_baseline_preds(pair_tbl, pair_logits_tbl, pair_ids: list[str]) -> np.ndarray:
    """BERT pairwise model's argmax prediction, mapped to binary (y = same/A_faster)."""
    cols = [c for c in ["pair_logit_0", "pair_logit_1", "pair_logit_2"]
            if c in pair_logits_tbl.schema.names]
    if len(cols) != 3:
        return np.zeros(len(pair_ids), dtype=np.int64)
    id_to_row = {id_: i for i, id_ in enumerate(pair_logits_tbl.column("pair_id").to_pylist())}
    raw = np.stack(
        [np.asarray(pair_logits_tbl.column(c).to_pylist(), dtype=np.float32) for c in cols],
        axis=1,
    )
    # Pair labels order in training (see common/labels.py): ("A_faster", "same", "B_faster")
    label_order = ("A_faster", "same", "B_faster")
    out = np.zeros(len(pair_ids), dtype=np.int64)
    for i, pid in enumerate(pair_ids):
        r = id_to_row.get(pid)
        if r is None:
            out[i] = 0
            continue
        argmax = int(np.argmax(raw[r]))
        out[i] = _TERNARY_TO_BINARY[label_order[argmax]]
    return out


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
    variant: str,
    seed: int,
    in_splits: Path,
    extraction_dir: Path,
    out_dir: Path,
    class_weight_mode: str = "auto",
    head_hp: dict | None = None,
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

    # BERT baseline on the same filtered test subset (if pair logits exist)
    pair_logit_pq = extraction_dir / "pair_logits_test.parquet"
    bert_pred = None
    if pair_logit_pq.exists():
        pair_tbl = pq.read_table(in_splits / "pair_test.parquet")
        pair_tbl = ds.filter_b_ge_a(pair_tbl)
        pl_tbl = pq.read_table(pair_logit_pq)
        bert_pred = _bert_baseline_preds(pair_tbl, pl_tbl, test.pair_ids)

    test_met = M.compute_all(test.y, test_pred, test_prob, bert_pair_pred=bert_pred)
    test_met["split"] = "test"

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
    pred_table = pa.table({
        "pair_id": test.pair_ids,
        "label_true": test.y.astype(np.int64),
        "label_pred": test_pred.astype(np.int64),
        "prob_same": 1.0 - test_prob.astype(np.float32),
        "prob_A_faster": test_prob.astype(np.float32),
        **({"bert_pair_pred": bert_pred.astype(np.int64)} if bert_pred is not None else {}),
    })
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
    if "bert_pairwise_baseline_comparison" in test_met:
        b = test_met["bert_pairwise_baseline_comparison"]
        print(f"[train_head] vs BERT pairwise baseline: "
              f"delta_acc={b['delta']:+.4f} mcnemar_p={b['mcnemar_p']:.4f}", flush=True)

    return test_met


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
    ap.add_argument("--variant", required=True, choices=["v1", "v2", "v3"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--in_splits", default="data/processed")
    ap.add_argument("--extraction_dir", default="runs/heads/extraction")
    ap.add_argument("--out_dir", default=None,
                    help="Default: runs/heads/{head}-{variant}-s{seed}")
    ap.add_argument("--class_weight", default="auto", choices=["auto", "none"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"runs/heads/{args.head}-{args.variant}-s{args.seed}"
    )
    run(
        head_name=args.head,
        variant=args.variant,
        seed=args.seed,
        in_splits=Path(args.in_splits),
        extraction_dir=Path(args.extraction_dir),
        out_dir=out_dir,
        class_weight_mode=args.class_weight,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
