"""Plot training curves and confusion matrix for a completed run.

Usage:
    python plot_metrics.py --run_dir runs/point-20260421-180000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.labels import PAIR_LABELS, POINT_LABELS


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def plot_eval_curves(run_dir: Path) -> None:
    import matplotlib.pyplot as plt
    records = _read_jsonl(run_dir / "metrics.jsonl")
    if not records:
        print(f"[plot] no metrics.jsonl at {run_dir}")
        return
    by_split = defaultdict(list)
    for r in records:
        by_split[r.get("split", "val")].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for sp, recs in by_split.items():
        xs = [r["step"] for r in recs]
        ys = [r.get("accuracy", 0.0) for r in recs]
        ax.plot(xs, ys, marker=".", label=sp)
    ax.set_xlabel("step"); ax.set_ylabel("accuracy"); ax.set_title("eval accuracy")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    for sp, recs in by_split.items():
        xs = [r["step"] for r in recs]
        ys = [r.get("macro_f1", 0.0) for r in recs]
        ax.plot(xs, ys, marker=".", label=sp)
    ax.set_xlabel("step"); ax.set_ylabel("macro F1"); ax.set_title("eval macro-F1")
    ax.grid(True, alpha=0.3); ax.legend()

    out = run_dir / "curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_train_loss(run_dir: Path) -> None:
    import matplotlib.pyplot as plt
    records = _read_jsonl(run_dir / "train_loss.jsonl")
    if not records:
        return
    xs = [r["step"] for r in records]
    ys = [r["loss"] for r in records]
    # EMA for smoothing
    alpha = 0.05
    ema = []
    cur = ys[0] if ys else 0.0
    for y in ys:
        cur = alpha * y + (1 - alpha) * cur
        ema.append(cur)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, alpha=0.25, label="loss")
    ax.plot(xs, ema, linewidth=2, label=f"EMA (α={alpha})")
    ax.set_xlabel("step"); ax.set_ylabel("train loss"); ax.set_title("train loss")
    ax.grid(True, alpha=0.3); ax.legend()
    out = run_dir / "train_loss.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_confusion(run_dir: Path) -> None:
    import matplotlib.pyplot as plt
    test_metrics_path = run_dir / "test_metrics.json"
    if not test_metrics_path.exists():
        print(f"[plot] no test_metrics.json at {run_dir}")
        return
    met = json.loads(test_metrics_path.read_text(encoding="utf-8"))
    cm = np.asarray(met["confusion_matrix"])
    labels = POINT_LABELS if cm.shape[0] == len(POINT_LABELS) else PAIR_LABELS
    # Row-normalize (true-class recall)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sum = cm.sum(axis=1, keepdims=True)
        cmn = np.where(row_sum > 0, cm / row_sum, 0.0)

    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(labels), 0.8 + 0.6 * len(labels)))
    im = ax.imshow(cmn, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            if v == 0:
                continue
            color = "white" if cmn[i, j] > 0.5 else "black"
            ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"confusion matrix (row-normalized) — {met['accuracy']:.3f} acc, {met['macro_f1']:.3f} macro-F1")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = run_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_per_class_f1(run_dir: Path) -> None:
    import matplotlib.pyplot as plt
    test_metrics_path = run_dir / "test_metrics.json"
    if not test_metrics_path.exists():
        return
    met = json.loads(test_metrics_path.read_text(encoding="utf-8"))
    per_class = met.get("per_class", {})
    if not per_class:
        return
    labels = list(per_class.keys())
    f1s = [per_class[l]["f1"] for l in labels]
    supports = [per_class[l]["support"] for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels)), 4.5))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, f1s, color="tab:blue")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("F1"); ax.set_ylim(0, 1)
    ax.set_title("per-class F1")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (b, s) in enumerate(zip(bars, supports)):
        ax.text(b.get_x() + b.get_width()/2, min(0.97, b.get_height() + 0.02),
                f"n={s}", ha="center", va="bottom", fontsize=8)
    out = run_dir / "per_class_f1.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_all(run_dir: Path) -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[plot] matplotlib not installed; pip install matplotlib")
        return
    plot_eval_curves(run_dir)
    plot_train_loss(run_dir)
    plot_confusion(run_dir)
    plot_per_class_f1(run_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    plot_all(Path(args.run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
