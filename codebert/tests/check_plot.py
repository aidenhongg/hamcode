"""Write a synthetic test_metrics.json + metrics.jsonl and exercise plotting."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common.labels import POINT_LABELS
from plot_metrics import plot_all

random.seed(0)

run_dir = Path("tests/_plot_smoke_run")
run_dir.mkdir(parents=True, exist_ok=True)

# Synthetic eval curve
with (run_dir / "metrics.jsonl").open("w") as f:
    for step in range(200, 2001, 200):
        acc = min(0.9, 0.2 + step * 0.0003 + random.gauss(0, 0.02))
        f1 = min(0.85, 0.1 + step * 0.00035 + random.gauss(0, 0.02))
        f.write(json.dumps({
            "step": step, "epoch": step // 500, "split": "val",
            "accuracy": acc, "macro_f1": f1,
        }) + "\n")

# Synthetic train loss
with (run_dir / "train_loss.jsonl").open("w") as f:
    for step in range(1, 2001):
        loss = max(0.1, 2.5 - step * 0.001 + random.gauss(0, 0.15))
        f.write(json.dumps({"step": step, "epoch": step // 500, "loss": loss}) + "\n")

# Synthetic confusion matrix (diagonal-dominant)
n = len(POINT_LABELS)
cm = np.zeros((n, n), dtype=int)
rng = np.random.default_rng(0)
for i in range(n):
    cm[i, i] = rng.integers(20, 60)
    for j in range(n):
        if i != j:
            cm[i, j] = rng.integers(0, 8)

per_class = {}
for i, lab in enumerate(POINT_LABELS):
    tp = int(cm[i, i])
    fp = int(cm[:, i].sum() - tp)
    fn = int(cm[i, :].sum() - tp)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-8, (p + r))
    per_class[lab] = {"precision": p, "recall": r, "f1": f1, "support": int(cm[i, :].sum())}

total = cm.sum()
acc = cm.trace() / max(1, total)
macro_f1 = float(np.mean([pc["f1"] for pc in per_class.values()]))

(run_dir / "test_metrics.json").write_text(json.dumps({
    "accuracy": float(acc),
    "macro_f1": macro_f1,
    "within_1_tier_accuracy": 0.78,
    "per_class": per_class,
    "confusion_matrix": cm.tolist(),
}, indent=2), encoding="utf-8")

plot_all(run_dir)

expected = ["curves.png", "train_loss.png", "confusion_matrix.png", "per_class_f1.png"]
missing = [n for n in expected if not (run_dir / n).exists()]
assert not missing, f"missing: {missing}"
print("PLOT SMOKE OK:", sorted([p.name for p in run_dir.glob("*.png")]))
