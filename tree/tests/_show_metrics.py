"""Display test metrics + per-source breakdown after training."""
import json
import pandas as pd
from sklearn.metrics import f1_score
from complexity.features import FEATURE_NAMES
from complexity.labels import LABEL_TO_IDX
from complexity.train import load_model

m = json.load(open("models/metrics.json"))
t = m["test"]
print(f"OVERALL TEST: acc={t['accuracy']:.4f}  f1_macro={t['f1_macro']:.4f}  f1_weighted={t['f1_weighted']:.4f}")
print()
print(f"{'class':25s}{'prec':>8s}{'recall':>8s}{'f1':>8s}{'support':>10s}")
for k, v in t["classification_report"].items():
    if not (isinstance(v, dict) and "support" in v and v["support"] > 0):
        continue
    if k in ("micro avg", "macro avg", "weighted avg"):
        continue
    print(f"{k:25s}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1-score']:>8.3f}{int(v['support']):>10d}")

# Per-source F1 on test set
print()
print("Per-source test F1:")
model = load_model("models/point_rf.joblib")
df = pd.read_parquet("data/features/all.parquet")
test = df[df["split"] == "test"].copy()
for src in test["source"].unique():
    sub = test[test["source"] == src]
    if len(sub) < 10:
        continue
    import numpy as np
    X = sub[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = sub["label"].map(LABEL_TO_IDX).to_numpy(dtype=int)
    y_pred = model.predict(X)
    f1m = f1_score(y, y_pred, average="macro", zero_division=0)
    acc = (y_pred == y).mean()
    print(f"  {src:20s}  n={len(sub):4d}  acc={acc:.3f}  f1_macro={f1m:.3f}")
