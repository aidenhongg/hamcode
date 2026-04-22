"""Test hypothesis: CodeComplex (Codeforces) code style differs from LeetCode,
so mixing them may have hurt generalization on the LeetCode test set."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

from complexity.labels import LABEL_TO_IDX, POINT_LABELS
from complexity.features import FEATURE_NAMES
from complexity.train import load_model

model = load_model("models/point_rf.joblib")
df = pd.read_parquet("data/features/all.parquet")

# Per-source breakdown of the test set
test = df[df["split"] == "test"].copy()
print("Test set composition by source:")
print(test["source"].value_counts().to_string())
print()

# Evaluate separately on each source
for src in test["source"].unique():
    sub = test[test["source"] == src]
    if len(sub) < 10:
        continue
    X = sub[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = sub["label"].map(LABEL_TO_IDX).to_numpy(dtype=int)
    y_pred = model.predict(X)
    f1m = f1_score(y, y_pred, average="macro", zero_division=0)
    acc = (y_pred == y).mean()
    print(f"{src:25s}  n={len(sub):4d}  acc={acc:.3f}  f1_macro={f1m:.3f}")

# Feature statistics per source — do the same labels look different across sources?
print("\nMean feature values per source (for same labels):")
shared = df[df["label"].isin(["O(n)", "O(n^2)", "O(n log n)", "O(log n)"])].copy()
key_feats = ["nested_loop_depth", "no_of_loop", "noOfMethods", "no_of_sort", "recursion_present"]
tab = shared.groupby(["source", "label"])[key_feats].mean().round(2)
print(tab.to_string())
