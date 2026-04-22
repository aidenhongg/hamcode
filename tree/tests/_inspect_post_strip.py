"""Check what survives the stripper — find high-loop-count CodeComplex O(1) rows."""
import pandas as pd
from complexity.cp_preprocess import preprocess
from complexity.features import extract_features

df = pd.read_parquet("data/interim/codecomplex.parquet")
o1 = df[df["label"] == "O(1)"].head(200)

worst = []
for _, row in o1.iterrows():
    stripped = preprocess(row["code"])
    f = extract_features(stripped)
    worst.append((f["no_of_loop"], f["nested_loop_depth"], row["problem_id"], row["code"], stripped))
worst.sort(key=lambda x: -x[0])

for n_loops, depth, pid, orig, stripped in worst[:3]:
    print(f"\n{'='*80}")
    print(f"O(1) with {n_loops} loops after stripping, problem={pid}")
    print(f"{'='*80}")
    print("ORIGINAL (first 1500 chars):")
    print(orig[:1500])
    print("\n--- STRIPPED (first 1500 chars): ---")
    print(stripped[:1500])
    print("\n")
