"""Sample random CodeComplex snippets to understand harness patterns."""
import random
import pandas as pd

df = pd.read_parquet("data/interim/codecomplex.parquet")
random.seed(42)

# One sample per class so we see variety
for label in ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "exponential"]:
    subset = df[df["label"] == label]
    if len(subset) == 0:
        continue
    row = subset.sample(1, random_state=42).iloc[0]
    code = row["code"]
    print(f"=== {label} (problem={row['problem_id']}) ===")
    print(code[:800])
    print("..." if len(code) > 800 else "")
    print()
