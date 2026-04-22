"""Verify the per-source feature distribution shift is reduced after preprocessing."""
import pandas as pd

df = pd.read_parquet("data/features/all.parquet")
key = ["nested_loop_depth", "no_of_loop", "noOfMethods", "no_of_sort"]
shared = df[df["label"].isin(["O(n)", "O(n^2)", "O(n log n)", "O(log n)"])].copy()

print("Mean features per source AFTER cp_preprocess:")
print(shared.groupby(["source", "label"])[key].mean().round(2).to_string())

print("\n\nCompare before vs after for the O(n) overlap:")
print("Before: codecomplex O(n) no_of_loop=3.22, leetcode O(n) no_of_loop=1.45 (2x gap)")
codecomplex_on = df[(df['source']=='codecomplex') & (df['label']=='O(n)')]['no_of_loop'].mean()
leetcode_on = df[(df['source']=='leetcode') & (df['label']=='O(n)')]['no_of_loop'].mean()
print(f"After:  codecomplex O(n) no_of_loop={codecomplex_on:.2f}, leetcode O(n) no_of_loop={leetcode_on:.2f}")

print("\nBefore: codecomplex O(1) no_of_loop=? (had 25-loop outliers)")
cc_o1 = df[(df['source']=='codecomplex') & (df['label']=='O(1)')]['no_of_loop']
print(f"After:  codecomplex O(1): mean={cc_o1.mean():.2f}, max={cc_o1.max()}, P95={cc_o1.quantile(0.95):.1f}")
