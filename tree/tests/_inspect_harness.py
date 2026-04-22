"""Find CodeComplex samples with high loop counts for cheap labels — these
should show us the harness patterns we need to strip."""
import pandas as pd
from complexity.features import extract_features

df = pd.read_parquet("data/interim/codecomplex.parquet")

# For O(1) and O(log n) snippets, any loop is probably harness
print("=" * 80)
print("CodeComplex O(1) snippets that our extractor sees as having loops (harness noise):")
print("=" * 80)
o1 = df[df["label"] == "O(1)"].head(200)
weird = []
for _, row in o1.iterrows():
    f = extract_features(row["code"])
    if f["no_of_loop"] > 0:
        weird.append((f["no_of_loop"], f["nested_loop_depth"], row))
weird.sort(key=lambda x: -x[0])
print(f"\nFound {len(weird)}/{len(o1)} O(1) snippets with loops > 0")
for n_loops, depth, row in weird[:3]:
    print(f"\n--- no_of_loop={n_loops}, depth={depth}, problem={row['problem_id']} ---")
    print(row['code'][:600])
    print("...")

print("\n" + "=" * 80)
print("Search full dataset for `for _ in range(int(input()))` harness pattern:")
print("=" * 80)
has_t_harness = df["code"].str.contains(
    r"for\s+_\s+in\s+range\s*\(\s*int\s*\(\s*input\s*\(\s*\)", regex=True, na=False
)
print(f"{has_t_harness.sum()} / {len(df)} snippets have the T-harness pattern")

# Also: does `input()` get called in loops?
called_in_loop = df["code"].str.contains(
    r"for\s+\w+\s+in\s+range\s*\([^)]+\):\s*[^\n]*input\(\)", regex=True, na=False
)
print(f"{called_in_loop.sum()} / {len(df)} snippets have input() called inside for-loop")
