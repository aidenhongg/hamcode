"""Quick smoke test of normalizer + complexity-mining regex."""
from complexity.normalizer import normalize, normalize_any
from complexity.ingest.utils import find_complexity_in_text

print("=== normalizer (O(...) strings) ===")
for raw in [
    "O(n)", "O(n^2)", "O(m * n)", "O(m + n)", "O(log n)", "O(n log n)",
    "O(2^n)", "O((m+n) log(m+n))", "O(rows * cols)", r"O(\log n)",
    "O(n*log n)", "O(V + E)", "O(V + E log V)", "O(k * n)", "O(n!)",
]:
    print(f"  {raw!r:40s} -> {normalize(raw)}")

print()
print("=== bare word labels (CodeComplex style) ===")
for raw in ["constant", "linear", "quadratic", "cubic", "logn", "nlogn",
            "np", "np-hard", "exponential", "factorial"]:
    print(f"  {raw!r:15s} -> {normalize_any(raw)}")

print()
print("=== find_complexity_in_text ===")
samples = [
    "# Time complexity: O(n log n)\n# Space: O(n)",
    '"""Solution to Two Sum.\n\n    Time: O(n), Space: O(n)\n    """',
    "// Time Complexity: O(m*n)",
    "class Solution:\n    # O(m + n) time\n    def solve(self):\n        pass",
    "// A lovely solution: O(2^n) time, O(n) space.",
    "no complexity here at all",
    "Time Complexity: O((m + n) log(m + n))",
    "Time - O(log n)",
]
for s in samples:
    found = find_complexity_in_text(s)
    print(f"  found={str(found)!r:40s} | snippet: {s[:60]!r}")
