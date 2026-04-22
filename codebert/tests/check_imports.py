"""Quick import check — verifies common modules load and do basic work."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import labels, schemas, normalizer

print("labels:", len(labels.POINT_LABELS), "classes")
print("pair labels:", labels.PAIR_LABELS)
print("tier of exponential:", labels.TIER["exponential"])
print("schema fields:", [f.name for f in schemas.POINT_SCHEMA])
print("normalize O(n log n):", normalizer.normalize("$O(n \\log n)$"))
print("normalize O(m*n):", normalizer.normalize("$O(m \\times n)$"))
print("normalize O(m+n) log(m+n):", normalizer.normalize("$O((m + n) \\times \\log(m + n))$"))
print("normalize exponential:", normalizer.normalize("$O(2^n)$"))
print("pair_label O(n) vs O(n^2):", labels.pair_label_from_labels("O(n)", "O(n^2)"))
print("pair_label O(m*n) vs O(n^2):", labels.pair_label_from_labels("O(m*n)", "O(n^2)"))
print("IMPORTS OK")
