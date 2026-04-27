"""End-to-end import + smoke test for the full stack (minus tree_sitter/torch-cuda)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plot_metrics
import tune
import pick_best
print("plot_metrics OK:", [n for n in dir(plot_metrics) if not n.startswith("_")][:5])
print("tune OK:", [n for n in dir(tune) if not n.startswith("_")][:5])
print("pick_best OK:", [n for n in dir(pick_best) if not n.startswith("_")][:5])

import train
print("train Config has n_seeds?:", "n_seeds" in train.Config.__dataclass_fields__)
print("train Config fields:", list(train.Config.__dataclass_fields__))
print("FULL STACK OK")
