"""Import-check heavier modules that need torch/transformers but not tree_sitter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data
import model
import metrics
print("data.py         :", [name for name in dir(data) if not name.startswith("_")][:8])
print("model.py        :", [name for name in dir(model) if not name.startswith("_")][:8])
print("metrics.py      :", [name for name in dir(metrics) if not name.startswith("_")][:8])

import train
print("train.py OK, Config fields:", list(train.Config.__dataclass_fields__)[:8])

import predict
print("predict.py OK")

print("ALL IMPORTS OK")
