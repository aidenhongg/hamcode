"""Classifier heads. All implement the Head protocol in base.py.

Import order is important on Windows: torch must load BEFORE xgboost /
lightgbm, otherwise their native DLLs can conflict with torch's (WinError 1114
on c10.dll). Hence `mlp` is imported first.
"""

# torch-bearing head loaded first to avoid DLL search path conflicts with
# xgboost/lightgbm native libraries on Windows
from . import mlp  # noqa: F401 — registration side effect

from .base import Head, HeadRegistry, get_head
from . import xgb, lgbm, logreg, rf, knn, gnb, stacked  # noqa: F401 — registration side effect

__all__ = ["Head", "HeadRegistry", "get_head"]
