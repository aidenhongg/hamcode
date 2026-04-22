"""Pre-import torch before any other native-lib-bearing package.

Windows has a persistent DLL search-path conflict between torch's c10.dll
and xgboost/lightgbm native libraries. When pytest collects tests that
transitively import xgboost / lightgbm FIRST, a later torch import fails
with WinError 1114. Importing torch here at package init bootstraps it
first so subsequent tests succeed.
"""

try:
    import torch  # noqa: F401
except Exception:
    # If torch is completely unavailable, tests that need it will skip naturally.
    pass
