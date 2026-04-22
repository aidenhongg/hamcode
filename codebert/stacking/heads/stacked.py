"""Stacked meta-learner: XGB + LGBM + MLP base heads, LogReg meta-classifier.

Base heads are trained on full train; their val-fold predictions train the meta.
Since we don't want to nested-CV (too slow on small val), we follow the common
"use base-head TRAINING predictions for the meta". Optimistic but simple, and
the whole stack is a tiny head — any overfitting is limited.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .base import HeadRegistry, compute_class_weight


@HeadRegistry.register("stacked")
class StackedHead:
    def __init__(self, seed: int = 42) -> None:
        from .xgb import XGBHead
        from .lgbm import LGBMHead
        from .mlp import MLPHead
        from .logreg import LogRegHead
        self.hp = dict(seed=seed)
        self.base = {
            "xgb": XGBHead(seed=seed),
            "lgbm": LGBMHead(seed=seed),
            "mlp": MLPHead(seed=seed),
        }
        self.meta = LogRegHead(seed=seed)

    def _stack_features(self, X_base_probs: dict[str, np.ndarray]) -> np.ndarray:
        # Each base head yields (N, 2); take the P(class=1) column.
        cols = [X_base_probs[k][:, 1:2] for k in sorted(self.base)]
        return np.concatenate(cols, axis=1)

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weight=None):
        base_probs_train: dict[str, np.ndarray] = {}
        summaries: dict[str, dict] = {}
        for name, head in self.base.items():
            summaries[name] = head.fit(
                X_train, y_train, X_val, y_val, class_weight=class_weight,
            )
            base_probs_train[name] = head.predict_proba(X_train)
        X_meta_train = self._stack_features(base_probs_train)
        meta_summary = self.meta.fit(X_meta_train, y_train, class_weight=class_weight)
        summaries["meta"] = meta_summary
        return summaries

    def _meta_features(self, X):
        base_probs = {name: h.predict_proba(X) for name, h in self.base.items()}
        return self._stack_features(base_probs)

    def predict(self, X):
        return self.meta.predict(self._meta_features(X))

    def predict_proba(self, X):
        return self.meta.predict_proba(self._meta_features(X))

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, head in self.base.items():
            head.save(out_dir / name)
        self.meta.save(out_dir / "meta")
        joblib.dump({"hp": self.hp, "base_names": list(self.base)},
                    out_dir / "stacked_meta.pkl")

    @classmethod
    def load(cls, out_dir: Path):
        from .xgb import XGBHead
        from .lgbm import LGBMHead
        from .mlp import MLPHead
        from .logreg import LogRegHead
        meta_info = joblib.load(out_dir / "stacked_meta.pkl")
        inst = cls(**meta_info["hp"])
        klass_by_name = {"xgb": XGBHead, "lgbm": LGBMHead, "mlp": MLPHead}
        inst.base = {n: klass_by_name[n].load(out_dir / n) for n in meta_info["base_names"]}
        inst.meta = LogRegHead.load(out_dir / "meta")
        return inst

    def feature_importance(self) -> dict[str, float] | None:
        return self.meta.feature_importance()
