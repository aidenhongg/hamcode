"""Gaussian Naive Bayes head (sklearn)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .base import HeadRegistry


@HeadRegistry.register("gnb")
class GNBHead:
    def __init__(self, seed: int = 42, var_smoothing: float = 1e-9) -> None:
        from sklearn.naive_bayes import GaussianNB
        self.hp = dict(seed=seed, var_smoothing=var_smoothing)
        self.model = GaussianNB(var_smoothing=var_smoothing)

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weight=None):
        sample_weight = None
        if class_weight is not None:
            sample_weight = np.asarray(
                [class_weight[int(v)] for v in y_train], dtype=np.float32,
            )
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        return {}

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_dir / "gnb.pkl")
        joblib.dump({"hp": self.hp}, out_dir / "gnb_meta.pkl")

    @classmethod
    def load(cls, out_dir: Path):
        meta = joblib.load(out_dir / "gnb_meta.pkl")
        inst = cls(**meta["hp"])
        inst.model = joblib.load(out_dir / "gnb.pkl")
        return inst

    def feature_importance(self) -> dict[str, float] | None:
        return None
