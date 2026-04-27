"""Logistic Regression head (sklearn)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .base import HeadRegistry


@HeadRegistry.register("logreg")
class LogRegHead:
    def __init__(
        self,
        seed: int = 42,
        C: float = 1.0,
        max_iter: int = 500,
    ) -> None:
        from sklearn.linear_model import LogisticRegression
        self.hp = dict(seed=seed, C=C, max_iter=max_iter)
        self.model = LogisticRegression(
            C=C, max_iter=max_iter, random_state=seed, n_jobs=-1,
        )
        self._columns: list[str] = []

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weight=None):
        # sklearn honors class_weight through constructor; re-create to pass it.
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(
            C=self.hp["C"],
            max_iter=self.hp["max_iter"],
            random_state=self.hp["seed"],
            class_weight=class_weight,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        return {"n_iter": int(self.model.n_iter_[0])}

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_dir / "logreg.pkl")
        joblib.dump({"hp": self.hp}, out_dir / "logreg_meta.pkl")

    @classmethod
    def load(cls, out_dir: Path):
        meta = joblib.load(out_dir / "logreg_meta.pkl")
        inst = cls(**meta["hp"])
        inst.model = joblib.load(out_dir / "logreg.pkl")
        return inst

    def feature_importance(self) -> dict[str, float] | None:
        # Return |coef| — magnitudes tell us which features the linear model weighted.
        coef = self.model.coef_[0]
        return {f"feat_{i}": float(abs(c)) for i, c in enumerate(coef)}
