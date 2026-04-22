"""Random Forest head (sklearn)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .base import HeadRegistry


@HeadRegistry.register("rf")
class RFHead:
    def __init__(
        self,
        seed: int = 42,
        n_estimators: int = 400,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier
        self.hp = dict(
            seed=seed, n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, random_state=seed, n_jobs=-1,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weight=None):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=self.hp["n_estimators"],
            max_depth=self.hp["max_depth"],
            min_samples_leaf=self.hp["min_samples_leaf"],
            random_state=self.hp["seed"],
            class_weight=class_weight,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        return {"n_trees": self.hp["n_estimators"]}

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_dir / "rf.pkl")
        joblib.dump({"hp": self.hp}, out_dir / "rf_meta.pkl")

    @classmethod
    def load(cls, out_dir: Path):
        meta = joblib.load(out_dir / "rf_meta.pkl")
        inst = cls(**meta["hp"])
        inst.model = joblib.load(out_dir / "rf.pkl")
        return inst

    def feature_importance(self) -> dict[str, float] | None:
        imp = self.model.feature_importances_
        return {f"feat_{i}": float(v) for i, v in enumerate(imp)}
