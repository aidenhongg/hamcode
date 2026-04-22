"""KNN head (sklearn, cosine metric)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .base import HeadRegistry


@HeadRegistry.register("knn")
class KNNHead:
    def __init__(
        self,
        seed: int = 42,
        n_neighbors: int = 11,
        weights: str = "distance",
        metric: str = "cosine",
    ) -> None:
        from sklearn.neighbors import KNeighborsClassifier
        self.hp = dict(
            seed=seed, n_neighbors=n_neighbors, weights=weights, metric=metric,
        )
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, metric=metric, n_jobs=-1,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weight=None):
        # Clip k to N-1 in case of very small data
        k = min(self.hp["n_neighbors"], max(1, len(y_train) - 1))
        if k != self.hp["n_neighbors"]:
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(
                n_neighbors=k, weights=self.hp["weights"],
                metric=self.hp["metric"], n_jobs=-1,
            )
        # KNN doesn't use class_weight directly; we replicate minority rows if needed.
        X_fit, y_fit = X_train, y_train
        if class_weight is not None and class_weight.get(0) != class_weight.get(1):
            reps = np.asarray(
                [int(round(class_weight[int(v)])) for v in y_train], dtype=int,
            )
            reps = np.maximum(reps, 1)
            X_fit = np.repeat(X_train, reps, axis=0)
            y_fit = np.repeat(y_train, reps, axis=0)
        self.model.fit(X_fit, y_fit)
        return {"k_effective": int(self.model.n_neighbors)}

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_dir / "knn.pkl")
        joblib.dump({"hp": self.hp}, out_dir / "knn_meta.pkl")

    @classmethod
    def load(cls, out_dir: Path):
        meta = joblib.load(out_dir / "knn_meta.pkl")
        inst = cls(**meta["hp"])
        inst.model = joblib.load(out_dir / "knn.pkl")
        return inst

    def feature_importance(self) -> dict[str, float] | None:
        return None
