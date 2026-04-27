"""XGBoost head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .base import HeadRegistry


@HeadRegistry.register("xgb")
class XGBHead:
    def __init__(
        self,
        seed: int = 42,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        n_estimators: int = 400,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
    ) -> None:
        self.hp = dict(
            seed=seed, max_depth=max_depth, learning_rate=learning_rate,
            n_estimators=n_estimators, subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight, gamma=gamma,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.model = None   # constructed at fit time so we can toggle early stopping
        self._columns: list[str] = []

    def _build(self, early_stop: bool):
        from xgboost import XGBClassifier
        kwargs = dict(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            max_depth=self.hp["max_depth"],
            learning_rate=self.hp["learning_rate"],
            n_estimators=self.hp["n_estimators"],
            subsample=self.hp["subsample"],
            colsample_bytree=self.hp["colsample_bytree"],
            min_child_weight=self.hp["min_child_weight"],
            gamma=self.hp["gamma"],
            reg_alpha=self.hp["reg_alpha"],
            reg_lambda=self.hp["reg_lambda"],
            random_state=self.hp["seed"],
            n_jobs=1,               # deterministic for reproducibility
            verbosity=0,
        )
        if early_stop:
            kwargs["early_stopping_rounds"] = self.hp["early_stopping_rounds"]
        return XGBClassifier(**kwargs)

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weight=None):
        sample_weight = None
        if class_weight is not None:
            sample_weight = np.asarray(
                [class_weight[int(v)] for v in y_train], dtype=np.float32,
            )
        has_val = X_val is not None and y_val is not None and len(y_val) > 0
        self.model = self._build(early_stop=has_val)
        eval_set = [(X_val, y_val)] if has_val else None
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=False,
        )
        return {"n_trees": int(self.model.get_booster().num_boosted_rounds())}

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(out_dir / "xgb.json"))
        joblib.dump({"hp": self.hp, "columns": self._columns},
                    out_dir / "xgb_meta.pkl")

    @classmethod
    def load(cls, out_dir: Path):
        from xgboost import XGBClassifier
        meta = joblib.load(out_dir / "xgb_meta.pkl")
        inst = cls(**meta["hp"])
        inst.model = XGBClassifier()
        inst.model.load_model(str(out_dir / "xgb.json"))
        inst._columns = meta["columns"]
        return inst

    def feature_importance(self) -> dict[str, float] | None:
        booster = self.model.get_booster()
        raw = booster.get_score(importance_type="gain")
        return {k: float(v) for k, v in raw.items()}
