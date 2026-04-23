"""LightGBM head."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .base import HeadRegistry


@HeadRegistry.register("lgbm")
class LGBMHead:
    def __init__(
        self,
        seed: int = 42,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        n_estimators: int = 400,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        min_split_gain: float = 0.0,
        early_stopping_rounds: int = 50,
    ) -> None:
        from lightgbm import LGBMClassifier
        self.hp = dict(
            seed=seed, num_leaves=num_leaves, max_depth=max_depth,
            learning_rate=learning_rate, n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            subsample=subsample, colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            min_split_gain=min_split_gain,
            early_stopping_rounds=early_stopping_rounds,
        )
        # bagging_freq must be >0 for subsample < 1 to take effect in LightGBM.
        bagging_freq = 1 if subsample < 1.0 else 0
        self.model = LGBMClassifier(
            objective="binary",
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_split_gain=min_split_gain,
            subsample_freq=bagging_freq,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        self._early_stop = early_stopping_rounds

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weight=None):
        import lightgbm as lgb
        sample_weight = None
        if class_weight is not None:
            sample_weight = np.asarray(
                [class_weight[int(v)] for v in y_train], dtype=np.float32,
            )
        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks = [lgb.early_stopping(self._early_stop, verbose=False),
                         lgb.log_evaluation(0)]
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        return {"best_iter": int(self.model.best_iteration_ or self.hp["n_estimators"])}

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_dir / "lgbm.pkl")
        joblib.dump({"hp": self.hp}, out_dir / "lgbm_meta.pkl")

    @classmethod
    def load(cls, out_dir: Path):
        meta = joblib.load(out_dir / "lgbm_meta.pkl")
        inst = cls(**meta["hp"])
        inst.model = joblib.load(out_dir / "lgbm.pkl")
        return inst

    def feature_importance(self) -> dict[str, float] | None:
        imp = self.model.booster_.feature_importance(importance_type="gain")
        names = self.model.booster_.feature_name()
        return {n: float(v) for n, v in zip(names, imp)}
