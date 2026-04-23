"""Stacked meta-learner. Configurable base subset + meta classifier.

Base heads: any subset of {xgb, lgbm, mlp, logreg, rf}. Each is trained on
the full train set; their train-set class-1 probabilities become a feature
vector for the meta. Meta is one of {logreg, mlp, xgb}.

We do NOT nested-CV the base-head predictions for the meta (too slow on
the head sweep). In practice the meta is already a small model on a tiny
probability vector (≤5 dims), so the overfitting headroom is limited;
what matters is that the base heads are diverse enough that the meta
finds a non-trivial combination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

from .base import HeadRegistry


_DEFAULT_BASES: tuple[str, ...] = ("xgb", "lgbm", "mlp")
_ALLOWED_BASES: tuple[str, ...] = ("xgb", "lgbm", "mlp", "logreg", "rf")
_ALLOWED_METAS: tuple[str, ...] = ("logreg", "mlp", "xgb")


@HeadRegistry.register("stacked")
class StackedHead:
    def __init__(
        self,
        seed: int = 42,
        bases: Iterable[str] = _DEFAULT_BASES,
        meta: str = "logreg",
        base_hp: dict[str, dict] | None = None,
        meta_hp: dict | None = None,
    ) -> None:
        bases = tuple(bases)
        for b in bases:
            if b not in _ALLOWED_BASES:
                raise ValueError(
                    f"unknown base head {b!r}; expected subset of {_ALLOWED_BASES}"
                )
        if meta not in _ALLOWED_METAS:
            raise ValueError(
                f"unknown meta {meta!r}; expected one of {_ALLOWED_METAS}"
            )
        self.hp = dict(seed=seed, bases=list(bases), meta=meta,
                       base_hp=dict(base_hp or {}),
                       meta_hp=dict(meta_hp or {}))

        # Lazy-import the base head classes to avoid circular module churn.
        self.base = self._build_bases()
        self.meta_head = self._build_meta()

    # -----------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------

    def _build_bases(self) -> dict[str, object]:
        from . import xgb, lgbm, mlp, logreg, rf  # noqa: F401
        factories = {
            "xgb":    lambda hp: _XGB(seed=self.hp["seed"], **hp),
            "lgbm":   lambda hp: _LGBM(seed=self.hp["seed"], **hp),
            "mlp":    lambda hp: _MLP(seed=self.hp["seed"], **hp),
            "logreg": lambda hp: _LR(seed=self.hp["seed"], **hp),
            "rf":     lambda hp: _RF(seed=self.hp["seed"], **hp),
        }
        # Resolve head classes at call-time (avoid top-level import cycles).
        from .xgb import XGBHead as _XGB
        from .lgbm import LGBMHead as _LGBM
        from .mlp import MLPHead as _MLP
        from .logreg import LogRegHead as _LR
        from .rf import RFHead as _RF
        out: dict[str, object] = {}
        for b in self.hp["bases"]:
            hp = self.hp["base_hp"].get(b, {})
            out[b] = factories[b](hp)
        return out

    def _build_meta(self):
        m = self.hp["meta"]
        mhp = self.hp["meta_hp"]
        if m == "logreg":
            from .logreg import LogRegHead
            return LogRegHead(seed=self.hp["seed"], **mhp)
        if m == "mlp":
            from .mlp import MLPHead
            # Small defaults so meta doesn't overfit 2-5 input dims.
            defaults = dict(hidden_layers=1, hidden_dim=16, dropout=0.0,
                             epochs=40, batch_size=128, patience=5)
            defaults.update(mhp)
            return MLPHead(seed=self.hp["seed"], **defaults)
        if m == "xgb":
            from .xgb import XGBHead
            defaults = dict(max_depth=3, learning_rate=0.05, n_estimators=200)
            defaults.update(mhp)
            return XGBHead(seed=self.hp["seed"], **defaults)
        raise ValueError(m)

    # -----------------------------------------------------------------
    # Fit / predict
    # -----------------------------------------------------------------

    def _stack_features(self, X_base_probs: dict[str, np.ndarray]) -> np.ndarray:
        # Each base head yields (N, 2); take the P(class=1) column.
        cols = [X_base_probs[k][:, 1:2] for k in self.hp["bases"]]
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
        # Hand the meta the train-fold predictions. The meta can use its own
        # val-based early stopping if it's MLP/XGB; LogReg just converges.
        X_meta_val = None; y_meta_val = None
        if X_val is not None and y_val is not None:
            base_probs_val = {name: h.predict_proba(X_val)
                              for name, h in self.base.items()}
            X_meta_val = self._stack_features(base_probs_val)
            y_meta_val = y_val
        summaries["meta"] = self.meta_head.fit(
            X_meta_train, y_train, X_meta_val, y_meta_val,
            class_weight=class_weight,
        )
        return summaries

    def _meta_features(self, X):
        base_probs = {name: h.predict_proba(X) for name, h in self.base.items()}
        return self._stack_features(base_probs)

    def predict(self, X):
        return self.meta_head.predict(self._meta_features(X))

    def predict_proba(self, X):
        return self.meta_head.predict_proba(self._meta_features(X))

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, head in self.base.items():
            head.save(out_dir / f"base_{name}")
        self.meta_head.save(out_dir / "meta")
        joblib.dump({"hp": self.hp}, out_dir / "stacked_meta.pkl")

    @classmethod
    def load(cls, out_dir: Path):
        info = joblib.load(out_dir / "stacked_meta.pkl")
        hp = info["hp"]
        inst = cls(
            seed=hp["seed"], bases=hp["bases"], meta=hp["meta"],
            base_hp=hp.get("base_hp"), meta_hp=hp.get("meta_hp"),
        )
        # Rehydrate base + meta from disk (don't reuse the freshly-built ones)
        from . import xgb, lgbm, mlp, logreg, rf  # noqa
        from .xgb import XGBHead
        from .lgbm import LGBMHead
        from .mlp import MLPHead
        from .logreg import LogRegHead
        from .rf import RFHead
        klasses = {"xgb": XGBHead, "lgbm": LGBMHead, "mlp": MLPHead,
                   "logreg": LogRegHead, "rf": RFHead}
        inst.base = {n: klasses[n].load(out_dir / f"base_{n}") for n in hp["bases"]}
        meta_kls = {"logreg": LogRegHead, "mlp": MLPHead, "xgb": XGBHead}[hp["meta"]]
        inst.meta_head = meta_kls.load(out_dir / "meta")
        return inst

    def feature_importance(self) -> dict[str, float] | None:
        # Meta's coef/importance tells us relative weight on each base head.
        fi = self.meta_head.feature_importance()
        if fi is None:
            return None
        # Remap generic feature-index keys ("feat_0", ...) to base head names.
        out: dict[str, float] = {}
        for i, b in enumerate(self.hp["bases"]):
            out[b] = float(fi.get(f"feat_{i}", fi.get(f"f{i}", 0.0)))
        return out
