"""Head protocol + registry.

Every head implements:
    Head(seed: int, **hp).fit(X_train, y_train, X_val, y_val, class_weight)
    Head.predict(X) -> np.ndarray  (N,)
    Head.predict_proba(X) -> np.ndarray  (N, 2)
    Head.save(out_dir: Path)
    Head.load(out_dir: Path) -> Head

Registry allows sweep.py and train_head.py to look up a head by short name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Head(Protocol):
    name: str

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        class_weight: dict[int, float] | None = None,
    ) -> dict: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, out_dir: Path) -> None: ...

    @classmethod
    def load(cls, out_dir: Path) -> "Head": ...

    def feature_importance(self) -> dict[str, float] | None: ...


class HeadRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def deco(klass):
            cls._registry[name] = klass
            klass.name = name
            return klass
        return deco

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._registry:
            raise KeyError(f"no head named {name!r}; registered: {list(cls._registry)}")
        return cls._registry[name]

    @classmethod
    def all(cls) -> list[str]:
        return list(cls._registry)


def get_head(name: str, seed: int, **hp) -> Head:
    return HeadRegistry.get(name)(seed=seed, **hp)


def compute_class_weight(y: np.ndarray) -> dict[int, float]:
    """Balanced class weight (sklearn convention)."""
    n = len(y)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        return {0: 1.0, 1: 1.0}
    return {0: n / (2 * n0), 1: n / (2 * n1)}
