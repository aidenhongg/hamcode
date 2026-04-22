"""Inference helper: given Python source code, predict its complexity class."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_NAMES, extract_features
from .train import load_model


class ComplexityPredictor:
    def __init__(self, model_path: Path):
        self.model = load_model(model_path)
        self.feature_names = self.model.feature_names

    def predict_one(self, code: str) -> dict[str, Any]:
        feats = extract_features(code)
        x = np.array([[feats[name] for name in self.feature_names]], dtype=np.float64)
        probs = self.model.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        return {
            "label": self.model.labels[idx],
            "confidence": float(probs[idx]),
            "probabilities": {
                self.model.labels[i]: float(probs[i]) for i in range(len(probs))
            },
            "features": feats,
        }

    def predict_many(self, codes: list[str]) -> list[dict[str, Any]]:
        return [self.predict_one(c) for c in codes]
