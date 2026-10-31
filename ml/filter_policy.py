"""Predict a bounded alpha from shared DSP features, with heuristic fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping
import math

import joblib
import numpy as np

from ml.features import FEATURE_NAMES
from shared.contracts import MotionFeatures, to_vector


class FilterPolicy:
    def __init__(self, model_path: str | Path | None = None,
                 min_alpha: float = 0.12, max_alpha: float = 0.85):
        if not (0 < min_alpha <= max_alpha <= 1):
            raise ValueError("Require 0 < min_alpha <= max_alpha <= 1")
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.model = None
        self.feature_names = list(FEATURE_NAMES)
        self.source = "heuristic_fallback"
        if model_path is not None and Path(model_path).is_file():
            try:
                artifact = joblib.load(model_path)
                if isinstance(artifact, dict) and "model" in artifact:
                    names = artifact.get("feature_names")
                    if names is None or list(names) != self.feature_names:
                        raise ValueError("Model feature schema does not match the live DSP schema")
                    model = artifact["model"]
                else:
                    model = artifact
                model_features = getattr(model, "n_features_in_", len(self.feature_names))
                if model_features != len(self.feature_names):
                    raise ValueError("Model feature count does not match the live DSP schema")
                self.model = model
            except Exception as exc:
                self.load_error = str(exc)
            else:
                self.source = "ml"
        self.load_error = getattr(self, "load_error", None)

    def predict_alpha(self, features: MotionFeatures | Mapping[str, float]) -> float:
        if self.model is not None:
            try:
                if isinstance(features, MotionFeatures):
                    values = to_vector(features)
                else:
                    values = [float(features[name]) for name in self.feature_names]
                vector = np.asarray([values], dtype=float)
                if not np.isfinite(vector).all():
                    raise ValueError("Non-finite feature")
                target = float(self.model.predict(vector)[0])
                if not math.isfinite(target):
                    raise ValueError("Non-finite model prediction")
            except (KeyError, TypeError, ValueError, AttributeError):
                target = self._heuristic(features)
                self.source = "heuristic_fallback"
            else:
                self.source = "ml"
        else:
            target = self._heuristic(features)
            self.source = "heuristic_fallback"
        return float(np.clip(target, self.min_alpha, self.max_alpha))

    def _heuristic(self, features: MotionFeatures | Mapping[str, float]) -> float:
        if isinstance(features, MotionFeatures):
            score = features.oscillation_score
        else:
            score = features.get("oscillation_score", 0.0)
        score = float(np.clip(score, 0.0, 1.0))
        return self.max_alpha - score * (self.max_alpha - self.min_alpha)

    def get_status(self) -> dict:
        return {"source": self.source, "load_error": self.load_error}
