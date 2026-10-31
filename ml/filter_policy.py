"""Runtime alpha policy with bounds, rate limiting and heuristic fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping
import math

import joblib
import numpy as np

from ml.features import FEATURE_NAMES


class FilterPolicy:
    def __init__(self, model_path: str | Path | None = None,
                 min_alpha: float = 0.12, max_alpha: float = 0.85,
                 initial_alpha: float = 0.65, max_alpha_step: float = 0.05):
        if not (0 < min_alpha <= max_alpha <= 1):
            raise ValueError("Require 0 < min_alpha <= max_alpha <= 1")
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.alpha = float(np.clip(initial_alpha, min_alpha, max_alpha))
        self.max_alpha_step = float(max_alpha_step)
        self.model = None
        self.feature_names = list(FEATURE_NAMES)
        self.source = "heuristic_fallback"
        if model_path is not None and Path(model_path).is_file():
            artifact = joblib.load(model_path)
            if isinstance(artifact, dict) and "model" in artifact:
                names = artifact.get("feature_names", FEATURE_NAMES)
                if list(names) != list(FEATURE_NAMES):
                    raise ValueError("Model feature schema does not match this runtime feature schema")
                self.model = artifact["model"]
            else:
                self.model = artifact
            self.source = "ml"

    def predict_alpha(self, features: Mapping[str, float]) -> float:
        if self.model is not None:
            try:
                vector = np.array([[float(features[name]) for name in self.feature_names]], dtype=float)
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
        target = float(np.clip(target, self.min_alpha, self.max_alpha))
        delta = float(np.clip(target - self.alpha, -self.max_alpha_step, self.max_alpha_step))
        self.alpha = float(np.clip(self.alpha + delta, self.min_alpha, self.max_alpha))
        return self.alpha

    def _heuristic(self, features: Mapping[str, float]) -> float:
        score = float(features.get("oscillation_score", 0.0))
        score = float(np.clip(score, 0.0, 1.0))
        return self.max_alpha - score * (self.max_alpha - self.min_alpha)

    def get_status(self) -> dict:
        return {"source": self.source, "alpha": self.alpha}
