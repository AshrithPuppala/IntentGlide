"""Random-Forest policy that predicts an adaptive Butterworth cutoff in Hz."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping
import math

import joblib
import numpy as np

from ml.features import FEATURE_NAMES
from shared.contracts import MotionFeatures, to_vector


class ButterworthCutoffPolicy:
    def __init__(
        self,
        model_path: str | Path | None = None,
        min_cutoff_hz: float = 1.5,
        max_cutoff_hz: float = 18.0,
        warmup_cutoff_hz: float = 8.0,
    ):
        if not (0.05 < min_cutoff_hz < max_cutoff_hz):
            raise ValueError("Invalid cutoff bounds")
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.max_cutoff_hz = float(max_cutoff_hz)
        self.warmup_cutoff_hz = float(warmup_cutoff_hz)
        self.model = None
        self.feature_names = list(FEATURE_NAMES)
        self.source = "heuristic_fallback"
        self.load_error = None

        if model_path is not None and Path(model_path).is_file():
            try:
                artifact = joblib.load(model_path)
                if not isinstance(artifact, dict) or "model" not in artifact:
                    raise ValueError("Cutoff artifact must contain model metadata")
                if list(artifact.get("feature_names", [])) != self.feature_names:
                    raise ValueError("Cutoff model feature schema does not match live DSP schema")
                if artifact.get("target") != "cutoff_optimal_hz":
                    raise ValueError("Artifact target is not cutoff_optimal_hz")
                bounds = artifact.get("cutoff_bounds_hz", [self.min_cutoff_hz, self.max_cutoff_hz])
                if len(bounds) != 2 or not (0 < float(bounds[0]) < float(bounds[1])):
                    raise ValueError("Invalid artifact cutoff bounds")
                self.min_cutoff_hz = max(self.min_cutoff_hz, float(bounds[0]))
                self.max_cutoff_hz = min(self.max_cutoff_hz, float(bounds[1]))
                if self.min_cutoff_hz >= self.max_cutoff_hz:
                    raise ValueError("Model and runtime cutoff bounds do not overlap")
                model = artifact["model"]
                if getattr(model, "n_features_in_", len(self.feature_names)) != len(self.feature_names):
                    raise ValueError("Model feature count does not match live DSP schema")
                self.model = model
                self.source = "ml"
            except Exception as exc:  # invalid/missing artifacts must not disable the app
                self.load_error = str(exc)

    def _fallback(self, features: MotionFeatures | Mapping[str, float]) -> float:
        if isinstance(features, MotionFeatures):
            score = float(features.oscillation_score)
            valid = bool(features.valid)
        else:
            score = float(features.get("oscillation_score", 0.0))
            valid = True
        if not valid:
            return self.warmup_cutoff_hz
        score = float(np.clip(score, 0.0, 1.0))
        return self.max_cutoff_hz * (self.min_cutoff_hz / self.max_cutoff_hz) ** score

    def predict_cutoff(self, features: MotionFeatures | Mapping[str, float]) -> float:
        if isinstance(features, MotionFeatures) and not features.valid:
            self.source = "warmup"
            return self.warmup_cutoff_hz

        if self.model is not None:
            try:
                if isinstance(features, MotionFeatures):
                    values = to_vector(features)
                else:
                    values = [float(features[name]) for name in self.feature_names]
                vector = np.asarray([values], dtype=float)
                if vector.shape != (1, len(self.feature_names)) or not np.isfinite(vector).all():
                    raise ValueError("Invalid cutoff-policy feature vector")
                prediction = float(self.model.predict(vector)[0])
                if not math.isfinite(prediction):
                    raise ValueError("Non-finite cutoff prediction")
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                self.load_error = str(exc)
                self.source = "heuristic_fallback"
                return self._fallback(features)
            self.source = "ml"
            return float(np.clip(prediction, self.min_cutoff_hz, self.max_cutoff_hz))

        self.source = "heuristic_fallback"
        return self._fallback(features)

    def get_status(self) -> dict:
        return {"source": self.source, "load_error": self.load_error}
