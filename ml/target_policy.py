"""Optional, separately trained ML target ranker with heuristic fallback."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np

from interaction.target_analyzer import TARGET_FEATURE_NAMES, heuristic_target_score


class TargetIntentPolicy:
    """Scores each candidate target from causal cursor-history features.

    This is separate from FilterPolicy: FilterPolicy predicts smoothing alpha;
    this policy ranks candidate UI targets. The model artifact is optional.
    """
    def __init__(self, model_path: str | Path | None = None):
        default_path = Path(__file__).resolve().parent / "artifacts" / "intent_target_model.joblib"
        self.model_path = Path(model_path) if model_path is not None else default_path
        self.model = None
        self.source = "heuristic_fallback"
        self.load_error = None
        if self.model_path.is_file():
            try:
                artifact = joblib.load(self.model_path)
                if not isinstance(artifact, dict) or "model" not in artifact:
                    raise ValueError("Expected a model artifact dictionary")
                if artifact.get("status") != "evaluated_candidate":
                    raise ValueError("Intent artifact has not passed evaluation and promotion")
                if list(artifact.get("feature_names", [])) != list(TARGET_FEATURE_NAMES):
                    raise ValueError("Intent model feature schema mismatch")
                model = artifact["model"]
                if not hasattr(model, "predict_proba"):
                    raise ValueError("Intent model must implement predict_proba")
                self.model = model
                self.source = "ml"
            except Exception as exc:
                self.load_error = str(exc)

    def score_features(self, features: dict[str, float]) -> float:
        if self.model is None:
            return heuristic_target_score(features)
        try:
            vector = np.asarray([[float(features[name]) for name in TARGET_FEATURE_NAMES]], dtype=float)
            if not np.isfinite(vector).all():
                raise ValueError("Non-finite target feature")
            classes = list(self.model.classes_)
            if 1 not in classes:
                raise ValueError("Intent model has no positive class")
            positive_index = classes.index(1)
            score = float(self.model.predict_proba(vector)[0][positive_index])
            if not np.isfinite(score):
                raise ValueError("Non-finite intent score")
            return float(np.clip(score, 0.0, 1.0))
        except Exception as exc:
            self.load_error = str(exc)
            self.source = "heuristic_fallback"
            self.model = None
            return heuristic_target_score(features)

    def get_status(self) -> dict:
        return {"source": self.source, "load_error": self.load_error,
                "model_path": str(self.model_path)}
