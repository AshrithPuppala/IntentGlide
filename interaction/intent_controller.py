"""Confidence-gated controller with explicit ambiguity abstention."""
from __future__ import annotations

from typing import Iterable

from shared.contracts import FilterOutput, IntentDecision, MotionFeatures, Target
from interaction.target_analyzer import rank_targets


class IntentController:
    def __init__(
        self,
        confidence_threshold: float = 0.70,
        ambiguity_margin: float = 0.12,
        max_assist_px: float = 5.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.ambiguity_margin = ambiguity_margin
        self.max_assist_px = max_assist_px

    def decide(
        self,
        raw_x: float,
        raw_y: float,
        filtered: FilterOutput,
        features: MotionFeatures,
        targets: Iterable[Target],
        override: bool = False,
    ) -> IntentDecision:
        if override:
            return IntentDecision("override", None, 0.0, 0.0, 0.0)

        target_list = list(targets)
        if not target_list:
            state = "oscillation_detected" if features.oscillation_score > 0.5 else "stable"
            return IntentDecision(state, None, 0.0, 0.0, 0.0)

        ranked = rank_targets(
            filtered.x, filtered.y,
            raw_x - filtered.x, raw_y - filtered.y,
            target_list,
        )
        best_target, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score

        if best_score < self.confidence_threshold or (
            len(ranked) > 1 and margin < self.ambiguity_margin
        ):
            return IntentDecision(
                "target_ambiguous", None,
                float(max(0.0, min(1.0, best_score))), 0.0, 0.0
            )

        dx = best_target.center_x - filtered.x
        dy = best_target.center_y - filtered.y
        norm = max(1e-9, (dx * dx + dy * dy) ** 0.5)
        magnitude = min(self.max_assist_px, norm * 0.04)
        return IntentDecision(
            "target_confident", best_target.id,
            float(max(0.0, min(1.0, best_score))),
            float(dx / norm * magnitude),
            float(dy / norm * magnitude),
        )
