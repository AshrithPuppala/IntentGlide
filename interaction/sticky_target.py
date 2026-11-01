"""Hysteretic target lock that tolerates brief tremor excursions."""
from __future__ import annotations

import math
from shared.contracts import Target
from interaction.target_geometry import distance_to_actual


class StickyTargetController:
    def __init__(self, exit_margin_px: float = 20.0, grace_seconds: float = 0.4,
                 switch_confirmation_seconds: float = 0.25, deliberate_speed_px_s: float = 140.0):
        if min(exit_margin_px, grace_seconds, switch_confirmation_seconds, deliberate_speed_px_s) < 0:
            raise ValueError("sticky-target settings must be non-negative")
        self.exit_margin_px = float(exit_margin_px)
        self.grace_seconds = float(grace_seconds)
        self.switch_confirmation_seconds = float(switch_confirmation_seconds)
        self.deliberate_speed_px_s = float(deliberate_speed_px_s)
        self.target_id: str | None = None
        self._exit_since: float | None = None
        self._challenger_id: str | None = None
        self._challenger_since: float | None = None

    @property
    def locked(self) -> bool:
        return self.target_id is not None

    def reset(self):
        self.target_id = self._challenger_id = None
        self._exit_since = self._challenger_since = None

    def acquire(self, target_id: str):
        self.target_id = target_id
        self._exit_since = self._challenger_since = None
        self._challenger_id = None

    def retain(self, *, x: float, y: float, now: float, targets: list[Target],
               velocity_x: float = 0.0, velocity_y: float = 0.0,
               challenger_id: str | None = None, challenger_margin: float = 0.0,
               required_margin: float = 0.12) -> Target | None:
        if self.target_id is None:
            return None
        target = next((t for t in targets if t.id == self.target_id), None)
        if target is None:
            self.reset()
            return None
        speed = math.hypot(velocity_x, velocity_y)
        dist = distance_to_actual(target, x, y)
        if speed >= self.deliberate_speed_px_s and dist > 1.0:
            # If movement points away from the nearest target boundary, release now.
            from interaction.target_geometry import actual_rect
            rect = actual_rect(target)
            qx, qy = min(max(x, rect.left), rect.right), min(max(y, rect.top), rect.bottom)
            dot = velocity_x * (x-qx) + velocity_y * (y-qy)
            if dot > 0:
                self.reset()
                return None
        if challenger_id and challenger_id != target.id and challenger_margin >= required_margin:
            if self._challenger_id != challenger_id:
                self._challenger_id, self._challenger_since = challenger_id, now
            elif self._challenger_since is not None and now-self._challenger_since >= self.switch_confirmation_seconds:
                self.acquire(challenger_id)
                return None
        else:
            self._challenger_id = self._challenger_since = None
        if dist <= self.exit_margin_px:
            self._exit_since = None
            return target
        if self._exit_since is None:
            self._exit_since = now
        if now - self._exit_since <= self.grace_seconds:
            return target
        self.reset()
        return None
