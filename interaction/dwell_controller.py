"""Continuous, one-shot dwell activation with an explicit READY safety state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from interaction.target_geometry import distance_to_actual, inside_actual
from shared.contracts import Target


@dataclass(frozen=True)
class DwellResult:
    state: str
    target_id: Optional[str]
    progress: float
    activated: bool = False
    reason: str = ""


class DwellController:
    def __init__(self, dwell_seconds: float = 3.0, tolerance_px: float = 32.0,
                 exit_grace_seconds: float = 0.75, minimum_confidence: float = 0.65,
                 max_stable_speed_px_s: float = 90.0):
        if dwell_seconds <= 0 or tolerance_px < 0 or exit_grace_seconds < 0:
            raise ValueError("Invalid dwell timing or tolerance configuration")
        self.dwell_seconds = float(dwell_seconds)
        self.tolerance_px = float(tolerance_px)
        self.exit_grace_seconds = float(exit_grace_seconds)
        self.minimum_confidence = float(minimum_confidence)
        self.max_stable_speed_px_s = float(max_stable_speed_px_s)
        self.target_id: Optional[str] = None
        self._target: Optional[Target] = None
        self._elapsed = 0.0
        self._last_update: Optional[float] = None
        self._last_inside_time: Optional[float] = None
        self._ready = False
        self._last_cursor: tuple[float, float] | None = None
        self._blocked_target: Optional[Target] = None
        self._blocked_state: Optional[str] = None
        self._blocked_reason = ""

    def reset(self) -> None:
        self.target_id = None
        self._target = None
        self._elapsed = 0.0
        self._last_update = None
        self._last_inside_time = None
        self._ready = False
        self._last_cursor = None
        self._blocked_target = None
        self._blocked_state = None
        self._blocked_reason = ""

    def _in_region(self, x: float, y: float, target: Target) -> bool:
        return distance_to_actual(target, x, y) <= self.tolerance_px

    def _block(self, target: Target, state: str, reason: str = "") -> None:
        self._blocked_target = target
        self._blocked_state = state
        self._blocked_reason = reason
        self.target_id = None
        self._target = None
        self._elapsed = 0.0
        self._last_update = None
        self._last_inside_time = None
        self._ready = False
        self._last_cursor = None

    def mark_activation_blocked(self, reason: str) -> DwellResult:
        target = self._target or self._blocked_target
        if target is None:
            return DwellResult("blocked", self.target_id, 1.0, reason=str(reason))
        self._blocked_target = target
        self._blocked_state = "blocked"
        self._blocked_reason = str(reason)
        return DwellResult("blocked", target.id, 1.0, reason=str(reason))

    def _advance(self, now: float) -> None:
        if self._last_update is not None:
            # Timestamps are monotonic; cap a stalled frame to avoid time jumps.
            self._elapsed += min(0.20, max(0.0, now - self._last_update))
        self._last_update = now
        self._ready = self._ready or self._elapsed >= self.dwell_seconds

    def update(self, *, now: float, target: Optional[Target], confidence: float,
               cursor_x: float, cursor_y: float, filtered_speed_px_s: float,
               enabled: bool = True, override: bool = False,
               cancel_requested: bool = False,
               allow_virtual_activation: bool = False) -> DwellResult:
        now, x, y = float(now), float(cursor_x), float(cursor_y)

        # One shot: activation/cancel stays latched until the pointer exits the
        # outer acquisition region. No second activation while hovering.
        if self._blocked_target is not None:
            if self._in_region(x, y, self._blocked_target):
                state = self._blocked_state or "blocked"
                return DwellResult(state, self._blocked_target.id,
                                   1.0 if state == "activated" else 0.0,
                                   reason=self._blocked_reason or "leave target to re-arm")
            self._blocked_target = None
            self._blocked_state = None
            self._blocked_reason = ""

        if self._target is not None and self._last_cursor is not None:
            previous_distance = distance_to_actual(self._target, *self._last_cursor)
            current_distance = distance_to_actual(self._target, x, y)
            if (float(filtered_speed_px_s) > self.max_stable_speed_px_s
                    and current_distance > previous_distance + 1.0):
                old_id = self.target_id
                self.target_id = None
                self._target = None
                self._elapsed, self._ready = 0.0, False
                self._last_update = self._last_inside_time = None
                self._last_cursor = (x, y)
                return DwellResult("idle", old_id, 0.0, reason="deliberate movement away")
        self._last_cursor = (x, y)

        if cancel_requested or override or not enabled:
            previous = self._target
            if previous is not None and (cancel_requested or override):
                reason = "cancel requested" if cancel_requested else "manual override"
                self._block(previous, "cancelled", reason)
                return DwellResult("cancelled", previous.id, 0.0, reason=reason)
            self.reset()
            return DwellResult("idle", None, 0.0, reason="inactive")

        if target is not None and confidence < self.minimum_confidence:
            target = None

        if target is not None:
            if self.target_id is not None and target.id != self.target_id:
                # Never transfer accumulated dwell across competing targets.
                self.target_id, self._target = None, None
                self._elapsed, self._ready = 0.0, False
                self._last_update = self._last_inside_time = None
            if self.target_id is None:
                self.target_id, self._target = target.id, target
                self._elapsed, self._ready = 0.0, False
                self._last_update = now
            self._target = target
            inside = self._in_region(x, y, target)
            if inside:
                self._last_inside_time = now
                self._advance(now)
            else:
                # Target intent may persist briefly when the cursor strays.
                self._advance(now)
                if self._last_inside_time is None or now - self._last_inside_time > self.exit_grace_seconds:
                    old = target
                    self.target_id = None
                    self._target = None
                    self._elapsed, self._ready = 0.0, False
                    self._last_update = self._last_inside_time = None
                    return DwellResult("idle", old.id, 0.0, reason="outside acquisition region")
        else:
            # Brief intent loss or an excursion through the outer edge keeps
            # time moving; after grace, reset the in-flight attempt.
            if self._target is None:
                return DwellResult("idle", None, 0.0, reason="no confirmed target")
            self._advance(now)
            if self._last_inside_time is None or now - self._last_inside_time > self.exit_grace_seconds:
                old_id = self.target_id
                self.target_id = None
                self._target = None
                self._elapsed, self._ready = 0.0, False
                self._last_update = self._last_inside_time = None
                return DwellResult("idle", old_id, 0.0, reason="target lost beyond grace")

        active = self._target
        if active is None:
            return DwellResult("idle", None, 0.0)
        progress = min(1.0, self._elapsed / self.dwell_seconds)
        in_actual = inside_actual(active, x, y)
        in_activation_region = in_actual or (
            allow_virtual_activation and self._in_region(x, y, active)
        )
        stable = float(filtered_speed_px_s) <= self.max_stable_speed_px_s
        if self._ready:
            if in_activation_region and stable and target is not None:
                self._block(active, "activated", "dwell completed")
                return DwellResult("activated", active.id, 1.0, True, "dwell completed")
            reason = "return to target to activate" if not in_activation_region else "hold steady to activate"
            return DwellResult("ready", active.id, 1.0, False, reason)
        return DwellResult("dwelling", active.id, progress,
                           reason="" if stable else "timer continues; activation requires steady cursor")
