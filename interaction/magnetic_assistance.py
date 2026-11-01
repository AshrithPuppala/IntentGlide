"""Time-based, edge-aware magnetic assistance for IntentGlide.

Call this only after the intent controller has confirmed a target. The helper
attracts the cursor toward the nearest point on the target rectangle, never
teleports it to the centre, and caps displacement by pixels/second.
"""
from __future__ import annotations

import math
from shared.contracts import FilterOutput, Target


def distance_to_target_rect(x: float, y: float, target: Target) -> float:
    left = target.center_x - target.width / 2.0
    right = target.center_x + target.width / 2.0
    top = target.center_y - target.height / 2.0
    bottom = target.center_y + target.height / 2.0
    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    return math.hypot(dx, dy)


def nearest_point_on_target(x: float, y: float, target: Target) -> tuple[float, float]:
    left = target.center_x - target.width / 2.0
    right = target.center_x + target.width / 2.0
    top = target.center_y - target.height / 2.0
    bottom = target.center_y + target.height / 2.0
    return min(max(x, left), right), min(max(y, top), bottom)


def apply_magnetic_assistance(
    filtered: FilterOutput,
    target: Target | None,
    *,
    dt: float,
    intent_confirmed: bool,
    confidence: float,
    minimum_confidence: float = 0.70,
    acquisition_radius_px: float = 70.0,
    attraction_rate: float = 9.0,
    max_speed_px_s: float = 130.0,
    manual_override: bool = False,
    moving_away: bool = False,
) -> FilterOutput:
    """Return a gently attracted output, or the original output unchanged.

    ``dt`` must be seconds. The caller supplies ``intent_confirmed`` from the
    temporal intent controller; this function does not infer intent itself.
    The cursor is attracted only while OUTSIDE the real target rectangle. Once
    inside, attraction stops so it does not drag the pointer toward the centre.
    """
    unchanged = FilterOutput(
        filtered.x, filtered.y, filtered.alpha, filtered.mode,
        filtered.processing_time_ms,
    )
    if (
        target is None
        or manual_override
        or moving_away
        or not intent_confirmed
        or confidence < minimum_confidence
        or acquisition_radius_px <= 0
        or max_speed_px_s <= 0
        or dt <= 0
    ):
        return unchanged

    distance = distance_to_target_rect(filtered.x, filtered.y, target)
    # Inside the actual clickable rectangle: don't tug the pointer around.
    if distance <= 1e-6 or distance >= acquisition_radius_px:
        return unchanged

    qx, qy = nearest_point_on_target(filtered.x, filtered.y, target)
    dx, dy = qx - filtered.x, qy - filtered.y
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return unchanged

    # Attraction increases smoothly as the pointer approaches the target edge.
    proximity = max(0.0, 1.0 - distance / acquisition_radius_px)
    safe_dt = min(float(dt), 0.05)  # avoid a large jump after a stalled frame
    fraction = 1.0 - math.exp(-attraction_rate * proximity * safe_dt)
    step = min(norm * fraction, max_speed_px_s * safe_dt)
    scale = step / norm

    return FilterOutput(
        x=filtered.x + dx * scale,
        y=filtered.y + dy * scale,
        alpha=filtered.alpha,
        mode=filtered.mode,
        processing_time_ms=filtered.processing_time_ms,
    )


class MagneticAssistance:
    """Stateful wrapper that accumulates the helper's bounded per-frame pull."""

    PROFILES = {
        "gentle": {"radius": 60.0, "rate": 7.0, "speed": 85.0},
        "balanced": {"radius": 76.0, "rate": 10.0, "speed": 145.0},
        "strong": {"radius": 82.0, "rate": 12.0, "speed": 150.0},
    }

    def __init__(self, profile: str = "balanced", max_offset_px: float = 40.0):
        if profile not in self.PROFILES:
            raise ValueError(f"Unknown magnetic profile: {profile}")
        self.profile = profile
        self.max_offset_px = float(max_offset_px)
        self.user_radius_px = self.PROFILES[profile]["radius"]
        self.user_speed_px_s = self.PROFILES[profile]["speed"]
        self.user_strength_scale = 1.0
        self.offset_x = self.offset_y = 0.0
        self.target_id: str | None = None
        self.last_time: float | None = None
        self.last_output: tuple[float, float] | None = None

    def reset(self) -> None:
        self.offset_x = self.offset_y = 0.0
        self.target_id = None
        self.last_time = None
        self.last_output = None

    def cycle_profile(self, direction: int = 1) -> str:
        names = tuple(self.PROFILES)
        index = names.index(self.profile)
        self.profile = names[(index + direction) % len(names)]
        self.reset()
        return self.profile

    def apply_profile(self, profile) -> None:
        """Keep the user's profile knobs independent from DSP/filter policies."""
        strength = str(profile.magnetic_strength)
        if strength in self.PROFILES:
            self.profile = strength
        self.user_radius_px = float(profile.maximum_assistance_distance_px)
        self.user_speed_px_s = float(profile.maximum_magnetic_speed_px_s)
        strength_scale = {"gentle": 0.7, "balanced": 1.0, "strong": 1.3}[self.profile]
        strength_scale *= float(profile.magnetic_strength_scale)
        self.user_strength_scale = float(profile.magnetic_strength_scale)
        self.max_offset_px = min(60.0, max(12.0, 40.0 * strength_scale))

    def apply(self, filtered: FilterOutput, target: Target | None, *,
              timestamp: float, velocity_x: float = 0.0, velocity_y: float = 0.0,
              tremor_amplitude: float = 0.0, confidence: float = 0.0,
              intent_confirmed: bool = False, manual_override: bool = False) -> FilterOutput:
        now = float(timestamp)
        dt = (1.0 / 120.0 if self.last_time is None else
              min(0.05, max(0.0, now - self.last_time)))
        self.last_time = now
        settings = self.PROFILES[self.profile]

        if target is None or not intent_confirmed or manual_override:
            decay = math.exp(-dt / 0.32)
            self.offset_x *= decay
            self.offset_y *= decay
            if math.hypot(self.offset_x, self.offset_y) < 0.05:
                self.offset_x = self.offset_y = 0.0
            self.target_id = None
            return self._output(filtered)

        self.target_id = target.id
        guided = FilterOutput(filtered.x + self.offset_x, filtered.y + self.offset_y,
                              filtered.alpha, filtered.mode, filtered.processing_time_ms)
        qx, qy = nearest_point_on_target(guided.x, guided.y, target)
        tx, ty = qx - guided.x, qy - guided.y
        speed = math.hypot(velocity_x, velocity_y)
        moving_away = (speed > 100.0 and math.hypot(tx, ty) > 1.0 and
                       (velocity_x * tx + velocity_y * ty) / (speed * math.hypot(tx, ty)) < -0.15)
        radius = min(150.0, self.user_radius_px + max(0.0, tremor_amplitude) * 1.2)
        result = apply_magnetic_assistance(
            guided, target, dt=dt, intent_confirmed=True, confidence=confidence,
            minimum_confidence=0.50, acquisition_radius_px=radius,
            attraction_rate=settings["rate"]*self.user_strength_scale,
            max_speed_px_s=self.user_speed_px_s,
            moving_away=moving_away,
        )
        if moving_away:
            decay = math.exp(-dt / 0.22)
            self.offset_x *= decay
            self.offset_y *= decay
        else:
            self.offset_x, self.offset_y = result.x - filtered.x, result.y - filtered.y
        length = math.hypot(self.offset_x, self.offset_y)
        if length > self.max_offset_px:
            scale = self.max_offset_px / length
            self.offset_x *= scale
            self.offset_y *= scale
        return self._output(filtered)

    def _output(self, filtered: FilterOutput) -> FilterOutput:
        self.last_output = (filtered.x + self.offset_x, filtered.y + self.offset_y)
        return FilterOutput(self.last_output[0], self.last_output[1], filtered.alpha,
                            filtered.mode, filtered.processing_time_ms)
