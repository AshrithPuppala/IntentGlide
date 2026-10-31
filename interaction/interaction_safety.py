"""Safety helpers for override, cancellation and bounded assistance."""
from shared.contracts import FilterOutput


def respect_manual_override(
    raw: FilterOutput, filtered: FilterOutput, override: bool
) -> FilterOutput:
    """On override, return raw position and disable any assistance."""
    if override:
        return FilterOutput(raw.x, raw.y, raw.alpha, raw.mode)
    return filtered


def cancel_when_moving_away(
    current_x: float,
    current_y: float,
    target_x: float,
    target_y: float,
    velocity_x: float,
    velocity_y: float,
    cosine_threshold: float = -0.15,
) -> bool:
    """Return True when velocity points materially away from the target."""
    tx, ty = target_x - current_x, target_y - current_y
    denom = ((tx * tx + ty * ty) ** 0.5
             * (velocity_x * velocity_x + velocity_y * velocity_y) ** 0.5)
    if denom <= 1e-9:
        return False
    cosine = (tx * velocity_x + ty * velocity_y) / denom
    return cosine < cosine_threshold
