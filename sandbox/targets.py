"""Target geometry and predefined layouts for the interactive sandbox."""
from shared.contracts import Target


def make_targets(layout: str = "standard", width: int = 900, height: int = 600):
    if layout == "ambiguous":
        return [
            Target("left", width * 0.48, height * 0.48, 56, 56),
            Target("right", width * 0.55, height * 0.48, 56, 56),
        ]
    size = {"small": 28, "large": 90}.get(layout, 56)
    return [
        Target(f"target-{i+1}", x, y, size, size)
        for i, (x, y) in enumerate([
            (width * 0.25, height * 0.30),
            (width * 0.70, height * 0.28),
            (width * 0.50, height * 0.68),
        ])
    ]
