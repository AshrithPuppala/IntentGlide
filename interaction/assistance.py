"""Apply a bounded IntentDecision offset to the selected filtered position."""
from shared.contracts import FilterOutput, IntentDecision


def apply_assistance(filtered: FilterOutput, decision: IntentDecision) -> FilterOutput:
    if decision.state != "target_confident":
        return FilterOutput(filtered.x, filtered.y, filtered.alpha, filtered.mode)
    return FilterOutput(
        x=filtered.x + decision.assist_dx,
        y=filtered.y + decision.assist_dy,
        alpha=filtered.alpha,
        mode=filtered.mode,
    )
