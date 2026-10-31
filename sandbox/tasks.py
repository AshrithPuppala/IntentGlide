"""Minimal state model for target-selection and drawing trials."""
from dataclasses import dataclass, field
from time import monotonic
from typing import Optional


@dataclass
class TrialState:
    task_type: str = "target-selection"
    running: bool = False
    start_time: Optional[float] = None
    selected_target_ids: list[str] = field(default_factory=list)
    wrong_target_activations: int = 0

    def start(self):
        self.running = True
        self.start_time = monotonic()

    def reset(self):
        self.running = False
        self.start_time = None
        self.selected_target_ids.clear()
        self.wrong_target_activations = 0

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, monotonic() - self.start_time) if self.start_time else 0.0
