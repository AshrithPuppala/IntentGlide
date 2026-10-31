"""Small benchmark metric accumulator."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Metrics:
    trials: int = 0
    successful_acquisitions: int = 0
    wrong_target_activations: int = 0
    false_assistance: int = 0
    abstentions: int = 0
    processing_times_ms: list[float] = field(default_factory=list)
    completion_times_s: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful_acquisitions / self.trials if self.trials else 0.0

    @property
    def median_processing_ms(self) -> Optional[float]:
        if not self.processing_times_ms:
            return None
        values = sorted(self.processing_times_ms)
        n = len(values)
        mid = n // 2
        return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2
