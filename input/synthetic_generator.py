"""Repeatable intended pointer trajectories with optional sinusoidal disturbance."""
from __future__ import annotations

import math
from typing import Iterator

from shared.contracts import InputSample


class SyntheticInputGenerator:
    def __init__(
        self,
        sample_rate: float = 120.0,
        disturbance_frequency: float = 8.0,
        amplitude: float = 8.0,
        source: str = "synthetic",
        start_time: float = 0.0,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 4.0 <= disturbance_frequency <= 12.0:
            raise ValueError("disturbance_frequency must be in the 4–12 Hz test range")
        if amplitude < 0:
            raise ValueError("amplitude cannot be negative")
        self.sample_rate = float(sample_rate)
        self.frequency = float(disturbance_frequency)
        self.amplitude = float(amplitude)
        self.source = source
        self.start_time = float(start_time)

    def sample(self, index: int) -> InputSample:
        t = index / self.sample_rate
        intended_x = 100.0 + 120.0 * t
        intended_y = 180.0 + 45.0 * t
        phase = 2.0 * math.pi * self.frequency * t
        return InputSample(
            timestamp=self.start_time + t,
            x=intended_x + self.amplitude * math.sin(phase),
            y=intended_y + self.amplitude * 0.65 * math.cos(phase),
            source=self.source,
        )

    def generate(self, count: int) -> Iterator[InputSample]:
        if count < 0:
            raise ValueError("count cannot be negative")
        for i in range(count):
            yield self.sample(i)
