import math
import random
import time
from typing import Callable
from shared.contracts import InputSample

def _bounce(value: float, lo: float, hi: float) -> float:
    """Fold an unbounded position into [lo, hi] like a ball bouncing."""
    span = hi - lo
    m = (value - lo) % (2 * span)
    return lo + (m if m <= span else 2 * span - m)

class SyntheticInput:
    def __init__(self, start_x=100.0, start_y=100.0, speed=100.0,
                 tremor_freq=8.0, tremor_amp=15.0,
                 bounds=(0.0, 0.0, 1280.0, 720.0),   # x0, y0, x1, y1
                 amp_jitter=0.0, seed=None,
                 clock: Callable[[], float] = time.monotonic):
        self.start_x, self.start_y = start_x, start_y
        self.speed = speed
        self.freq = tremor_freq
        self.amp = tremor_amp
        self.bounds = bounds
        self.amp_jitter = amp_jitter    # 0.2 = +/-20% slow amplitude wobble
        self.rng = random.Random(seed)
        self._phase = self.rng.uniform(0, 2 * math.pi) if amp_jitter else 0.0
        self.clock = clock              # inject a fake clock for tests
        self.start_time = clock()

    def get_sample(self) -> InputSample:
        now = self.clock()
        t = now - self.start_time
        x0, y0, x1, y1 = self.bounds

        base_x = _bounce(self.start_x + self.speed * t, x0, x1)
        base_y = _bounce(self.start_y + self.speed * t, y0, y1)

        amp = self.amp
        if self.amp_jitter:
            amp *= 1 + self.amp_jitter * math.sin(0.5 * t + self._phase)

        tremor = amp * math.sin(2 * math.pi * self.freq * t)
        return InputSample(timestamp=now, x=base_x + tremor,
                           y=base_y + tremor, source="synthetic")
