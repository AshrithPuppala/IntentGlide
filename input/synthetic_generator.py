import math
import random
import time
from typing import Callable, List, Tuple
from shared.contracts import InputSample, GroundTruth, Target
from input.labels import make_label

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


class SyntheticReach:

    def __init__(self, start, target, duration=None, tremor_freq=8.0,
                 tremor_amp=10.0, trial_id=0, candidates=None,
                 clock=time.monotonic):
        self.x0, self.y0 = start
        self.target = target
        self.freq, self.amp = tremor_freq, tremor_amp
        self.trial_id = trial_id
        self.candidates = candidates      # all targets on screen, for ambiguity labels
        self.clock = clock
        dist = math.hypot(target.center_x - self.x0, target.center_y - self.y0)
        w = max(min(target.width, target.height), 1.0)
        self.duration = duration or (0.25 + 0.15 * math.log2(dist / w + 1))
        self.restart()

    def restart(self):
        self.start_time = self.clock()
        self.last_truth = None

    @property
    def target_id(self) -> str:
        return self.target.id

    def truth_at(self, t: float) -> GroundTruth:
        """Intended, tremor-free position at t seconds after start."""
        tau = min(max(t / self.duration, 0.0), 1.0)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5      # minimum-jerk
        return GroundTruth(
            timestamp=self.start_time + t,
            x=self.x0 + s * (self.target.center_x - self.x0),
            y=self.y0 + s * (self.target.center_y - self.y0),
            progress=tau, arrived=tau >= 1.0)

    def path(self, n: int = 200) -> List[GroundTruth]:
        """Whole ground-truth path, for plotting or scoring."""
        return [self.truth_at(self.duration * i / (n - 1)) for i in range(n)]

    @property
    def arrived(self) -> bool:
        return self.last_truth is not None and self.last_truth.arrived

    def get_sample(self) -> InputSample:
        now = self.clock()
        t = now - self.start_time
        truth = self.truth_at(t)
        self.last_truth = truth
        tremor = self.amp * math.sin(2 * math.pi * self.freq * t)
        return InputSample(timestamp=now, x=truth.x + tremor,
                           y=truth.y + tremor, source="synthetic")
    
    def get_label(self, override: bool = False):
        """Label for the most recent get_sample() call."""
        return make_label(self, self.last_truth, self.trial_id, self.candidates or [self.target], override)

class SyntheticInputGenerator:
    """Batch helper: returns n tremor samples at a fixed time step.
    Uses a fake clock, so it runs instantly and is deterministic."""

    def __init__(self, dt: float = 0.008, **kwargs):
        self.dt = dt
        self._t = [0.0]
        self._src = SyntheticInput(clock=lambda: self._t[0], **kwargs)

    def generate(self, n: int):
        samples = []
        for _ in range(n):
            self._t[0] += self.dt
            samples.append(self._src.get_sample())
        return samples

    