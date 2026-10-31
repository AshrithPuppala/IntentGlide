import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional
from shared.contracts import Target, TrialResult

@dataclass
class Trial:
    trial_id: int
    target_id: str
    start_time: float

class TrialManager:
    def __init__(self, targets: List[Target], timeout: float = 10.0,
                 end_on_miss: bool = False, seed=None,
                 clock: Callable[[], float] = time.monotonic):
        self.targets = {t.id: t for t in targets}
        self.timeout = timeout
        self.end_on_miss = end_on_miss
        self.rng = random.Random(seed)
        self.clock = clock
        self._next_id = 0
        self._trial: Optional[Trial] = None
        self._result: Optional[TrialResult] = None
        self._misses = 0

    # --- starting ---
    def start_trial(self, target_id: Optional[str] = None) -> Trial:
        if target_id is None:
            target_id = self.rng.choice(list(self.targets))
        if target_id not in self.targets:
            raise KeyError(f"Unknown target: {target_id}")
        self._next_id += 1
        self._trial = Trial(self._next_id, target_id, self.clock())
        self._result, self._misses = None, 0
        return self._trial

    # --- intended target ---
    @property
    def intended_target_id(self) -> Optional[str]:
        return self._trial.target_id if self._trial else None

    @property
    def intended_target(self) -> Optional[Target]:
        return self.targets[self._trial.target_id] if self._trial else None

    # --- click reporting ---
    def _hit(self, x: float, y: float) -> Optional[str]:
        for t in self.targets.values():
            if (abs(x - t.center_x) <= t.width / 2 and
                    abs(y - t.center_y) <= t.height / 2):
                return t.id
        return None

    def report_click(self, x: float, y: float) -> Optional[TrialResult]:
        """Pass the DISPLAYED cursor position. Returns a result if the trial ended."""
        if not self._trial or self._result:
            return None
        clicked = self._hit(x, y)
        if clicked == self._trial.target_id:
            return self._finish("success", clicked)
        if clicked is not None:
            return self._finish("wrong_target", clicked)
        self._misses += 1
        return self._finish("miss", None) if self.end_on_miss else None

    def update(self) -> Optional[TrialResult]:
        """Call once per frame to enforce the timeout."""
        if self._trial and not self._result and self.elapsed >= self.timeout:
            return self._finish("timeout", None)
        return None

    def _finish(self, outcome, clicked) -> TrialResult:
        end = self.clock()
        self._result = TrialResult(
            trial_id=self._trial.trial_id,
            intended_target_id=self._trial.target_id,
            clicked_target_id=clicked, outcome=outcome,
            success=(outcome == "success"),
            start_time=self._trial.start_time, end_time=end,
            elapsed=end - self._trial.start_time, miss_count=self._misses)
        return self._result

    # --- completion + time ---
    @property
    def is_complete(self) -> bool:
        return self._result is not None

    @property
    def elapsed(self) -> float:
        if self._result:
            return self._result.elapsed
        return self.clock() - self._trial.start_time if self._trial else 0.0

    @property
    def result(self) -> Optional[TrialResult]:
        return self._result