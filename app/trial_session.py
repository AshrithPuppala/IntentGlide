"""Trial mode (step 13): reach for a highlighted calculator key and click it.

Glue between four existing pieces, adding no new task logic of its own:
  sandbox.tasks.TrialManager     decides the goal and scores the click (success / wrong / miss / timeout)
  input.synthetic_generator      SyntheticReach drives the cursor toward the goal in auto mode
  sandbox.desktop.Desktop        draws the goal highlight and exposes the keypad as targets
  app.orchestrator               filters the input; the click lands on the PROCESSED cursor

Two ways to run a trial:
  manual  the user reaches with the (tremor-disturbed) mouse and clicks.
  auto    SyntheticReach reaches for the key and the session clicks after a short dwell.
          Auto trials chain until stopped. This is the mode the benchmark (step 14) will reuse.

One click ends a trial: hitting the goal is a success, hitting another key is a wrong target,
hitting nothing is a miss, and running out of time is a timeout.
"""
from __future__ import annotations

import random
import statistics
import time
from typing import Callable, Optional

from input.synthetic_generator import SyntheticReach
from sandbox.calculator import LABELS
from sandbox.tasks import TrialManager
from shared.contracts import TrialResult

KEY_PREFIX = "calculator:key:"


def key_label(target_id: Optional[str]) -> str:
    """'calculator:key:plus' -> '+'"""
    if not target_id:
        return "-"
    name = target_id.removeprefix(KEY_PREFIX)
    return LABELS.get(name, name)


class TrialSession:
    def __init__(self, desktop, inputs, on_reset: Callable[[], None], *,
                 clock: Callable[[], float] = time.monotonic, timeout: float = 8.0,
                 seed: Optional[int] = None, tremor_freq: float = 8.0, tremor_amp: float = 12.0,
                 dwell: float = 0.25, pause: float = 0.8):
        self.desktop = desktop
        self.inputs = inputs
        self.on_reset = on_reset          # called when the input source changes (clears filter state)
        self.clock = clock
        self.timeout = timeout
        self.rng = random.Random(seed)
        self.tremor_freq, self.tremor_amp = tremor_freq, tremor_amp
        self.dwell, self.pause = dwell, pause

        self.auto = False
        self.records: list[tuple[str, TrialResult]] = []     # (mode, result)
        self.last: Optional[tuple[str, TrialResult]] = None
        self.manager: Optional[TrialManager] = None
        self.reach: Optional[SyntheticReach] = None
        self.mode = ""
        self._prev_source: Optional[str] = None
        self._arrived_at: Optional[float] = None
        self._next_at: Optional[float] = None
        self._last_goal: Optional[str] = None

    # ---- state ---------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.manager is not None and not self.manager.is_complete

    @property
    def goal_id(self) -> Optional[str]:
        return self.manager.intended_target_id if self.active else None

    @property
    def elapsed(self) -> float:
        return self.manager.elapsed if self.manager else 0.0

    # ---- starting / stopping ------------------------------------------
    def start(self, mode: str, cursor_xy) -> bool:
        """Begin a trial in ``mode``. Returns False if the keypad is not available."""
        self.cancel()
        self.desktop.open_window("calculator")               # opens or raises it; closes any menu
        targets = self.desktop.keypad_targets()
        if not targets:
            return False
        ids = [t.id for t in targets if t.id != self._last_goal] or [t.id for t in targets]
        goal = self.rng.choice(ids)
        self.manager = TrialManager(targets, timeout=self.timeout, end_on_miss=True, clock=self.clock)
        trial = self.manager.start_trial(goal)
        self.mode, self._last_goal = mode, goal
        self._arrived_at = self._next_at = None
        self.desktop.goal_id = goal
        if self.auto:
            sx, sy = self._clamp(cursor_xy)
            self.reach = SyntheticReach(
                (sx, sy), self.manager.intended_target, tremor_freq=self.tremor_freq,
                tremor_amp=self.tremor_amp, trial_id=trial.trial_id, candidates=targets, clock=self.clock)
            self.inputs.sources["reach"] = self.reach
            self._prev_source = self.inputs.active
            self.inputs.set_source("reach")
            self.on_reset()
        return True

    def cancel(self):
        """Abandon the current trial without recording it (mode change, source change, quit)."""
        self.manager = None
        self._release_reach()
        self.desktop.goal_id = None
        self._next_at = None

    def stop_auto(self):
        self.auto = False
        self.cancel()

    def clear(self):
        self.cancel()
        self.records.clear()
        self.last = None

    def _clamp(self, xy):
        x = min(max(xy[0], 20.0), self.desktop.width - 20.0)
        y = min(max(xy[1], 20.0), self.desktop.height - self.desktop.TASKBAR_H - 20.0)
        return x, y

    def _release_reach(self):
        if self.reach is None:
            return
        self.reach = None
        if self._prev_source is not None and self.inputs.active == "reach":
            self.inputs.set_source(self._prev_source)
            self.on_reset()
        self._prev_source = None

    # ---- events ---------------------------------------------------------
    def click(self, cursor_xy) -> Optional[TrialResult]:
        """Report a click at the DISPLAYED (processed) cursor position."""
        if not self.active:
            return None
        result = self.manager.report_click(*cursor_xy)
        return self._finish(result) if result else None

    def update(self, cursor_xy, mode: str) -> Optional[TrialResult]:
        """Call once per frame. Handles the auto click, the timeout and chaining the next trial."""
        now = self.clock()
        if self.active:
            if self.reach is not None and self.reach.arrived:
                if self._arrived_at is None:
                    self._arrived_at = now
                elif now - self._arrived_at >= self.dwell:
                    self.desktop.click(*cursor_xy)          # press whatever is under the FILTERED cursor
                    return self.click(cursor_xy)            # ...then score that same click
            timed_out = self.manager.update()
            return self._finish(timed_out) if timed_out else None
        if self.auto and self._next_at is not None and now >= self._next_at:
            self.start(mode, cursor_xy)
        return None

    def _finish(self, result: TrialResult) -> TrialResult:
        self.records.append((self.mode, result))
        self.last = (self.mode, result)
        self.desktop.goal_id = None
        self._release_reach()
        self._next_at = self.clock() + self.pause if self.auto else None
        return result

    # ---- statistics -----------------------------------------------------
    def stats(self, mode: Optional[str] = None) -> dict:
        rows = [r for m, r in self.records if mode is None or m == mode]
        count = lambda o: sum(1 for r in rows if r.outcome == o)
        times = [r.elapsed for r in rows if r.success]
        n = len(rows)
        return {
            "n": n, "success": count("success"), "wrong_target": count("wrong_target"),
            "miss": count("miss"), "timeout": count("timeout"),
            "success_rate": (count("success") / n) if n else None,
            "median_time": statistics.median(times) if times else None,
        }

    def view(self, mode: str) -> dict:
        """Everything the HUD needs, as plain data."""
        return {
            "active": self.active, "auto": self.auto, "mode": mode,
            "goal": key_label(self.goal_id), "elapsed": self.elapsed, "timeout": self.timeout,
            "last": self.last, "last_label": (key_label(self.last[1].clicked_target_id)
                                              if self.last and self.last[1].clicked_target_id else None),
            "stats": self.stats(mode), "total": self.stats(),
        }
