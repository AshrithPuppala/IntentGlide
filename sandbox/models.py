"""Plain-Python state behind the Settings and Benchmark windows (no pygame).

The windows only DRAW and EDIT these objects, so wiring up real data later is small:
  settings   main.py calls ``desktop.settings.subscribe(fn)`` and applies each change to the pipeline /
             disturbed mouse; set ``desktop.settings.connected = True`` to hide the "not connected" note.
  benchmark  the runner calls ``desktop.benchmark.set_results(rows, source)`` or ``.load_csv(path)``.
"""
from __future__ import annotations

import csv
from typing import Callable, Optional

MODES = ["raw", "fixed", "adaptive", "full", "ml_butterworth"]
MODE_LABELS = {"raw": "Raw", "fixed": "Fixed", "adaptive": "Adaptive", "full": "Full", "ml_butterworth": "ML filter"}


class SettingsModel:
    FREQ_RANGE = (3.0, 14.0)      # Hz
    AMP_RANGE = (0.0, 30.0)       # px
    FREQ_STEP, AMP_STEP = 1.0, 1.0
    PROFILE_RANGES = {
        "dwell_seconds": (2.0, 8.0), "acquisition_base_margin_px": (15.0, 50.0),
        "maximum_assistance_distance_px": (30.0, 150.0), "sticky_grace_ms": (200.0, 1000.0),
        "maximum_magnetic_speed_px_s": (30.0, 250.0),
    }
    PROFILE_STEPS = {"dwell_seconds": .5, "acquisition_base_margin_px": 5.0,
        "maximum_assistance_distance_px": 10.0, "sticky_grace_ms": 50.0,
        "maximum_magnetic_speed_px_s": 10.0}
    STRENGTHS = ("gentle", "balanced", "strong")

    def __init__(self, mode: str = "full", tremor_freq: float = 8.0, tremor_amp: float = 7.0,
                 tremor_on: bool = True):
        self.mode, self.tremor_freq, self.tremor_amp, self.tremor_on = mode, tremor_freq, tremor_amp, tremor_on
        self.dwell_seconds = 3.0
        self.acquisition_base_margin_px = 30.0
        self.maximum_assistance_distance_px = 76.0
        self.sticky_grace_ms = 400.0
        self.maximum_magnetic_speed_px_s = 145.0
        self.magnetic_strength = "balanced"
        self.connected = False
        self._listeners: list[Callable[[str, object], None]] = []

    def subscribe(self, fn: Callable[[str, object], None]):
        """fn(name, value) is called after every change; name is mode/tremor_freq/tremor_amp/tremor_on."""
        self._listeners.append(fn)

    def _emit(self, name: str, value):
        for fn in self._listeners:
            fn(name, value)

    def set_mode(self, mode: str):
        if mode in MODES and mode != self.mode:
            self.mode = mode
            self._emit("mode", mode)

    def step_freq(self, direction: int):
        self._set_num("tremor_freq", self.tremor_freq + direction * self.FREQ_STEP, self.FREQ_RANGE)

    def step_amp(self, direction: int):
        self._set_num("tremor_amp", self.tremor_amp + direction * self.AMP_STEP, self.AMP_RANGE)

    def _set_num(self, name: str, value: float, bounds):
        value = min(max(value, bounds[0]), bounds[1])
        if value != getattr(self, name):
            setattr(self, name, value)
            self._emit(name, value)

    def toggle_tremor(self):
        self.tremor_on = not self.tremor_on
        self._emit("tremor_on", self.tremor_on)

    def set_profile_values(self, profile):
        for name in (*self.PROFILE_RANGES, "magnetic_strength"):
            setattr(self, name, getattr(profile, name))

    def step_profile(self, name: str, direction: int):
        if name not in self.PROFILE_RANGES:
            raise KeyError(name)
        low, high = self.PROFILE_RANGES[name]
        step = self.PROFILE_STEPS[name]
        value = min(high, max(low, getattr(self, name)+direction*step))
        if value != getattr(self, name):
            setattr(self, name, value)
            self._emit(name, value)

    def cycle_strength(self):
        index = self.STRENGTHS.index(self.magnetic_strength)
        self.magnetic_strength = self.STRENGTHS[(index+1) % len(self.STRENGTHS)]
        self._emit("magnetic_strength", self.magnetic_strength)


# key, short column label, kind, which direction is better
METRICS = [
    ("success_rate", "Success", "pct", "high"),
    ("wrong_target_rate", "Wrong", "pct", "low"),
    ("miss_rate", "Miss", "pct", "low"),
    ("median_time_s", "Time s", "sec", "low"),
    ("trajectory_error_px", "Err px", "px", "low"),
    ("tremor_attenuation", "Atten.", "pct", "high"),
    ("processing_ms", "Proc ms", "ms", "low"),
    ("assist_rate", "Assist", "pct", None),
]
METRIC_LABELS = {k: lab for k, lab, _, _ in METRICS}
_FMT = {"pct": lambda v: f"{v * 100:.0f}%", "sec": lambda v: f"{v:.2f}", "px": lambda v: f"{v:.1f}",
        "ms": lambda v: f"{v:.2f}"}


def format_metric(key: str, value: Optional[float]) -> str:
    if value is None:
        return "-"
    kind = next(k for mk, _, k, _ in METRICS if mk == key)
    return _FMT[kind](value)


class BenchmarkModel:
    def __init__(self):
        self.rows: dict[str, dict[str, Optional[float]]] = {}
        self.source = "none"                    # none | simulated | live  (always label simulated runs)
        self.chart_metric = "success_rate"
        self.on_run: Optional[Callable[[], None]] = None

    @property
    def has_data(self) -> bool:
        return bool(self.rows)

    def set_results(self, rows: dict, source: str = "simulated"):
        self.rows = {m: {k: r.get(k) for k, *_ in METRICS} for m, r in rows.items()}
        self.source = source

    def clear(self):
        self.rows, self.source = {}, "none"

    def load_csv(self, path: str, source: str = "simulated") -> int:
        """Load a summary CSV: one row per mode, a ``mode`` column plus any METRICS keys."""
        rows = {}
        with open(path, newline="") as fh:
            for rec in csv.DictReader(fh):
                rows[rec["mode"]] = {k: (float(rec[k]) if rec.get(k) not in (None, "") else None)
                                     for k, *_ in METRICS}
        self.set_results(rows, source)
        return len(rows)

    def cycle_metric(self):
        keys = [k for k, *_ in METRICS]
        self.chart_metric = keys[(keys.index(self.chart_metric) + 1) % len(keys)]

    def best_mode(self, key: str) -> Optional[str]:
        better = next(b for mk, _, _, b in METRICS if mk == key)
        vals = {m: r.get(key) for m, r in self.rows.items() if r.get(key) is not None}
        if better is None or not vals:
            return None
        return (max if better == "high" else min)(vals, key=vals.get)

    def request_run(self):
        if self.on_run:
            self.on_run()

