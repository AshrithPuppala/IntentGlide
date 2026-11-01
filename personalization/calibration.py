"""Guided, instructed-target calibration with pseudonymous trial records.

Calibration initializes a profile only; it intentionally does not train an ML model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import csv
import math
from pathlib import Path
import statistics
import time
import uuid
from typing import Iterable

from interaction.target_geometry import actual_rect, inside_actual, point_rect_distance
from interaction.target_analyzer import TARGET_FEATURE_NAMES, target_feature_dict
from personalization.user_profile import UserProfile
from shared.contracts import Target


@dataclass
class CalibrationSample:
    timestamp: float
    raw_x: float
    raw_y: float
    filtered_x: float
    filtered_y: float
    velocity_x: float
    velocity_y: float
    residual_x: float
    residual_y: float
    magnetic_strength: str
    acquisition_margin_px: float
    oscillation_score: float = 0.0
    dominant_frequency_hz: float = 0.0
    estimated_tremor_amplitude_px: float = 0.0
    acceleration_px_s2: float = 0.0
    assisted_x: float | None = None
    assisted_y: float | None = None
    magnetic_dx: float = 0.0
    magnetic_dy: float = 0.0


@dataclass
class CalibrationTrial:
    session_id: str
    profile_id: str
    trial_index: int
    intended_target_id: str
    target_width: float
    target_height: float
    target_center_x: float
    target_center_y: float
    start_time: float
    samples: list[CalibrationSample] = field(default_factory=list)
    actual_selected_target_id: str | None = None
    dwell_outcome: str = "incomplete"
    explicit_feedback: str | None = None
    completed: bool = False

    def metrics(self) -> dict:
        if len(self.samples) < 3 or not self.completed:
            raise ValueError("Incomplete calibration trial: need at least 3 samples and an outcome")
        times = [s.timestamp for s in self.samples]
        speeds = [math.hypot(s.velocity_x, s.velocity_y) for s in self.samples]
        residuals = [math.hypot(s.residual_x, s.residual_y) for s in self.samples]
        local_target = Target(self.intended_target_id, self.target_center_x, self.target_center_y,
                              self.target_width, self.target_height)
        rect = actual_rect(local_target)
        # Overshoot is movement beyond the far side after the first actual-bound entry.
        entered = False
        overshoots = []
        for sample in self.samples:
            if inside_actual(local_target, sample.filtered_x, sample.filtered_y):
                entered = True
            elif entered:
                overshoots.append(point_rect_distance(sample.filtered_x, sample.filtered_y, rect))
        corrections = 0
        last_direction = None
        for prev, curr in zip(self.samples, self.samples[1:]):
            dx, dy = curr.filtered_x-prev.filtered_x, curr.filtered_y-prev.filtered_y
            if math.hypot(dx, dy) < 1.0:
                continue
            direction = (1 if dx > 0 else -1 if dx < 0 else 0,
                         1 if dy > 0 else -1 if dy < 0 else 0)
            if last_direction is not None and direction != last_direction:
                corrections += 1
            last_direction = direction
        acquisition_time = max(0.0, times[-1] - self.start_time)
        return {
            "acquisition_time_s": acquisition_time,
            "median_movement_speed_px_s": statistics.median(speeds),
            "motion_variability_rms": math.sqrt(statistics.mean(r*r for r in residuals)),
            "overshoot_px": max(overshoots, default=0.0),
            "corrective_movements": corrections,
            "intended_target_id": self.intended_target_id,
            "actual_selected_target_id": self.actual_selected_target_id,
            "success": self.actual_selected_target_id == self.intended_target_id,
            "dwell_outcome": self.dwell_outcome,
            "explicit_feedback": self.explicit_feedback,
        }


class CalibrationSession:
    def __init__(self, profile: UserProfile, targets: Iterable[Target],
                 trial_count: int = 12, clock=time.monotonic):
        if not 10 <= trial_count <= 15:
            raise ValueError("Calibration uses 10–15 instructed target trials")
        self.profile = profile
        self.targets = list(targets)
        if not self.targets:
            raise ValueError("Calibration needs reachable targets")
        self.session_id = "session-" + uuid.uuid4().hex[:12]
        self.trial_count = trial_count
        self.clock = clock
        # Cycle among differently sized/positioned targets when the scene offers them.
        ordered = sorted(self.targets, key=lambda t: (t.width * t.height, t.center_y, t.center_x))
        self.instructions = [ordered[(i * max(1, len(ordered)//3) + i) % len(ordered)]
                             for i in range(trial_count)]
        self.trials: list[CalibrationTrial] = []
        self.current: CalibrationTrial | None = None
        self._start_next()

    @property
    def complete(self) -> bool:
        return len(self.trials) >= self.trial_count and self.current is None

    @property
    def current_target_id(self) -> str | None:
        return self.current.intended_target_id if self.current else None

    @property
    def progress(self) -> tuple[int, int]:
        return len(self.trials), self.trial_count

    def _start_next(self):
        index = len(self.trials)
        if index >= self.trial_count:
            self.current = None
            return
        target = self.instructions[index]
        self.current = CalibrationTrial(self.session_id, self.profile.profile_id, index, target.id,
                                        target.width, target.height, target.center_x,
                                        target.center_y, float(self.clock()))

    def record_sample(self, *, timestamp: float, raw_x: float, raw_y: float,
                      filtered_x: float, filtered_y: float, velocity_x: float = 0.0,
                      velocity_y: float = 0.0, magnetic_strength: str = "balanced",
                      acquisition_margin_px: float = 30.0,
                      oscillation_score: float = 0.0, dominant_frequency_hz: float = 0.0,
                      estimated_tremor_amplitude_px: float = 0.0,
                      acceleration_px_s2: float = 0.0, assisted_x: float | None = None,
                      assisted_y: float | None = None, magnetic_dx: float = 0.0,
                      magnetic_dy: float = 0.0):
        if self.current is None:
            return
        # Residual is defined against a local linear trajectory trend, not the
        # total cursor displacement. Use a trailing 350 ms least-squares fit.
        history = self.current.samples
        window = [*history, CalibrationSample(float(timestamp), float(raw_x), float(raw_y),
                  float(filtered_x), float(filtered_y), float(velocity_x), float(velocity_y),
                  0.0, 0.0, magnetic_strength, float(acquisition_margin_px))]
        recent = [s for s in window if timestamp-s.timestamp <= 0.35]
        rx = ry = 0.0
        if len(recent) >= 4:
            t0 = recent[0].timestamp
            ts = [s.timestamp-t0 for s in recent]
            den = sum((v-statistics.mean(ts))**2 for v in ts)
            if den > 1e-9:
                mx = statistics.mean(s.filtered_x for s in recent)
                my = statistics.mean(s.filtered_y for s in recent)
                vx = sum((t-statistics.mean(ts))*(s.filtered_x-mx) for t,s in zip(ts,recent))/den
                vy = sum((t-statistics.mean(ts))*(s.filtered_y-my) for t,s in zip(ts,recent))/den
                rx = float(raw_x) - (mx + vx*(float(timestamp)-statistics.mean(s.timestamp for s in recent)))
                ry = float(raw_y) - (my + vy*(float(timestamp)-statistics.mean(s.timestamp for s in recent)))
        sample = CalibrationSample(float(timestamp), float(raw_x), float(raw_y),
            float(filtered_x), float(filtered_y), float(velocity_x), float(velocity_y),
            rx, ry, magnetic_strength, float(acquisition_margin_px),
            float(oscillation_score), float(dominant_frequency_hz),
            float(estimated_tremor_amplitude_px), float(acceleration_px_s2),
            float(assisted_x) if assisted_x is not None else None,
            float(assisted_y) if assisted_y is not None else None,
            float(magnetic_dx), float(magnetic_dy))
        self.current.samples.append(sample)

    def finish_trial(self, actual_selected_target_id: str | None, *, dwell_outcome: str,
                     explicit_feedback: str | None = None):
        if self.current is None:
            raise RuntimeError("Calibration session is already complete")
        self.current.actual_selected_target_id = actual_selected_target_id
        self.current.dwell_outcome = str(dwell_outcome)
        self.current.explicit_feedback = explicit_feedback
        self.current.completed = True
        # Fail safely on incomplete trace, but retain the record for diagnostics.
        if len(self.current.samples) < 3:
            self.current.completed = False
            raise ValueError("Calibration trial needs at least three timestamped samples")
        self.trials.append(self.current)
        self._start_next()

    def summary(self) -> dict:
        metrics = [t.metrics() for t in self.trials if t.completed]
        if not metrics:
            raise ValueError("No completed calibration trials")
        return {
            "trial_count": len(metrics),
            "motion_variability_rms": statistics.median(m["motion_variability_rms"] for m in metrics),
            "median_overshoot_px": statistics.median(m["overshoot_px"] for m in metrics),
            "median_acquisition_time_s": statistics.median(m["acquisition_time_s"] for m in metrics),
            "median_movement_speed_px_s": statistics.median(m["median_movement_speed_px_s"] for m in metrics),
            "correction_frequency": statistics.mean(m["corrective_movements"] for m in metrics),
            "success_rate": statistics.mean(float(m["success"]) for m in metrics),
            "source": "calibration_observations_not_clinical_measurements",
        }

    def export_jsonl(self, path: Path | str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            for trial in self.trials:
                record = asdict(trial)
                record["metrics"] = trial.metrics() if trial.completed else None
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        return output

    def export_intent_csv(self, path: Path | str) -> Path:
        """Export candidate rows using the instructed target as the only label."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = ["schema_version", "data_source", "session_id", "profile_id", "trial_id",
                  "timestamp", "target_id", "intended_target_id", "is_intended", *TARGET_FEATURE_NAMES]
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for trial in self.trials:
                ordered = sorted(trial.samples, key=lambda sample: sample.timestamp)
                for index, sample in enumerate(ordered):
                    history = [(s.timestamp, s.filtered_x, s.filtered_y)
                               for s in ordered[:index+1] if sample.timestamp-s.timestamp <= 0.65]
                    profile_features = {"motion_variability_rms": self.profile.motion_variability_rms,
                        "median_overshoot_px": self.profile.median_overshoot_px,
                        "median_movement_speed": self.profile.median_movement_speed}
                    for target in self.targets:
                        features = target_feature_dict(history, sample.filtered_x, sample.filtered_y,
                            target, sample.acquisition_margin_px, profile_features)
                        writer.writerow({"schema_version": 2, "data_source": "calibration",
                            "session_id": trial.session_id, "profile_id": self.profile.profile_id,
                            "trial_id": trial.trial_index, "timestamp": sample.timestamp,
                            "target_id": target.id, "intended_target_id": trial.intended_target_id,
                            "is_intended": int(target.id == trial.intended_target_id), **features})
        return output
