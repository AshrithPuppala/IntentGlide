"""Headless intent-accuracy harness on the calculator keypad.

Simulates a user reaching for each keypad key with a bell-shaped (minimum-jerk) hand
movement plus tremor, then clicking. For every trial it records:
  * intent metrics (full mode): how often the controller was confident, and how often
    that confidence pointed at the CORRECT key;
  * the click outcome for all four modes (raw / fixed / adaptive / full).

Synthetic only: label results as simulated, not as user data.
Run:  python -m evaluation.intent_eval
"""
from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.orchestrator import IntentGlideOrchestrator
from sandbox.desktop import Desktop
from shared.contracts import InputSample

RATE = 120.0
MODES = ("raw", "fixed", "adaptive", "full")


def _min_jerk(tau: float) -> float:
    tau = min(1.0, max(0.0, tau))
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _progress(profile: str, tau: float) -> float:
    if profile == "linear":                       # constant speed, abrupt stop
        return min(1.0, max(0.0, tau))
    return _min_jerk(tau)                         # bell-shaped speed


def make_desktop() -> Desktop:
    desk = Desktop(820, 700)
    desk.click(*desk.icons[1].center)             # open Calculator
    return desk


def keypad_goals(desk: Desktop) -> dict:
    return {w.id: w for w in desk.windows[-1].widgets}


def start_points(goal, n: int = 3):
    gx, gy = goal.center
    pts = [(gx - 210, gy + 120), (gx + 190, gy - 110), (gx + 50, gy + 240), (gx - 120, gy - 170)]
    return [(min(800, max(10, x)), min(640, max(10, y))) for x, y in pts[:n]]


@dataclass
class Result:
    trials: int = 0
    # per phase: "start" = hand has covered <15% of the path (still near its start point, where the key
    # under the cursor is a legitimate reading), "approach" = real movement, "dwell" = after arrival
    phase: dict = field(default_factory=lambda: {p: Counter() for p in ("start", "approach", "dwell")})
    clicks: dict = field(default_factory=lambda: {m: Counter() for m in MODES})

    def add(self, other: "Result"):
        self.trials += other.trials
        for p in self.phase:
            self.phase[p].update(other.phase[p])
        for m in MODES:
            self.clicks[m].update(other.clicks[m])


def run_trial(desk, goal_id, start, seed, make_pipeline: Callable[[], IntentGlideOrchestrator],
              avg_speed=260.0, amp=7.0, hz=8.0, profile="minjerk", dwell=0.5,
              aim_sigma=10.0) -> Result:
    rng = random.Random(seed)
    goal = keypad_goals(desk)[goal_id]
    gx, gy = goal.center
    sx, sy = start
    # Endpoint error: the hand lands near, not exactly on, the intended key centre.
    # The user's INTENDED target stays `goal`; only where the hand physically ends is noisy.
    ex, ey = gx + rng.gauss(0, aim_sigma), gy + rng.gauss(0, aim_sigma)
    dist = math.hypot(ex - sx, ey - sy)
    move_t = max(0.3, dist / avg_speed)
    total = move_t + dwell
    ph1, ph2 = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)

    pipe = make_pipeline()
    pipe.set_mode("full")
    res = Result(trials=1)
    t, dt, last = 0.0, 1.0 / RATE, None
    targets = desk.targets
    while t <= total:
        f = _progress(profile, t / move_t)
        x = sx + (ex - sx) * f + amp * math.sin(2 * math.pi * hz * t + ph1)
        y = sy + (ey - sy) * f + amp * 0.65 * math.cos(2 * math.pi * hz * t + ph2)
        out = pipe.process(InputSample(1000.0 + t, x, y, "synthetic"), targets=targets)
        dec = out["decision"]
        ph = res.phase["dwell" if t >= move_t else ("start" if f < 0.15 else "approach")]
        ph["frames"] += 1
        if dec.state == "target_confident":
            ph["confident"] += 1
            ph["correct" if dec.target_id == goal_id else "wrong"] += 1
        last = out
        t += dt

    for mode, pos in (("raw", last["raw"]), ("fixed", last["fixed"]),
                      ("adaptive", last["adaptive"]), ("full", last["output"])):
        hit = desk._hit(pos.x, pos.y)
        outcome = "success" if hit is not None and hit.id == goal_id else ("wrong" if hit is not None else "miss")
        res.clicks[mode][outcome] += 1
    return res


def evaluate(make_pipeline, seeds=(0, 1), starts=3, goals: Optional[list] = None, **kw) -> Result:
    desk = make_desktop()
    total = Result()
    for goal_id in (goals or list(keypad_goals(desk))):
        for si, start in enumerate(start_points(keypad_goals(desk)[goal_id], starts)):
            for seed in seeds:
                total.add(run_trial(desk, goal_id, start, seed * 1000 + si, make_pipeline, **kw))
    return total


def report(res: Result, title: str = ""):
    print(f"\n{title}  ({res.trials} simulated trials)")
    for name, ph in res.phase.items():
        n, c = max(1, ph["frames"]), ph["confident"]
        print(f"  intent/{name:8s} confident {100 * c / n:5.1f}% of frames | "
              f"correct when confident {100 * ph['correct'] / max(1, c):5.1f}% | wrong-key frames {ph['wrong']}")
    print("  final click   success   wrong-key   miss")
    for m in MODES:
        c = res.clicks[m]
        n = max(1, sum(c.values()))
        print(f"    {m:9s} {100 * c['success'] / n:6.1f}%   {100 * c['wrong'] / n:7.1f}%  {100 * c['miss'] / n:6.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--amp", type=float, default=7.0)
    ap.add_argument("--aim", type=float, default=10.0, help="endpoint error sigma (px)")
    ap.add_argument("--profile", default="minjerk", choices=("minjerk", "linear"))
    ap.add_argument("--scoring", default="posterior", choices=("posterior", "legacy"),
                    help="intent scoring to evaluate (legacy = the original controller)")
    args = ap.parse_args()

    def make_pipeline():
        from interaction.intent_controller import IntentController
        pipe = IntentGlideOrchestrator()
        pipe.intent = IntentController(scoring=args.scoring)
        return pipe

    report(evaluate(make_pipeline, seeds=tuple(range(args.seeds)), amp=args.amp,
                    profile=args.profile, aim_sigma=args.aim), f"[{args.scoring}] profile={args.profile} tremor={args.amp}px aim-error={args.aim}px")
