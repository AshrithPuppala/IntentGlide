import matplotlib.pyplot as plt
import numpy as np
from shared.contracts import Target
from input.synthetic_generator import SyntheticReach
from dsp.motion_analyzer import MotionAnalyzer
from dsp.adaptive_filter import AdaptiveFilter
from dsp.fixed_filter import FixedFilter

t = [0.0]
clock = lambda: t[0]
target = Target("btn_1", 700.0, 450.0, 50.0, 50.0)
reach = SyntheticReach((100.0, 100.0), target, duration=1.8, tremor_freq=8.0, tremor_amp=14.0, clock=clock)

analyzer = MotionAnalyzer()
adaptive = AdaptiveFilter()
fixed = FixedFilter()

raw_x, raw_y = [], []
fixed_x, fixed_y = [], []
adapt_x, adapt_y = [], []
alphas = []
times = []

while not reach.arrived:
    t[0] += 0.008
    s = reach.get_sample()

    feats = analyzer.update(s)
    out_f = fixed.update(s)
    out_a = adaptive.update(s, feats)

    times.append(t[0])
    raw_x.append(s.x); raw_y.append(s.y)
    fixed_x.append(out_f.x); fixed_y.append(out_f.y)
    adapt_x.append(out_a.x); adapt_y.append(out_a.y)
    alphas.append(out_a.alpha or 0.0)

truth = reach.path(len(times))

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Trajectory Plot
ax1.plot(raw_x, raw_y, color="lightgray", label="Raw Noisy Input")
ax1.plot([p.x for p in truth], [p.y for p in truth], "k--", label="Ground Truth (Ideal)")
ax1.plot(fixed_x, fixed_y, color="crimson", alpha=0.7, label="Fixed Butterworth")
ax1.plot(adapt_x, adapt_y, color="royalblue", linewidth=2, label="Adaptive Filter")
ax1.scatter([target.center_x], [target.center_y], color="limegreen", s=180, zorder=5, label="Target")
ax1.set_title("2D Spatial Trajectory")
ax1.set_xlabel("X (px)")
ax1.set_ylabel("Y (px)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Time vs X Coordinate
ax2.plot(times, raw_x, color="lightgray", label="Raw X")
ax2.plot(times, [p.x for p in truth], "k--", label="Truth X")
ax2.plot(times, fixed_x, color="crimson", alpha=0.7, label="Fixed X")
ax2.plot(times, adapt_x, color="royalblue", linewidth=2, label="Adaptive X")
ax2.set_title("X-Axis Position vs. Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Position (px)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()