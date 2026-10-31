import matplotlib.pyplot as plt
from shared.contracts import Target
from input.synthetic_generator import SyntheticReach

t = [0.0]
clock = lambda: t[0]
tgt = Target("btn_1", 500.0, 500.0, 50.0, 50.0)
gen = SyntheticReach((100, 100), tgt, clock=clock)

xs, ys = [], []
while not gen.arrived:
    t[0] += 0.008
    s = gen.get_sample()
    xs.append(s.x)
    ys.append(s.y)

truth = gen.path()
plt.plot(xs, ys, label="noisy cursor")
plt.plot([p.x for p in truth], [p.y for p in truth], "r--", label="ground truth")
plt.scatter([500], [500], c="g", s=200, label="target")
plt.legend()
plt.axis("equal")
plt.show()
print(f"duration {gen.duration:.2f}s, samples {len(xs)}")

import numpy as np
plt.figure()
plt.plot(np.arange(len(xs)) * 0.008, xs, label="noisy x")
plt.plot([p.timestamp for p in truth], [p.x for p in truth], "r--", label="truth x")
plt.xlabel("time (s)"); plt.legend(); plt.show()