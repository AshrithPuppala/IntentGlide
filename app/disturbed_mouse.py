"""Real mouse plus an injected software tremor, so the desktop can be driven by hand.

Temporary home: Person 1 may move this into input/ (disturbance profiles).
"""
import math


class DisturbedMouse:
    def __init__(self, mouse, freq_hz: float = 8.0, amplitude_px: float = 7.0):
        self.mouse = mouse
        self.freq = freq_hz
        self.amp = amplitude_px
        self.enabled = True

    def get_sample(self):
        s = self.mouse.get_sample()
        if self.enabled:
            phase = 2.0 * math.pi * self.freq * s.timestamp
            s.x += self.amp * math.sin(phase)
            s.y += self.amp * 0.65 * math.cos(phase)
        return s
