"""Stateful second-order Butterworth low-pass baseline."""
from __future__ import annotations

from scipy.signal import butter, sosfilt, sosfilt_zi

from shared.contracts import FilterOutput, InputSample


class FixedButterworthFilter:
    def __init__(self, cutoff_hz: float = 3.0, sample_rate_hz: float = 120.0):
        if not 0 < cutoff_hz < sample_rate_hz / 2:
            raise ValueError("cutoff must be between 0 and Nyquist frequency")
        self.cutoff_hz = float(cutoff_hz)
        self.sample_rate_hz = float(sample_rate_hz)
        self.sos = butter(2, self.cutoff_hz, btype="lowpass",
                          fs=self.sample_rate_hz, output="sos")
        self._zi_x = None
        self._zi_y = None

    def reset(self):
        self._zi_x = None
        self._zi_y = None

    def update(self, sample: InputSample) -> FilterOutput:
        if self._zi_x is None:
            self._zi_x = sosfilt_zi(self.sos) * sample.x
            self._zi_y = sosfilt_zi(self.sos) * sample.y
        out_x, self._zi_x = sosfilt(self.sos, [sample.x], zi=self._zi_x)
        out_y, self._zi_y = sosfilt(self.sos, [sample.y], zi=self._zi_y)
        return FilterOutput(float(out_x[0]), float(out_y[0]), mode="fixed")
