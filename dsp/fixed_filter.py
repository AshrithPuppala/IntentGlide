import math
import time
from shared.contracts import InputSample, FilterOutput

class _Biquad:
    """One axis, transposed direct form II, fixed coefficients."""
    def __init__(self, b0: float, b1: float, b2: float, a1: float, a2: float):
        self.b0, self.b1, self.b2 = b0, b1, b2
        self.a1, self.a2 = a1, a2
        self.z1 = self.z2 = 0.0
        self.initialized = False

    def step(self, x: float) -> float:
        if not self.initialized:
            # Start at steady state: no startup transient
            self.z1 = (1 - self.b0) * x
            self.z2 = (self.b2 - self.a2) * x
            self.initialized = True
            
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y

class FixedFilter:
    def __init__(self, cutoff_hz=3.0, nominal_fs=120.0):
        self.fc = cutoff_hz
        self.fs = nominal_fs
        
        # Precompute coefficients once
        K = math.tan(math.pi * self.fc / self.fs)
        Q = 1 / math.sqrt(2)
        n = 1 / (1 + K / Q + K * K)
        
        b0 = K * K * n
        b1 = 2 * b0
        b2 = b0
        a1 = 2 * (K * K - 1) * n
        a2 = (1 - K / Q + K * K) * n
        
        self.fx = _Biquad(b0, b1, b2, a1, a2)
        self.fy = _Biquad(b0, b1, b2, a1, a2)

    def reset(self):
        """Reset the filter state."""
        self.fx.initialized = False
        self.fy.initialized = False
        self.fx.z1 = self.fx.z2 = 0.0
        self.fy.z1 = self.fy.z2 = 0.0

    def update(self, s: InputSample) -> FilterOutput:
        # Measure processing time to satisfy the new evaluation metrics
        t0 = time.perf_counter()
        
        out_x = self.fx.step(s.x)
        out_y = self.fy.step(s.y)
        
        t1 = time.perf_counter()
        processing_time_ms = (t1 - t0) * 1000.0
        
        return FilterOutput(
            x=out_x,
            y=out_y,
            alpha=None, 
            mode="fixed",
            processing_time_ms=processing_time_ms
        )

if __name__ == "__main__":
    # Self-test
    f = FixedFilter()
    sample = InputSample(timestamp=1.0, x=100.0, y=200.0, source="synthetic")
    print(f.update(sample))