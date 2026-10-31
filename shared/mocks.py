import sys
from pathlib import Path
import math
import random
from typing import List

# Add the repository root directory to sys.path so direct execution works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.contracts import (
    InputSample,
    MotionFeatures,
    FilterOutput,
    Target,
    IntentDecision,
)

def generate_mock_trajectory(
    duration_sec: float = 1.0,
    nominal_fs: float = 120.0,
    start_pos: tuple = (100.0, 100.0),
    end_pos: tuple = (500.0, 500.0),
    tremor_hz: float = 8.0,
    tremor_amp: float = 20.0
) -> List[InputSample]:
    """
    Generates a noisy sine-plus-ramp sequence to test DSP filters.
    Includes slight timestamp jitter to simulate Pygame's uneven polling.
    """
    samples = []
    num_samples = int(duration_sec * nominal_fs)
    
    for i in range(num_samples):
        # 1. Base timestamp with artificial Pygame-style jitter (+/- 2ms)
        jitter = random.uniform(-0.002, 0.002)
        t = max(0.0, (i / nominal_fs) + jitter)
        
        # 2. Intended linear motion (the ramp)
        progress = i / max(1, (num_samples - 1))
        base_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        base_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
        
        # 3. Tremor disturbance (circular/elliptical jitter)
        noise_x = math.sin(2 * math.pi * tremor_hz * t) * tremor_amp
        noise_y = math.cos(2 * math.pi * tremor_hz * t) * tremor_amp
        
        samples.append(InputSample(
            timestamp=t,
            x=base_x + noise_x,
            y=base_y + noise_y,
            source="synthetic"
        ))
    return samples

# --- EXPORTED MOCKS FOR THE TEAM ---

# 1. The input sequence for testing filters (Sine + Ramp at ~120Hz)
mock_input_sequence = generate_mock_trajectory()

# 2. Mock DSP outputs for Person 3 (ML) & Person 4 (Integration)
# CRITICAL: Person 4 must update shared/contracts.py with these new ML fields!
mock_motion_features = MotionFeatures(
    oscillation_score=0.85,
    dominant_frequency=8.0,
    periodicity_confidence=0.90,
    directional_consistency=0.30,
    velocity=400.0,
    # New expanded ML features
    acceleration=1500.0, 
    jerk=5000.0,
    spectral_entropy=0.4,
    est_amplitude=22.5
)

mock_filter_output_fixed = FilterOutput(
    x=300.0,
    y=300.0,
    alpha=None,
    mode="fixed",
    processing_time_ms=0.015
)

mock_filter_output_adaptive = FilterOutput(
    x=305.0,
    y=298.0,
    alpha=0.15,
    mode="adaptive",
    processing_time_ms=0.025
)

# 3. Mock Targets for Sandbox / Intent logic
mock_targets = [
    Target(id="target_a", center_x=500.0, center_y=500.0, width=80.0, height=80.0),
    Target(id="target_b_nearby", center_x=520.0, center_y=580.0, width=80.0, height=80.0)
]

# 4. Mock Intent Decisions for Person 4's UI
mock_intent_clear = IntentDecision(
    state="target_confident",
    target_id="target_a",
    confidence=0.92,
    assist_dx=12.5,
    assist_dy=-4.0
)

mock_intent_ambiguous = IntentDecision(
    state="target_ambiguous",
    target_id=None,
    confidence=0.45,
    assist_dx=0.0,
    assist_dy=0.0
)

if __name__ == "__main__":
    print(f"Generated {len(mock_input_sequence)} mock input samples.")
    print(f"First sample: {mock_input_sequence[0]}")
    print(f"Mock fixed output: {mock_filter_output_fixed}")