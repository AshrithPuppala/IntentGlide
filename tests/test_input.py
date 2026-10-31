import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # no real window needed
import pygame
from input.synthetic_generator import SyntheticInput
from input.input_provider import MouseProvider, InputManager

def test_synthetic_monotonic_and_in_bounds():
    gen = SyntheticInput(speed=500)
    prev = -1.0
    for _ in range(50):
        s = gen.get_sample()
        assert s.timestamp > prev
        assert -20 <= s.x <= 1300 and -20 <= s.y <= 740   # bounds +/- amp
        prev = s.timestamp

def test_synthetic_frequency():
    # Fake clock at exactly 1 kHz so the FFT peak is verifiable
    import numpy as np
    t = [0.0]
    gen = SyntheticInput(speed=0, tremor_freq=8.0, clock=lambda: t[0])
    xs = []
    for _ in range(2000):
        t[0] += 0.001
        xs.append(gen.get_sample().x)
    spec = np.abs(np.fft.rfft(np.array(xs) - np.mean(xs)))
    freqs = np.fft.rfftfreq(len(xs), d=0.001)
    assert abs(freqs[spec.argmax()] - 8.0) < 0.5

def test_manager_switch():
    pygame.init(); pygame.display.set_mode((200, 200)); pygame.event.pump()
    m = InputManager(synthetic=SyntheticInput(), mouse=MouseProvider())
    assert m.get_sample().source == "synthetic"
    m.set_source("mouse")
    assert m.get_sample().source == "mouse"