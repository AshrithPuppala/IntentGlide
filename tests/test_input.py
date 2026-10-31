import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from input.synthetic_generator import SyntheticInput
from input.input_provider import MouseProvider, InputManager
from input.synthetic_generator import SyntheticInput


def make_generator():
    current_time = [5.0]

    def clock():
        return current_time[0]

    generator = SyntheticInput(clock=clock)

    def next_sample():
        current_time[0] += 1 / 120
        return generator.get_sample()

    return next_sample


def test_synthetic_generator_is_deterministic():
    a = make_generator()
    b = make_generator()

    samples_a = [a() for _ in range(10)]
    samples_b = [b() for _ in range(10)]

    assert [(s.x, s.y, s.timestamp) for s in samples_a] == [
        (s.x, s.y, s.timestamp) for s in samples_b
    ]


def test_synthetic_generator_uses_120_hz_clock():
    next_sample = make_generator()
    samples = [next_sample() for _ in range(3)]

    assert abs(
        (samples[1].timestamp - samples[0].timestamp) - 1 / 120
    ) < 1e-9
def test_synthetic_monotonic_and_in_bounds():
    gen = SyntheticInput(speed=500)
    prev = -1.0

    for _ in range(50):
        s = gen.get_sample()
        assert s.timestamp > prev
        assert -20 <= s.x <= 1300
        assert -20 <= s.y <= 740
        prev = s.timestamp


def test_synthetic_frequency():
    import numpy as np

    t = [0.0]
    gen = SyntheticInput(
        speed=0,
        tremor_freq=8.0,
        clock=lambda: t[0]
    )
    xs = []

    for _ in range(2000):
        t[0] += 0.001
        xs.append(gen.get_sample().x)

    spec = np.abs(np.fft.rfft(np.array(xs) - np.mean(xs)))
    freqs = np.fft.rfftfreq(len(xs), d=0.001)

    assert abs(freqs[spec.argmax()] - 8.0) < 0.5


def test_manager_switch():
    pygame.init()
    pygame.display.set_mode((200, 200))
    pygame.event.pump()

    m = InputManager(
        synthetic=SyntheticInput(),
        mouse=MouseProvider()
    )

    assert m.get_sample().source == "synthetic"

    m.set_source("mouse")
    assert m.get_sample().source == "mouse"

    pygame.quit()