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