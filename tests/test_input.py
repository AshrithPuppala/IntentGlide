from input.synthetic_generator import SyntheticInputGenerator


def test_synthetic_generator_is_deterministic():
    a = list(SyntheticInputGenerator(start_time=5.0).generate(10))
    b = list(SyntheticInputGenerator(start_time=5.0).generate(10))
    assert [(s.x, s.y, s.timestamp) for s in a] == [
        (s.x, s.y, s.timestamp) for s in b
    ]


def test_synthetic_generator_uses_120_hz_clock():
    samples = list(SyntheticInputGenerator(start_time=2.0).generate(3))
    assert abs((samples[1].timestamp - samples[0].timestamp) - 1 / 120) < 1e-9
