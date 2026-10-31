from sandbox.models import BenchmarkModel, SettingsModel, format_metric


def test_settings_clamp_and_notify():
    m, seen = SettingsModel(), []
    m.subscribe(lambda n, v: seen.append((n, v)))
    m.set_mode("raw"); m.set_mode("raw"); m.set_mode("bogus")
    for _ in range(50):
        m.step_freq(+1)
    for _ in range(50):
        m.step_amp(-1)
    assert m.mode == "raw" and m.tremor_freq == m.FREQ_RANGE[1] and m.tremor_amp == m.AMP_RANGE[0]
    assert seen[0] == ("mode", "raw") and sum(1 for n, _ in seen if n == "mode") == 1


def test_benchmark_model_best_and_csv(tmp_path):
    b = BenchmarkModel()
    assert not b.has_data and b.source == "none" and format_metric("success_rate", None) == "-"
    p = tmp_path / "r.csv"
    p.write_text("mode,success_rate,median_time_s\nraw,0.5,1.4\nfull,0.9,1.1\n")
    assert b.load_csv(str(p)) == 2 and b.source == "simulated"
    assert b.best_mode("success_rate") == "full" and b.best_mode("median_time_s") == "full"
    assert b.best_mode("assist_rate") is None
    assert format_metric("success_rate", 0.9) == "90%"
