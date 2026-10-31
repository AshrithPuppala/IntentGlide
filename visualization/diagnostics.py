"""Format motion and controller diagnostics for the dashboard."""


def format_diagnostics(features, decision=None, processing_time_ms=None,
                       filter_source=None, cutoff_hz=None):
    freq = features.dominant_frequency
    rows = [
        f"Oscillation score: {features.oscillation_score:.2f}",
        f"Dominant frequency: {f'{freq:.1f} Hz' if freq is not None else '—'}",
        f"Periodicity confidence: {features.periodicity_confidence:.2f}",
        f"Directional consistency: {features.directional_consistency:.2f}",
        f"Velocity: {features.velocity:.0f} px/s",
    ]
    if filter_source is not None:
        cutoff = f" | cutoff {cutoff_hz:.1f} Hz" if cutoff_hz is not None else ""
        rows.append(f"Filter policy: {filter_source}{cutoff}")
    if decision is not None:
        rows += [
            f"Controller: {decision.state}",
            f"Target confidence: {decision.confidence:.2f}",
        ]
    if processing_time_ms is not None:
        rows.append(f"Processing: {processing_time_ms:.3f} ms")
    return rows