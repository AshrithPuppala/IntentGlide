"""Format motion and controller diagnostics for the dashboard."""
def format_diagnostics(features, decision=None, processing_time_ms=None):
    rows = [
        f"Oscillation score: {features.oscillation_score:.2f}",
        f"Dominant frequency: {features.dominant_frequency if features.dominant_frequency is not None else '—'}",
        f"Periodicity confidence: {features.periodicity_confidence:.2f}",
        f"Directional consistency: {features.directional_consistency:.2f}",
        f"Velocity: {features.velocity:.1f} px/s",
    ]
    if decision is not None:
        rows += [
            f"Controller: {decision.state}",
            f"Target confidence: {decision.confidence:.2f}",
        ]
    if processing_time_ms is not None:
        rows.append(f"Processing: {processing_time_ms:.3f} ms")
    return rows
