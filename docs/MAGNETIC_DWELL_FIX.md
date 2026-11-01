# IntentGlide magnetic assistance + tremor-tolerant dwell fix

## What this patch addresses

1. Magnetic assistance computes distance to the nearest edge of a target rectangle, not distance to the target centre.
2. Attraction is time-based and rate-limited in px/s. It never teleports the cursor and stops once the pointer is inside the real target rectangle.
3. Attraction is gated by an already-confirmed target. Do not lower intent confidence thresholds globally to force attraction; fix candidate geometry/ranking and inspect score margin first.
4. Dwell defaults to **3.0 seconds** and can be adjusted from 2–8 seconds.
5. Dwell considers an adaptive expanded region around the actual target (32 px minimum; up to 48 px when measured tremor is higher).
6. Brief excursions within the configured tolerance/grace retain active progress without introducing a paused state. If the cursor remains outside / unstable longer than `exit_grace_seconds` (750 ms default), progress resets.
7. Completed or cancelled targets remain latched until the cursor leaves the expanded region, preventing repeated clicks.

## Install

Copy `interaction/magnetic_assistance.py` and `interaction/dwell_controller.py` into the repository's `interaction/` directory. Merge the two test files into `tests/`.

Run:

```cmd
python -m pytest -q tests/test_magnetic_assistance.py tests/test_dwell_controller.py
```

## Magnetic assistance integration

In the orchestrator, first run the selected filter. Independently run intent mapping regardless of whether the selected filter is `adaptive` or `ml_butterworth`. Look up the `Target` matching `decision.target_id` and call:

```python
from interaction.magnetic_assistance import apply_magnetic_assistance

output = apply_magnetic_assistance(
    selected_filter_output,
    selected_target,
    dt=sample.timestamp - previous_timestamp,
    intent_confirmed=(decision is not None and decision.state == "target_confident"),
    confidence=decision.confidence if decision else 0.0,
    manual_override=override,
)
```

Only call after a temporally stable intent decision. The helper requires `intent_confirmed=True` and confidence >= 0.70 by default. The user's screenshot showed `CONFIRMING - SETTINGS`, score 0.39, lead 0.02, persistence 0 ms; those values are not yet sufficient to attract. Correct target ranking/confirmation first; do not simply bypass ambiguity checks.

## Dwell integration

Call `DwellController.update()` once per frame. Pass the *confirmed* target object, cursor position after filtering + magnetic assistance, current confidence, and velocity estimated from filtered movement history. Use the same monotonic `InputSample.timestamp` clock for every update.

```python
from interaction.dwell_controller import DwellController

    dwell = DwellController(dwell_seconds=3.0, tolerance_px=32,
                         exit_grace_seconds=0.75)

state = dwell.update(
    now=sample.timestamp,
    target=selected_target,
    confidence=decision.confidence if decision else 0.0,
    cursor_x=output.x,
    cursor_y=output.y,
    filtered_speed_px_s=filtered_speed,
    enabled=dwell_enabled,
    override=override,
    cancel_requested=space_cancel or right_click_cancel,
)

if state.activated:
    # Semantic activation, exactly once, after checking that the target is
    # still reachable and explicitly allowed for dwell activation.
    activated = desktop.activate_target(state.target_id)
    if not activated:
        dwell.reset()
```

`Desktop.activate_target` must validate that the target still exists and is dwell-safe. Keep destructive / irreversible actions out of the dwell allow-list. A failed activation must not be silently reported as successful.

## Debugging the screenshot

The screenshot showed `CONFIRMING - SETTINGS`, score `0.39`, lead `0.02`, persistence `0 ms`. That means there is a candidate, but the intent controller has not confirmed it. Magnetic assistance is intentionally gated off. The pointer looked close to the right edge of the Settings icon, so ranking should use distance to the target rectangle / acquisition region rather than only centre distance. Keep ambiguity abstention: if two nearby targets remain similarly ranked, do not pull toward either.

## Tuning order

1. Confirm the correct target ID and stable lead over competitors.
2. Check acquisition-region geometry and edge distance.
3. Tune `acquisition_radius_px` (e.g. 60–80 px) and `max_speed_px_s` (e.g. 100–140 px/s) in controlled tests.
4. Tune dwell `tolerance_px` (e.g. 20–30 px) and grace (e.g. 500–800 ms) using actual cursor traces.
5. Measure false attraction, wrong-target selection, successful activations, cancellation and time-to-acquire. Never tune by visual smoothness alone.

The default values are engineering starting points, not clinically validated parameters.
