# Assisted pointing and dwell interaction

IntentGlide combines independent filtering, target-intent ranking and user assistance. Assisted pointing can use a bounded attraction toward the nearest target edge after stable intent confirmation. Target geometry has separate actual and virtual acquisition bounds: the virtual region supports candidate acquisition, while activation remains tied to the actual target or confirmed target-intent path. Sticky acquisition retains a selected target through short tremor excursions and releases after deliberate departure. Competing enlarged regions remain separate candidates; the ranker can abstain when confidence or score margin is weak.

Dwell-to-click starts disabled and requires a stable, sufficiently confident target and cursor speed. It requires three continuous seconds by default. Small X/close controls get up to 16 additional pixels of dwell acquisition only when their actual bounds have enough clearance from every other clickable control. Crowded controls receive no extra area. The enlarged zone is visible; the close operation is a reversible single-window action. Bulk close and show-desktop remain blocked from dwell activation. Brief excursions retain progress within the configured tolerance/grace; longer departure resets it. Physical click, Space, or right-click can cancel. Activation is latched until the pointer leaves the region to prevent repeats.

Useful controls:

- `G`: toggle magnetic assistance; `H`: toggle dwell; `P`: toggle personalization; `C`: start 12 instructed calibration trials.
- `F11`: export the local profile; `F12`: reset it.
- `F1/F3`: magnetic strength feedback; `F4/F5`: dwell duration feedback; `F6/F7`: target correction/correct acceptance feedback; `F8/F9`: sticky grace feedback.
- `F2`: geometry diagnostics; `1`–`5`: filter selection.

The default profile is stored locally at `data/user_profiles/default.json`. Calibration traces and session outcomes are local JSONL/CSV files. See [personalization and ML](PERSONALIZATION_AND_ML.md) for the data pipeline, schemas, commands and evaluation limits.
