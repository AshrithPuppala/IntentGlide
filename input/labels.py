import math
from typing import List, Optional
from shared.contracts import GroundTruthLabel, GroundTruth, Target

AMBIG_DEG = 15.0   # tune: bearing gap that counts as "ambiguous"

def _bearing(x0, y0, t: Target) -> float:
    return math.degrees(math.atan2(t.center_y - y0, t.center_x - x0))

def _rival(reach, candidates: List[Target]) -> Optional[str]:
    own = _bearing(reach.x0, reach.y0, reach.target)
    best, best_gap = None, AMBIG_DEG
    for c in candidates:
        if c.id == reach.target.id:
            continue
        gap = abs((_bearing(reach.x0, reach.y0, c) - own + 180) % 360 - 180)
        if gap < best_gap:
            best, best_gap = c.id, gap
    return best

def make_label(reach, truth: GroundTruth, trial_id: int,
               candidates: List[Target], override: bool = False) -> GroundTruthLabel:
    p = truth.progress
    phase = ("onset" if p < 0.1 else "transit" if p < 0.7
             else "approach" if p < 1.0 else "dwell")
    rival = _rival(reach, candidates)
    ambiguous = rival is not None and p < 0.7
    tremor = reach.amp > 0

    # Labeling rules (document any changes to these for the team)
    if override:                      state = "override"
    elif ambiguous:                   state = "target_ambiguous"
    elif phase in ("onset", "dwell") and tremor:
                                      state = "oscillation_detected"
    elif phase in ("transit", "approach"):
                                      state = "target_confident"
    else:                             state = "stable"

    t = reach.target
    return GroundTruthLabel(
        trial_id=trial_id, timestamp=truth.timestamp,
        intended_target_id=t.id, true_state=state, phase=phase,
        progress=p, truth_x=truth.x, truth_y=truth.y,
        dist_to_target=math.hypot(t.center_x - truth.x, t.center_y - truth.y),
        tremor_active=tremor, tremor_freq=reach.freq, tremor_amp=reach.amp,
        ambiguous=ambiguous, rival_target_id=rival, override=override)