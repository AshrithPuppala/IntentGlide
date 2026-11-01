"""Launch the IntentGlide Pygame sandbox.

Controls:
  1 = raw, 2 = fixed Butterworth, 3 = ML adaptive EMA,
  4 = adaptive EMA, 5 = ML adaptive Butterworth (Architecture C)
  G = toggle magnetic assistance, H = toggle 4-second dwell-to-click
  E = adaptive target frame, J = sticky target, P = personalisation
  [ / ] = magnetic strength, - / = = dwell time, C = 12-trial calibration
  F2 = target geometry debug, F1/F3–F9 = explicit preference feedback
  A = standard targets, M = ambiguous targets, B = small, L = large
  D = desktop scene, T = test-target scene
  S = switch input: synthetic <-> real mouse, V = tremor on/off (mouse input)
  Ctrl+Shift+A = manual override
  ESC / window close = quit
"""
import pygame
from pathlib import Path
from time import perf_counter

from app.orchestrator import IntentGlideOrchestrator
from app.disturbed_mouse import DisturbedMouse
from input.input_provider import InputManager, MouseProvider
from input.synthetic_generator import SyntheticInput
from interaction.dwell_controller import DwellController
from interaction.target_analyzer import MovementHistory
from interaction.user_preferences import UserPreferences
from personalization.adaptation_engine import AdaptationEngine
from personalization.calibration import CalibrationSession
from personalization.profile_manager import ProfileManager
from personalization.session_logger import SessionLogger
from personalization.user_profile import UserProfile
from sandbox.desktop import Desktop
from sandbox.scene import SandboxScene
from visualization.cursor_canvas import draw_cursor
from visualization.panel import PANEL_W, draw_side_panel
from visualization.target_debug import draw_target_diagnostics


def main():
    pygame.init()
    width, height = 1100, 700
    area_w = width - PANEL_W
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("IntentGlide — Adaptive Cursor Assistance Sandbox")
    clock = pygame.time.Clock()

    test_scene = SandboxScene(width=area_w, height=height - 48)
    desktop = Desktop(width=area_w, height=height)
    scene = desktop
    pygame.mouse.set_visible(False)
    generator = SyntheticInput(
        start_x=100, start_y=180, speed=120, tremor_freq=8, tremor_amp=7,
        bounds=(0, 0, area_w, height - 48),
    )
    disturbed_mouse = DisturbedMouse(MouseProvider())
    inputs = InputManager(synthetic=generator, mouse=disturbed_mouse)
    pipeline = IntentGlideOrchestrator(analysis_rate=120)
    profile_manager = ProfileManager()
    had_profile = profile_manager.path.exists()
    profile = profile_manager.load()
    # Move untouched v1 defaults to the shorter requested dwell while keeping
    # an explicitly adjusted dwell preference intact.
    if (profile.dwell_seconds == 4.0 and profile.adaptation_source == "default"
            and profile.calibration_trial_count == 0):
        profile.dwell_seconds = 3.0
        profile_manager.save(profile)
    # Migrate the earlier compact settings file once, without losing profile fields.
    legacy_preferences = UserPreferences()
    if not had_profile and legacy_preferences.path.exists():
        legacy_dwell = legacy_preferences.values["dwell_seconds"]
        profile.dwell_seconds = 3.0 if legacy_dwell == 4.0 else legacy_dwell
        profile.magnetic_strength = legacy_preferences.values["magnetic_profile"]
        profile_manager.save(profile)
    pipeline.apply_user_profile(profile)
    adaptation = AdaptationEngine(profile)
    session_logger = SessionLogger(profile_id=profile.profile_id)
    desktop.settings.mode = pipeline.mode
    desktop.settings.tremor_freq = generator.freq
    desktop.settings.tremor_amp = disturbed_mouse.amp
    desktop.settings.tremor_on = disturbed_mouse.enabled
    desktop.settings.set_profile_values(profile)

    def apply_desktop_setting(name, value):
        if name == "mode":
            pipeline.set_mode(value)
            dwell.reset()
        elif name == "tremor_freq":
            generator.freq = disturbed_mouse.freq = float(value)
        elif name == "tremor_amp":
            generator.amp = disturbed_mouse.amp = float(value)
        elif name == "tremor_on":
            disturbed_mouse.enabled = bool(value)
        elif name in desktop.settings.PROFILE_RANGES:
            setattr(profile, name, float(value))
            profile.acquisition_margin_px = adaptation.acquisition_margin()
            persist_profile()
        elif name == "magnetic_strength":
            profile.magnetic_strength = str(value)
            persist_profile()

    desktop.settings.subscribe(apply_desktop_setting)
    desktop.settings.connected = True
    dwell = DwellController(
        dwell_seconds=profile.dwell_seconds,
        tolerance_px=profile.dwell_tolerance_px,
        exit_grace_seconds=profile.dwell_exit_grace_ms / 1000.0,
        minimum_confidence=profile.intent_confidence_threshold,
        max_stable_speed_px_s=profile.maximum_dwell_speed_px_s,
    )
    dwell_history = MovementHistory(max_seconds=0.40, max_points=64)
    cursor_xy = (0.0, 0.0)
    running = True
    safety_override = False
    assisted_pointing_enabled = profile.magnetic_assistance_enabled
    dwell_enabled = profile.dwell_enabled
    target_enlargement_enabled = profile.adaptive_target_enlargement_enabled
    sticky_enabled = profile.sticky_target_enabled
    personalization_enabled = profile.personalization_enabled
    debug_geometry_enabled = False
    calibration = None
    last_dwell_state = "idle"
    end_to_end_latencies_ms = []

    def persist_profile():
        profile.acquisition_margin_px = adaptation.acquisition_margin()
        profile_manager.save(profile)
        desktop.settings.set_profile_values(profile)
        pipeline.apply_user_profile(profile if personalization_enabled else UserProfile())
        dwell.dwell_seconds = profile.dwell_seconds if personalization_enabled else 3.0
        dwell.tolerance_px = max(profile.dwell_tolerance_px, profile.acquisition_margin_px) if personalization_enabled else 32.0
        dwell.exit_grace_seconds = profile.dwell_exit_grace_ms/1000.0 if personalization_enabled else 0.75
        dwell.minimum_confidence = profile.intent_confidence_threshold if personalization_enabled else 0.65
        dwell.max_stable_speed_px_s = profile.maximum_dwell_speed_px_s if personalization_enabled else 75.0

    pipeline.intent.adaptive_enlargement_enabled = target_enlargement_enabled
    pipeline.intent.sticky_enabled = sticky_enabled

    while running:
        frame_started = perf_counter()
        cancel_dwell = False
        manually_selected_target = None
        manual_click_event = False
        feedback_event = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    cancel_dwell = True
                elif event.key == pygame.K_g:
                    assisted_pointing_enabled = not assisted_pointing_enabled
                    profile.magnetic_assistance_enabled = assisted_pointing_enabled
                    persist_profile()
                elif event.key == pygame.K_h:
                    dwell_enabled = not dwell_enabled
                    profile.dwell_enabled = dwell_enabled
                    persist_profile()
                    if not dwell_enabled:
                        dwell.reset()
                elif event.key == pygame.K_e:
                    target_enlargement_enabled = not target_enlargement_enabled
                    profile.adaptive_target_enlargement_enabled = target_enlargement_enabled
                    pipeline.intent.adaptive_enlargement_enabled = target_enlargement_enabled
                    persist_profile()
                    dwell.reset()
                elif event.key == pygame.K_j:
                    sticky_enabled = not sticky_enabled
                    profile.sticky_target_enabled = sticky_enabled
                    pipeline.intent.sticky_enabled = sticky_enabled
                    persist_profile()
                    if not sticky_enabled:
                        pipeline.intent.sticky.reset()
                elif event.key == pygame.K_p:
                    personalization_enabled = not personalization_enabled
                    profile.personalization_enabled = personalization_enabled
                    persist_profile()
                    dwell.reset()
                elif event.key == pygame.K_F2:
                    debug_geometry_enabled = not debug_geometry_enabled
                elif event.key == pygame.K_c and calibration is None:
                    calibration_targets = [t for t in scene.targets
                        if not (t.id.endswith(":close") or t.id.endswith("close_all")
                                or t.id == "taskbar:desktop")]
                    if calibration_targets and hasattr(scene, "set_calibration_target"):
                        calibration = CalibrationSession(profile, calibration_targets, trial_count=12)
                        inputs.set_source("mouse")
                        pipeline.reset()
                        dwell_history.reset()
                        scene.set_calibration_target(calibration.current_target_id)
                        dwell.reset()
                        session_logger.log("calibration_started", trial_count=12,
                                           input_source=inputs.active)
                elif event.key == pygame.K_F11:
                    profile_manager.export(Path("data/exports") / f"{profile.profile_id}.json", profile)
                elif event.key == pygame.K_F12:
                    profile = profile_manager.reset()
                    session_logger.profile_id = profile.profile_id
                    adaptation = AdaptationEngine(profile)
                    assisted_pointing_enabled = profile.magnetic_assistance_enabled
                    target_enlargement_enabled = profile.adaptive_target_enlargement_enabled
                    sticky_enabled = profile.sticky_target_enabled
                    personalization_enabled = profile.personalization_enabled
                    dwell_enabled = profile.dwell_enabled
                    pipeline.intent.adaptive_enlargement_enabled = target_enlargement_enabled
                    pipeline.intent.sticky_enabled = sticky_enabled
                    persist_profile()
                    dwell.reset()
                elif event.key in (pygame.K_F1, pygame.K_F3, pygame.K_F4, pygame.K_F5,
                                   pygame.K_F6, pygame.K_F7, pygame.K_F8, pygame.K_F9):
                    feedback = {
                        pygame.K_F1: "magnetic_too_strong",
                        pygame.K_F3: "magnetic_too_weak",
                        pygame.K_F4: "dwell_too_short",
                        pygame.K_F5: "dwell_too_long",
                        pygame.K_F6: "wrong_target_corrected",
                        pygame.K_F7: "correct_accepted",
                        pygame.K_F8: "sticky_too_short",
                        pygame.K_F9: "sticky_too_long",
                    }[event.key]
                    update = adaptation.apply_feedback(feedback, explicit=True)
                    feedback_event = (feedback, update.changed)
                    persist_profile()
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                    direction = -1 if event.key == pygame.K_LEFTBRACKET else 1
                    profile.magnetic_strength = pipeline.magnetic.cycle_profile(direction)
                    persist_profile()
                elif event.key in (pygame.K_MINUS, pygame.K_EQUALS):
                    delta = -0.5 if event.key == pygame.K_MINUS else 0.5
                    profile.dwell_seconds = min(8.0, max(2.0, profile.dwell_seconds + delta))
                    persist_profile()
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    pipeline.set_mode({
                        pygame.K_1: "raw",
                        pygame.K_2: "fixed",
                        pygame.K_3: "adaptive",
                        pygame.K_4: "full",
                        pygame.K_5: "ml_butterworth",
                    }[event.key])
                    dwell.reset()
                    dwell_history.reset()
                elif event.key == pygame.K_a:
                    mods = pygame.key.get_mods()
                    if (mods & pygame.KMOD_CTRL) and (mods & pygame.KMOD_SHIFT):
                        safety_override = not safety_override
                        dwell.reset()
                    else:
                        scene.set_layout("standard")
                elif event.key == pygame.K_d:
                    calibration = None
                    scene = desktop
                    dwell.reset()
                    dwell_history.reset()
                elif event.key == pygame.K_t:
                    calibration = None
                    scene = test_scene
                    dwell.reset()
                    dwell_history.reset()
                elif event.key == pygame.K_s:
                    inputs.set_source("mouse" if inputs.active == "synthetic" else "synthetic")
                    pipeline.reset()
                    dwell.reset()
                    dwell_history.reset()
                    if calibration is not None:
                        session_logger.log("calibration_cancelled", reason="input source changed")
                        calibration = None
                        desktop.set_calibration_target(None)
                elif event.key == pygame.K_v:
                    disturbed_mouse.enabled = not disturbed_mouse.enabled
                elif event.key == pygame.K_m:
                    scene.set_layout("ambiguous")
                elif event.key == pygame.K_b:
                    scene.set_layout("small")
                elif event.key == pygame.K_l:
                    scene.set_layout("large")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    cancel_dwell = True
                elif event.button == 1:
                    cancel_dwell = True
                    manual_click_event = True
                    if hasattr(scene, "click"):
                        manually_selected_target = scene.click(*cursor_xy)

        sample = inputs.get_sample()
        scene_width = getattr(scene, "width", area_w)
        scene_height = getattr(scene, "height", height)
        inside_scene = 0 <= sample.x < scene_width and 0 <= sample.y < scene_height
        reachable_targets = scene.targets if inside_scene else []
        result = pipeline.process(sample, targets=reachable_targets, override=safety_override,
                                  assist_enabled=assisted_pointing_enabled and inside_scene)
        if feedback_event:
            session_logger.log("explicit_feedback", monotonic_time=sample.timestamp,
                               feedback=feedback_event[0], changed=feedback_event[1])
        if cancel_dwell:
            session_logger.log("dwell_cancel_requested", monotonic_time=sample.timestamp,
                               source="space" if pygame.key.get_pressed()[pygame.K_SPACE] else "right_click",
                               adaptation="record_only")
        if manual_click_event:
            session_logger.log("physical_click", monotonic_time=sample.timestamp,
                target_id=manually_selected_target,
                intended_target_id=calibration.current_target_id if calibration else None)
        raw, output = result["raw"], result["output"]
        cursor_xy = (output.x, output.y)
        if calibration is not None:
            diag = pipeline.intent.diagnostics
            motion = result["features"]
            calibration.record_sample(timestamp=sample.timestamp, raw_x=raw.x, raw_y=raw.y,
                filtered_x=result["filtered_output"].x, filtered_y=result["filtered_output"].y,
                velocity_x=diag.get("velocity_x", 0.0), velocity_y=diag.get("velocity_y", 0.0),
                magnetic_strength=profile.magnetic_strength,
                acquisition_margin_px=diag.get("acquisition_margin_px", profile.acquisition_margin_px),
                oscillation_score=motion.oscillation_score,
                dominant_frequency_hz=motion.dominant_frequency or 0.0,
                estimated_tremor_amplitude_px=motion.est_amplitude or 0.0,
                acceleration_px_s2=motion.acceleration or 0.0,
                assisted_x=output.x, assisted_y=output.y,
                magnetic_dx=pipeline.magnetic.offset_x, magnetic_dy=pipeline.magnetic.offset_y)
        if inside_scene and not safety_override:
            dwell_history.add(sample.timestamp, *cursor_xy)
            dwell_cursor_speed = dwell_history.estimate().speed
        else:
            dwell_history.reset()
            dwell_cursor_speed = 0.0
        scene.add_positions((raw.x, raw.y), cursor_xy)
        if hasattr(scene, "hover"):
            scene.hover(*cursor_xy)
        decision = result.get("decision")
        intent_target_id = (decision.target_id if decision is not None
                            and decision.state == "target_confident" else None)
        acquisition_margin = pipeline.intent.diagnostics.get("acquisition_margin_px", profile.acquisition_margin_px)
        dwell_base_tolerance = max(profile.dwell_tolerance_px, acquisition_margin) if personalization_enabled else 32.0
        exact_target_id = (scene.target_at(*cursor_xy, margin=0.0)
                           if hasattr(scene, "target_at") else None)
        virtual_candidate_id = (scene.target_at(*cursor_xy, margin=acquisition_margin)
                               if hasattr(scene, "target_at") else None)
        # Virtual bounds only make a widget eligible. They cannot replace a
        # confirmed intent ranking or trigger activation by themselves.
        actual_target_id = exact_target_id or (
            virtual_candidate_id if intent_target_id == virtual_candidate_id else None
        )
        actual_target = next((target for target in scene.targets
                              if target.id == actual_target_id), None)
        confidence = decision.confidence if decision is not None else 0.0
        dwell_target = next((target for target in scene.targets
                             if target.id == intent_target_id), None)
        dwell_confidence = confidence
        # A stable hit in the adaptive virtual frame is direct pointer evidence.
        # Overlapping candidates are rejected by Desktop.target_at; the real
        # widget bounds remain unchanged for click activation.
        if actual_target_id is not None and not safety_override:
            dwell_target = actual_target
            dwell_confidence = max(dwell_confidence, dwell.minimum_confidence)
        dwell_enlargement_px = (scene.dwell_enlargement_for(dwell_target.id)
            if dwell_enabled and dwell_target is not None
            and hasattr(scene, "dwell_enlargement_for") else 0.0)
        dwell.tolerance_px = dwell_base_tolerance + dwell_enlargement_px
        dwell_status = dwell.update(
            now=sample.timestamp,
            target=dwell_target,
            confidence=dwell_confidence,
            cursor_x=cursor_xy[0],
            cursor_y=cursor_xy[1],
            filtered_speed_px_s=dwell_cursor_speed,
            enabled=dwell_enabled and not safety_override and inside_scene,
            override=safety_override,
            cancel_requested=cancel_dwell,
            allow_virtual_activation=dwell_enlargement_px > 0,
        )
        if dwell_status.activated and dwell_status.target_id:
            if hasattr(scene, "activate_target"):
                activated = scene.activate_target(dwell_status.target_id, source="dwell")
            elif hasattr(scene, "click"):
                activated = scene.click(*cursor_xy) is not None
            else:
                activated = False
            if not activated:
                dwell_status = dwell.mark_activation_blocked(
                    "This action needs an explicit click"
                )
            session_logger.log("dwell_activation", monotonic_time=sample.timestamp,
                target_id=dwell_status.target_id, activated=bool(activated),
                result_state=dwell_status.state)
            if calibration is not None and activated:
                manually_selected_target = dwell_status.target_id
        if calibration is not None and manual_click_event:
            try:
                calibration.finish_trial(manually_selected_target,
                    dwell_outcome="activated" if dwell_status.activated else "physical_click")
                completed, total = calibration.progress
                session_logger.log("calibration_trial", monotonic_time=sample.timestamp,
                    trial_index=completed, intended_target_id=(calibration.trials[-1].intended_target_id),
                    actual_selected_target_id=manually_selected_target)
                if calibration.complete:
                    summary = calibration.summary()
                    adaptation.apply_calibration(summary)
                    profile_manager.save(profile)
                    trace_base = Path("data/calibration_sessions") / calibration.session_id
                    calibration.export_jsonl(trace_base.with_suffix(".jsonl"))
                    calibration.export_intent_csv(trace_base.with_name(trace_base.name + "_intent.csv"))
                    session_logger.log("calibration_completed", monotonic_time=sample.timestamp,
                                       summary=summary, data_source="calibration")
                    calibration = None
                    scene.set_calibration_target(None)
                    persist_profile()
                else:
                    scene.set_calibration_target(calibration.current_target_id)
            except ValueError as exc:
                session_logger.log("calibration_trial_rejected", monotonic_time=sample.timestamp,
                                   reason=str(exc))
        if hasattr(scene, "set_intent_feedback"):
            diagnostics = pipeline.intent.diagnostics
            preview_id = (dwell_status.target_id if dwell_status.state in {"dwelling", "ready", "cancelled", "activated", "blocked"}
                          else intent_target_id or diagnostics.get("candidate_id"))
            scene.set_intent_feedback(preview_id, dwell_status.progress,
                                      dwell_status.state in {"dwelling", "ready"},
                                      dwell_enlargement_px if preview_id == (dwell_target.id if dwell_target else None) else 0.0)
        scene.draw(screen, None)
        if debug_geometry_enabled:
            diagnostics = pipeline.intent.diagnostics
            draw_target_diagnostics(screen, reachable_targets, acquisition_margin,
                dwell_status.target_id or intent_target_id, cursor_xy,
                (diagnostics.get("velocity_x", 0.0), diagnostics.get("velocity_y", 0.0)),
                (pipeline.magnetic.offset_x, pipeline.magnetic.offset_y))
        draw_side_panel(screen, pygame.Rect(area_w, 0, PANEL_W, height), {
            "mode": pipeline.mode,
            "features": result["features"],
            "decision": result["decision"],
            "override": result["override"],
            "filter_source": result["filter_source"],
            "cutoff_hz": result["cutoff_hz"],
            "processing_ms": result["processing_time_ms"],
            "source": inputs.active,
            "tremor": disturbed_mouse.enabled,
            "assist_enabled": assisted_pointing_enabled,
            "dwell_enabled": dwell_enabled,
            "dwell_status": dwell_status,
            "dwell_seconds": dwell.dwell_seconds,
            "dwell_enlargement_px": dwell_enlargement_px,
            "acquisition_margin_px": acquisition_margin,
            "intent_diagnostics": pipeline.intent.diagnostics,
            "intent_source": result.get("intent_source", "heuristic_fallback"),
            "inside_scene": inside_scene,
            "magnetic_profile": result.get("magnetic_profile", "balanced"),
            "target_lock": result.get("target_lock"),
            "magnetic_displacement_px": result.get("magnetic_displacement_px", 0.0),
            "personalization_source": result.get("personalization_source", "default"),
            "target_enlargement_enabled": target_enlargement_enabled,
            "sticky_enabled": sticky_enabled,
            "personalization_enabled": personalization_enabled,
            "calibration": calibration,
        })
        draw_cursor(screen, cursor_xy)
        end_to_end_latencies_ms.append((perf_counter()-frame_started)*1000.0)
        del end_to_end_latencies_ms[:-1200]
        pygame.display.flip()
        clock.tick(120)

    sorted_latency = sorted(end_to_end_latencies_ms)
    p50 = sorted_latency[int(.50*(len(sorted_latency)-1))] if sorted_latency else None
    p95 = sorted_latency[int(.95*(len(sorted_latency)-1))] if sorted_latency else None
    session_logger.log("session_ended", monotonic_time=sample.timestamp if "sample" in locals() else None,
                       calibration_incomplete=calibration is not None,
                       end_to_end_latency_p50_ms=p50, end_to_end_latency_p95_ms=p95)
    pygame.quit()


if __name__ == "__main__":
    main()
