"""Launch the IntentGlide Pygame sandbox.

Controls:
  1 = raw, 2 = fixed Butterworth, 3 = ML adaptive EMA,
  4 = full intent-assisted system, 5 = ML adaptive Butterworth (Architecture C)
  A = standard targets, M = ambiguous targets, B = small, L = large
  D = desktop scene, T = test-target scene
  S = switch input: synthetic <-> real mouse, V = tremor on/off (mouse input)
  Ctrl+Shift+A = manual override
  SPACE = start a trial (reach for the highlighted calculator key and click it)
  R = auto-reach trials (SyntheticReach drives the cursor, trials chain), BACKSPACE = clear trial stats
  ESC / window close = quit
"""
import pygame

from app.orchestrator import IntentGlideOrchestrator
from app.disturbed_mouse import DisturbedMouse
from app.trial_session import TrialSession
from input.input_provider import InputManager, MouseProvider
from input.synthetic_generator import SyntheticInput
from sandbox.desktop import Desktop
from sandbox.scene import SandboxScene
from visualization.cursor_canvas import draw_cursor
from visualization.panel import PANEL_W, draw_side_panel
from visualization.trial_hud import draw_trial_hud, hud_rect


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
    cursor_xy = (0.0, 0.0)
    running = True
    safety_override = False
    trials = TrialSession(desktop, inputs, on_reset=pipeline.reset)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    trials.cancel()                  # a trial belongs to one mode; start a new one
                    pipeline.set_mode({
                        pygame.K_1: "raw",
                        pygame.K_2: "fixed",
                        pygame.K_3: "adaptive",
                        pygame.K_4: "full",
                        pygame.K_5: "ml_butterworth",
                    }[event.key])
                elif event.key == pygame.K_a:
                    mods = pygame.key.get_mods()
                    if (mods & pygame.KMOD_CTRL) and (mods & pygame.KMOD_SHIFT):
                        safety_override = not safety_override
                    else:
                        scene.set_layout("standard")
                elif event.key == pygame.K_d:
                    scene = desktop
                elif event.key == pygame.K_t:
                    trials.stop_auto()               # trials need the desktop's calculator
                    scene = test_scene
                elif event.key == pygame.K_s:
                    trials.stop_auto()
                    inputs.set_source("mouse" if inputs.active == "synthetic" else "synthetic")
                    pipeline.reset()
                elif event.key == pygame.K_SPACE:
                    scene = desktop
                    trials.auto = False
                    if inputs.active != "mouse":
                        inputs.set_source("mouse")   # a manual trial is driven by the (disturbed) mouse
                        pipeline.reset()
                    trials.start(pipeline.mode, cursor_xy)
                elif event.key == pygame.K_r:
                    scene = desktop
                    if trials.auto:
                        trials.stop_auto()
                    else:
                        trials.auto = True
                        trials.start(pipeline.mode, cursor_xy)
                elif event.key == pygame.K_BACKSPACE:
                    trials.clear()
                elif event.key == pygame.K_v:
                    disturbed_mouse.enabled = not disturbed_mouse.enabled
                elif event.key == pygame.K_m:
                    scene.set_layout("ambiguous")
                elif event.key == pygame.K_b:
                    scene.set_layout("small")
                elif event.key == pygame.K_l:
                    scene.set_layout("large")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if hasattr(scene, "click"):
                    scene.click(*cursor_xy)
                trials.click(cursor_xy)              # scored on the processed cursor, like the click itself

        sample = inputs.get_sample()
        result = pipeline.process(sample, targets=scene.targets, override=safety_override)
        raw, output = result["raw"], result["output"]
        cursor_xy = (output.x, output.y)
        scene.add_positions((raw.x, raw.y), cursor_xy)
        if hasattr(scene, "hover"):
            scene.hover(*cursor_xy)
        trials.update(cursor_xy, pipeline.mode)      # auto click, timeout, chaining the next trial
        scene.draw(screen, None)
        if scene is desktop:
            draw_trial_hud(screen, hud_rect(area_w), trials.view(pipeline.mode))
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
        })
        draw_cursor(screen, cursor_xy)
        pygame.display.flip()
        clock.tick(120)

    pygame.quit()


if __name__ == "__main__":
    main()
