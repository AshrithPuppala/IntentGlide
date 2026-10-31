"""Launch the IntentGlide Pygame sandbox.

Controls:
  1 = raw, 2 = fixed, 3 = adaptive, 4 = full system
  A = standard targets, M = ambiguous targets
  ESC / window close = quit
"""
import pygame

from app.orchestrator import IntentGlideOrchestrator
from input.synthetic_generator import SyntheticInput
from sandbox.scene import SandboxScene
from visualization.dashboard import draw_dashboard
from visualization.diagnostics import format_diagnostics


def main():
    pygame.init()
    width, height = 1100, 700
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("IntentGlide — Adaptive Cursor Assistance Sandbox")
    clock = pygame.time.Clock()

    scene = SandboxScene(width=width, height=600)
    generator = SyntheticInput(
    start_x=100,
    start_y=180,
    speed=120,
    tremor_freq=8,
    tremor_amp=7,
    bounds=(0, 0, width, 600),
    )
    pipeline = IntentGlideOrchestrator(analysis_rate=120)
    running = True
    debug_counter = 0
    safety_override = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    pipeline.mode = {
                        pygame.K_1: "raw",
                        pygame.K_2: "fixed",
                        pygame.K_3: "adaptive",
                        pygame.K_4: "full",
                    }[event.key]
                elif event.key == pygame.K_a:
                    mods = pygame.key.get_mods()

                    if (mods & pygame.KMOD_CTRL) and (mods & pygame.KMOD_SHIFT):
                        safety_override = not safety_override
                    else:
                        scene.set_layout("standard")
                elif event.key == pygame.K_m:
                    scene.set_layout("ambiguous")
                elif event.key == pygame.K_b:
                    scene.set_layout("small")
                elif event.key == pygame.K_l:
                    scene.set_layout("large") # Synthetic input is the hardware-independent demonstration fallback.
        sample = generator.get_sample()
        result = pipeline.process(
            sample,
            targets=scene.targets,
            override=safety_override
        )
        if debug_counter % 120 == 0:
            print(
                f"Oscillation: {result['features'].oscillation_score:.3f} | "
                f"Frequency: {result['features'].dominant_frequency} | "
                f"Periodicity: {result['features'].periodicity_confidence:.3f} | "
                f"Alpha: {result['adaptive'].alpha:.3f} | "
                f"Fixed: ({result['fixed'].x:.1f}, {result['fixed'].y:.1f}) | "
                f"Adaptive: ({result['adaptive'].x:.1f}, {result['adaptive'].y:.1f})"
                f"Safety Override: {'ON' if safety_override else 'OFF'}",
            )
        debug_counter += 1
        raw, output = result["raw"], result["output"]
        scene.add_positions((raw.x, raw.y), (output.x, output.y))
        scene.draw(screen, (output.x, output.y))

        rows = [
            f"Mode: {pipeline.mode.upper()} | 1 Raw  2 Fixed  3 Adaptive  4 Full",
            "A Standard targets | M Ambiguous targets | ESC Quit",
        ]
        rows += format_diagnostics(
            result["features"],
            result["decision"],
            result["processing_time_ms"],
        )
        draw_dashboard(screen, rows)
        pygame.display.flip()
        clock.tick(120)

    pygame.quit()


if __name__ == "__main__":
    main()
