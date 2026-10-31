"""Launch the IntentGlide Pygame sandbox.

Controls:
  1 = raw, 2 = fixed, 3 = adaptive, 4 = full system
  A = standard targets, M = ambiguous targets
  ESC / window close = quit
"""
import pygame

from app.orchestrator import IntentGlideOrchestrator
from input.synthetic_generator import SyntheticInputGenerator
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
    generator = SyntheticInputGenerator(
        sample_rate=120, disturbance_frequency=8, amplitude=7
    )
    pipeline = IntentGlideOrchestrator(analysis_rate=120)
    sample_index = 0
    running = True

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
                    scene.set_layout("standard")
                elif event.key == pygame.K_m:
                    scene.set_layout("ambiguous")

        # Synthetic input is the hardware-independent demonstration fallback.
        sample = generator.sample(sample_index)
        sample_index += 1
        result = pipeline.process(sample, targets=scene.targets)
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
