import pygame
from input.input_provider import MouseProvider
from dsp.motion_analyzer import MotionAnalyzer
from dsp.adaptive_filter import AdaptiveFilter
from dsp.fixed_filter import FixedFilter

pygame.init()
font = pygame.font.SysFont("Consolas", 16)
screen = pygame.display.set_mode((900, 650))
pygame.display.set_caption("IntentGlide DSP Live Demo: Raw vs Fixed vs Adaptive")
clock = pygame.time.Clock()

mouse = MouseProvider()
analyzer = MotionAnalyzer()
adaptive = AdaptiveFilter()
fixed = FixedFilter()

prev = None
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    s = mouse.get_sample()
    dt = (s.timestamp - prev.timestamp) if prev else 0.008
    prev = s

    # DSP pipeline
    features = analyzer.update(s)
    out_fixed = fixed.update(s)
    out_adapt = adaptive.update(s, features)

    # Render background
    screen.fill((18, 18, 24))

    # Cursors
    pygame.draw.circle(screen, (100, 100, 100), (int(s.x), int(s.y)), 6)          # Gray: Raw
    pygame.draw.circle(screen, (220, 60, 60), (int(out_fixed.x), int(out_fixed.y)), 8)  # Red: Fixed
    pygame.draw.circle(screen, (40, 220, 120), (int(out_adapt.x), int(out_adapt.y)), 10) # Green: Adaptive

    # Live Telemetry Overlay
    dom_f = f"{features.dominant_frequency:.1f} Hz" if features.dominant_frequency else "N/A"
    lines = [
        f"Velocity:     {features.velocity:6.1f} px/s",
        f"Oscillation:  {features.oscillation_score:6.2f}",
        f"Dominant Freq:{dom_f:>8}",
        f"Tremor Amp:   {features.est_amplitude or 0.0:6.1f} px",
        f"Alpha:        {out_adapt.alpha or 0.0:6.3f}",
        f"Policy:       {adaptive.last_source}",
    ]

    for idx, text in enumerate(lines):
        txt_surface = font.render(text, True, (200, 210, 225))
        screen.blit(txt_surface, (20, 20 + idx * 22))

    # Legend
    legend = [
        ((100, 100, 100), "Raw Mouse"),
        ((220, 60, 60), "Fixed Butterworth (Laggy)"),
        ((40, 220, 120), "Adaptive Filter (Target)"),
    ]
    for idx, (col, desc) in enumerate(legend):
        pygame.draw.circle(screen, col, (700, 30 + idx * 25), 6)
        screen.blit(font.render(desc, True, (220, 220, 220)), (720, 22 + idx * 25))

    pygame.display.flip()
    clock.tick(120)

pygame.quit()