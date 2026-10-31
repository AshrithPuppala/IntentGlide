import pygame
from input.input_provider import MouseProvider

pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Live mouse test - move the mouse here")
clock = pygame.time.Clock()
mouse = MouseProvider()

prev = None
running = True
while running:
    for event in pygame.event.get():          # keeps the window responsive
        if event.type == pygame.QUIT:
            running = False

    s = mouse.get_sample()
    dt = (s.timestamp - prev.timestamp) if prev else 0.0
    print(f"t={s.timestamp:.3f}  x={s.x:.0f}  y={s.y:.0f}  dt={dt*1000:.1f} ms  src={s.source}")
    prev = s

    screen.fill((20, 20, 20))
    pygame.draw.circle(screen, (0, 200, 120), (int(s.x), int(s.y)), 8)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()