"""Pygame text-panel rendering for diagnostic rows."""
def draw_dashboard(surface, rows, position=(12, 12)):
    import pygame
    font = pygame.font.Font(None, 22)
    x, y = position
    for row in rows:
        rendered = font.render(str(row), True, (225, 230, 245))
        surface.blit(rendered, (x, y))
        y += 23
