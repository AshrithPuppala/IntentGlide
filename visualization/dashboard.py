"""Pygame text-panel rendering for diagnostic rows."""
def draw_dashboard(surface, rows, position=(12, 12)):
    import pygame
    font = pygame.font.Font(None, 22)
    x, y = position
    for row in rows:
        rendered = font.render(str(row), True, (225, 230, 245))
        surface.blit(rendered, (x, y))
        y += 23
def draw_badge(surface, text, color, top_right=(1088, 12)):
    import pygame
    font = pygame.font.Font(None, 26)
    label = font.render(text, True, (15, 15, 30))
    rect = label.get_rect()
    rect.inflate_ip(24, 14)
    rect.topright = top_right
    pygame.draw.rect(surface, color, rect, border_radius=10)
    surface.blit(label, label.get_rect(center=rect.center))