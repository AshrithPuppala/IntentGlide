"""Drawing helpers for cursor paths and markers."""
def draw_cursor(surface, position, color=(255, 220, 120), radius=7):
    import pygame
    pygame.draw.circle(surface, color, (int(position[0]), int(position[1])), radius)


def draw_polyline(surface, points, color, width=2):
    import pygame
    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, width)
