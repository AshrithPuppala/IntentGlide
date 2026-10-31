"""Pygame sandbox scene: target workspace and cursor trail."""
from sandbox.targets import make_targets


class SandboxScene:
    def __init__(self, width: int = 900, height: int = 600):
        self.width = width
        self.height = height
        self.layout = "standard"
        self.targets = make_targets(self.layout, width, height)
        self.raw_trail = []
        self.output_trail = []

    def set_layout(self, layout: str):
        self.layout = layout
        self.targets = make_targets(layout, self.width, self.height)

    def add_positions(self, raw_xy, output_xy, trail_limit: int = 250):
        self.raw_trail.append(tuple(raw_xy))
        self.output_trail.append(tuple(output_xy))
        del self.raw_trail[:-trail_limit]
        del self.output_trail[:-trail_limit]

    def draw(self, surface, cursor_xy=None):
        import pygame

        surface.fill((18, 16, 38))
        for target in self.targets:
            rect = pygame.Rect(
                target.center_x - target.width / 2,
                target.center_y - target.height / 2,
                target.width,
                target.height,
            )
            pygame.draw.rect(surface, (73, 190, 230), rect, border_radius=8, width=2)
            label = pygame.font.Font(None, 20).render(target.id, True, (230, 235, 250))
            surface.blit(label, (rect.x, rect.y - 22))

        if len(self.raw_trail) > 1:
            pygame.draw.lines(surface, (130, 110, 160), False, self.raw_trail, 1)
        if len(self.output_trail) > 1:
            pygame.draw.lines(surface, (80, 220, 190), False, self.output_trail, 2)
        if cursor_xy:
            pygame.draw.circle(
                surface, (255, 220, 120),
                (int(cursor_xy[0]), int(cursor_xy[1])), 7
            )
