"""Optional live overlay for actual/virtual target geometry and assistance vectors."""
from __future__ import annotations

import math
import pygame

from interaction.target_geometry import actual_rect, expanded_rect, regions_overlap
from visualization import theme


def draw_target_diagnostics(surface, targets, margin_px, selected_id, cursor_xy,
                            velocity_xy=(0.0, 0.0), magnetic_xy=(0.0, 0.0)):
    targets = list(targets)
    virtual = {t.id: expanded_rect(t, max(0.0, margin_px)) for t in targets}
    for i, a in enumerate(targets):
        for b in targets[i+1:]:
            if regions_overlap(a, max(0.0, margin_px), b, max(0.0, margin_px)):
                ra, rb = virtual[a.id], virtual[b.id]
                box = pygame.Rect(round(max(ra.left, rb.left)), round(max(ra.top, rb.top)),
                    max(1, round(min(ra.right, rb.right)-max(ra.left, rb.left))),
                    max(1, round(min(ra.bottom, rb.bottom)-max(ra.top, rb.top))))
                shade = pygame.Surface(box.size, pygame.SRCALPHA)
                shade.fill((255, 174, 72, 35))
                surface.blit(shade, box.topleft)
    for target in targets:
        a = actual_rect(target)
        actual = pygame.Rect(round(a.left), round(a.top), round(a.width), round(a.height))
        v = virtual[target.id]
        expanded = pygame.Rect(round(v.left), round(v.top), round(v.width), round(v.height))
        color = (70, 230, 165) if target.id == selected_id else (89, 174, 255)
        pygame.draw.rect(surface, color, expanded, 1, border_radius=4)
        pygame.draw.rect(surface, (255, 110, 115), actual, 2, border_radius=3)
    x, y = cursor_xy
    vx, vy = velocity_xy
    speed = math.hypot(vx, vy)
    if speed > 1:
        end = (x + vx/speed*55, y + vy/speed*55)
        pygame.draw.line(surface, (246, 226, 96), (x, y), end, 3)
    mx, my = magnetic_xy
    if math.hypot(mx, my) > 0.2:
        pygame.draw.line(surface, (80, 242, 195), (x, y), (x+mx*3, y+my*3), 3)
    theme.text(surface, "ACTUAL  ·  VIRTUAL  ·  OVERLAP  ·  VELOCITY  ·  MAGNETIC",
               (10, 8), 13, (245, 247, 255))
