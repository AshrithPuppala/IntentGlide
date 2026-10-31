"""App icons drawn from simple shapes (no image assets, no trademarked artwork)."""
from __future__ import annotations

import math

import pygame

from visualization import theme

TILE_COLORS = {
    "notepad": (244, 192, 72),
    "calculator": (48, 190, 168),
    "settings": (112, 128, 206),
    "benchmark": (150, 110, 240),
    "files": (240, 160, 70),
    "browser": (70, 140, 235),
    "photos": (226, 96, 140),
}


def draw_app_icon(surface, key: str, rect):
    rect = pygame.Rect(rect)
    theme.rrect(surface, TILE_COLORS[key], rect, 16)
    theme.arect(surface, (255, 255, 255, 38), (rect.x, rect.y, rect.w, rect.h // 2), 16)  # soft sheen
    cx, cy = rect.center
    {"notepad": _notepad, "calculator": _calculator, "settings": _gear,
     "benchmark": _bars, "files": _folder, "browser": _globe, "photos": _photo}[key](surface, cx, cy, TILE_COLORS[key])


def _notepad(surface, cx, cy, tile):
    page = pygame.Rect(cx - 16, cy - 20, 32, 40)
    theme.rrect(surface, (250, 250, 253), page, 4)
    pygame.draw.rect(surface, (214, 96, 76), (page.x, page.y, page.w, 8),
                     border_top_left_radius=4, border_top_right_radius=4)
    for i in range(3):
        y = page.y + 16 + i * 7
        pygame.draw.line(surface, (150, 156, 178), (page.x + 6, y), (page.right - 6, y), 2)


def _calculator(surface, cx, cy, tile):
    body = pygame.Rect(cx - 16, cy - 21, 32, 42)
    theme.rrect(surface, (34, 44, 70), body, 5)
    theme.rrect(surface, (150, 236, 204), (body.x + 4, body.y + 4, 24, 9), 2)
    for r in range(3):
        for c in range(3):
            theme.rrect(surface, (232, 236, 246), (body.x + 4 + c * 8, body.y + 17 + r * 8, 6, 6), 1)


def _gear(surface, cx, cy, tile):
    for k in range(8):
        a = k * math.pi / 4
        pygame.draw.circle(surface, (248, 249, 253), (cx + math.cos(a) * 16, cy + math.sin(a) * 16), 4)
    pygame.draw.circle(surface, (248, 249, 253), (cx, cy), 14)
    pygame.draw.circle(surface, tile, (cx, cy), 6)


def _bars(surface, cx, cy, tile):
    for i, h in enumerate((14, 26, 20, 34)):
        theme.rrect(surface, (250, 250, 253), (cx - 22 + i * 12, cy + 18 - h, 9, h), 2)
    pygame.draw.line(surface, (250, 250, 253), (cx - 25, cy + 20), (cx + 25, cy + 20), 2)


def _folder(surface, cx, cy, tile):
    pygame.draw.rect(surface, (255, 226, 160), (cx - 20, cy - 14, 16, 8), border_top_left_radius=4,
                     border_top_right_radius=4)
    theme.rrect(surface, (252, 244, 220), (cx - 20, cy - 8, 40, 28), 5)
    theme.rrect(surface, (255, 214, 130), (cx - 20, cy - 1, 40, 21), 5)


def _globe(surface, cx, cy, tile):
    pygame.draw.circle(surface, (248, 249, 253), (cx, cy), 19)
    pygame.draw.circle(surface, tile, (cx, cy), 19, 2)
    pygame.draw.ellipse(surface, tile, (cx - 8, cy - 19, 16, 38), 2)
    pygame.draw.line(surface, tile, (cx - 19, cy), (cx + 19, cy), 2)
    pygame.draw.line(surface, tile, (cx - 15, cy - 9), (cx + 15, cy - 9), 2)
    pygame.draw.line(surface, tile, (cx - 15, cy + 9), (cx + 15, cy + 9), 2)


def _photo(surface, cx, cy, tile):
    frame = pygame.Rect(cx - 20, cy - 16, 40, 32)
    theme.rrect(surface, (250, 250, 253), frame, 5)
    pygame.draw.circle(surface, (250, 190, 70), (frame.x + 28, frame.y + 10), 4)
    pygame.draw.polygon(surface, (70, 170, 130), [(frame.x + 3, frame.bottom - 4), (frame.x + 15, frame.y + 14),
                                                  (frame.x + 24, frame.bottom - 4)])
    pygame.draw.polygon(surface, (50, 140, 110), [(frame.x + 17, frame.bottom - 4), (frame.x + 27, frame.y + 17),
                                                  (frame.right - 3, frame.bottom - 4)])


def draw_start_glyph(surface, rect, hovered: bool = False):
    rect = pygame.Rect(rect)
    cx, cy = rect.center
    color = (140, 200, 255) if hovered else (96, 165, 250)
    for dx in (-9, 1):
        for dy in (-9, 1):
            theme.rrect(surface, color, (cx + dx, cy + dy, 8, 8), 2)
