"""Desktop icons in a flat, Windows-11-like style: glyphs sit directly on the wallpaper (no coloured tile),
with a soft ground shadow. Drawn from simple shapes, so no image assets and no trademarked artwork."""
from __future__ import annotations

import math

import pygame

from visualization import theme

WHITE = (250, 250, 253)


def draw_app_icon(surface, key: str, rect):
    rect = pygame.Rect(rect)
    cx, cy = rect.center
    theme.arect(surface, (0, 0, 0, 60), (cx - 21, rect.bottom - 7, 42, 7), 4)      # soft ground shadow
    _GLYPHS[key](surface, cx, cy - 2)


def _files(surface, cx, cy):
    theme.rrect(surface, (226, 160, 36), (cx - 25, cy - 18, 21, 12), 4)             # back tab
    theme.rrect(surface, (240, 176, 48), (cx - 25, cy - 13, 50, 34), 5)             # back panel
    theme.rrect(surface, WHITE, (cx - 19, cy - 10, 38, 14), 3)                      # sheet peeking out
    theme.rrect(surface, (255, 214, 92), (cx - 25, cy - 5, 50, 27), 5)              # front panel
    theme.arect(surface, (255, 255, 255, 60), (cx - 25, cy - 5, 50, 8), 5)          # sheen


def _notepad(surface, cx, cy):
    page = pygame.Rect(cx - 17, cy - 23, 34, 46)
    theme.rrect(surface, WHITE, page, 5)
    pygame.draw.rect(surface, (46, 124, 230), (page.x, page.y, page.w, 11),
                     border_top_left_radius=5, border_top_right_radius=5)
    for dx in (-8, 8):
        pygame.draw.circle(surface, (210, 228, 252), (cx + dx, page.y + 5), 2)       # binder rings
    for i in range(4):
        y = page.y + 19 + i * 6
        pygame.draw.line(surface, (160, 168, 192), (page.x + 6, y), (page.right - (6 if i < 3 else 14), y), 2)


def _calculator(surface, cx, cy):
    body = pygame.Rect(cx - 18, cy - 23, 36, 46)
    theme.rrect(surface, (54, 62, 86), body, 7)
    theme.rrect(surface, (156, 230, 208), (body.x + 4, body.y + 5, 28, 10), 3)       # display
    for r in range(3):
        for c in range(3):
            accent = r == 2 and c == 2
            theme.rrect(surface, (255, 170, 70) if accent else (216, 222, 238),
                        (body.x + 4 + c * 10, body.y + 20 + r * 8, 8, 6), 2)


def _settings(surface, cx, cy):
    for k in range(8):
        a = k * math.pi / 4
        pygame.draw.circle(surface, (196, 205, 224), (cx + math.cos(a) * 18, cy + math.sin(a) * 18), 5)
    pygame.draw.circle(surface, (196, 205, 224), (cx, cy), 16)
    pygame.draw.circle(surface, (96, 108, 142), (cx, cy), 7)
    pygame.draw.circle(surface, (150, 162, 196), (cx, cy), 3)


def _benchmark(surface, cx, cy):
    panel = pygame.Rect(cx - 24, cy - 21, 48, 42)
    theme.rrect(surface, (248, 249, 253), panel, 7)
    for i, (h, color) in enumerate(((11, (70, 140, 235)), (19, (60, 190, 170)),
                                    (15, (150, 110, 240)), (26, (245, 170, 60)))):
        theme.rrect(surface, color, (panel.x + 7 + i * 10, panel.bottom - 7 - h, 7, h), 2)


def _browser(surface, cx, cy):
    pygame.draw.circle(surface, (46, 128, 236), (cx, cy), 23)
    pygame.draw.polygon(surface, (118, 214, 150), [(cx - 14, cy - 8), (cx - 4, cy - 15), (cx + 2, cy - 6),
                                                   (cx - 6, cy + 2), (cx - 15, cy)])
    pygame.draw.polygon(surface, (118, 214, 150), [(cx + 5, cy + 3), (cx + 15, cy + 1), (cx + 12, cy + 14),
                                                   (cx + 5, cy + 11)])
    pygame.draw.circle(surface, (30, 92, 196), (cx, cy), 23, 2)
    theme.arect(surface, (255, 255, 255, 46), (cx - 15, cy - 20, 22, 9), 5)          # highlight


def _photos(surface, cx, cy):
    inner = pygame.Rect(cx - 20, cy - 15, 40, 30)
    theme.rrect(surface, WHITE, inner.inflate(8, 8), 6)
    theme.rrect(surface, (130, 196, 250), inner, 3)
    pygame.draw.circle(surface, (255, 208, 84), (inner.x + 29, inner.y + 9), 4)
    pygame.draw.polygon(surface, (68, 168, 124), [(inner.x, inner.bottom), (inner.x + 13, inner.y + 12),
                                                  (inner.x + 24, inner.bottom)])
    pygame.draw.polygon(surface, (44, 134, 100), [(inner.x + 14, inner.bottom), (inner.x + 27, inner.y + 15),
                                                  (inner.right, inner.bottom)])


_GLYPHS = {"files": _files, "notepad": _notepad, "calculator": _calculator, "settings": _settings,
           "benchmark": _benchmark, "browser": _browser, "photos": _photos}


def draw_start_glyph(surface, rect, hovered: bool = False):
    rect = pygame.Rect(rect)
    cx, cy = rect.center
    color = (140, 200, 255) if hovered else (96, 165, 250)
    for dx in (-9, 1):
        for dy in (-9, 1):
            theme.rrect(surface, color, (cx + dx, cy + dy, 8, 8), 2)
