"""Shared colours, fonts and small drawing helpers for the IntentGlide UI."""
from __future__ import annotations

import pygame

BG_TOP = (26, 34, 84)
BG_BOTTOM = (9, 11, 28)
SURFACE = (24, 27, 52)
SURFACE_2 = (36, 41, 74)
BORDER = (72, 84, 150)
TEXT = (236, 240, 252)
TEXT_DIM = (148, 156, 190)
ACCENT = (96, 165, 250)
GREEN = (110, 225, 160)
AMBER = (245, 190, 90)
RED = (240, 110, 110)
GREY = (170, 170, 185)
DARK = (14, 16, 32)

_fonts: dict[int, pygame.font.Font] = {}
_shadows: dict[tuple, pygame.Surface] = {}


def font(size: int) -> pygame.font.Font:
    if size not in _fonts:
        _fonts[size] = pygame.font.Font(None, size)
    return _fonts[size]


def text(surface, s: str, pos, size: int = 22, color=TEXT, anchor: str = "topleft",
         shadow: bool = False) -> pygame.Rect:
    f = font(size)
    if shadow:
        dark = f.render(s, True, (0, 0, 0))
        surface.blit(dark, dark.get_rect(**{anchor: (pos[0] + 1, pos[1] + 1)}))
    img = f.render(s, True, color)
    rect = img.get_rect(**{anchor: pos})
    surface.blit(img, rect)
    return rect


def rrect(surface, color, rect, radius: int = 8, border=None, width: int = 1):
    rect = pygame.Rect(rect)
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border is not None:
        pygame.draw.rect(surface, border, rect, width=width, border_radius=radius)


def arect(surface, rgba, rect, radius: int = 8):
    """Rounded rectangle with alpha."""
    rect = pygame.Rect(rect)
    layer = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(layer, rgba, layer.get_rect(), border_radius=radius)
    surface.blit(layer, rect.topleft)


def drop_shadow(surface, rect, radius: int = 10, spread: int = 12, strength: int = 12):
    rect = pygame.Rect(rect)
    key = (rect.w, rect.h, radius, spread, strength)
    layer = _shadows.get(key)
    if layer is None:
        layer = pygame.Surface((rect.w + 2 * spread, rect.h + 2 * spread), pygame.SRCALPHA)
        for i in range(spread):
            inner = pygame.Rect(i, i, rect.w + 2 * (spread - i), rect.h + 2 * (spread - i))
            pygame.draw.rect(layer, (0, 0, 0, strength), inner, border_radius=radius + spread - i)
        _shadows[key] = layer
    surface.blit(layer, (rect.x - spread, rect.y - spread + 4))
