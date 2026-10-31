"""Basic desktop widgets. Every clickable widget can be exposed as a Target."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pygame

from shared.contracts import Target
from visualization import theme


_DEFAULT_STYLE = dict(fill=theme.SURFACE_2, hover_fill=(60, 70, 128), border=theme.BORDER,
                      text_color=theme.TEXT, size=20, radius=7)


@dataclass
class Widget:
    id: str
    x: float
    y: float
    w: float
    h: float
    label: str = ""
    on_click: Optional[Callable[[], None]] = None
    hovered: bool = False
    style: Optional[dict] = None          # per-widget colours/size overrides for draw()

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.w, self.h)

    def to_target(self) -> Target:
        cx, cy = self.center
        return Target(self.id, cx, cy, self.w, self.h)

    def draw(self, surface, **overrides):
        o = {**_DEFAULT_STYLE, **(self.style or {}), **overrides}
        theme.rrect(surface, o["hover_fill"] if self.hovered else o["fill"], self.rect, o["radius"], o["border"])
        if self.label:
            theme.text(surface, self.label, self.rect.center, o["size"], o["text_color"], anchor="center")


class Window:
    TITLE_H = 32

    def __init__(self, id: str, title: str, x: float, y: float, w: float, h: float,
                 on_close: Optional[Callable[[], None]] = None, body_lines: Optional[list[str]] = None):
        self.id, self.title = id, title
        self.x, self.y, self.w, self.h = x, y, w, h
        self.body_lines = body_lines or []
        self.close_btn = Widget(f"{id}:close", x + w - 38, y + 4, 32, 24, "x", on_close)
        self.widgets: list[Widget] = []   # content widgets (calculator keys, toggles...)
        self.on_draw: Optional[Callable] = None   # extra content drawn under the widgets: fn(surface, window)

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.w, self.h)

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def add_button(self, id: str, label: str, rx: float, ry: float, w: float, h: float,
                   on_click: Optional[Callable[[], None]] = None) -> Widget:
        """Add a button at a position RELATIVE to the window's content area."""
        btn = Widget(f"{self.id}:{id}", self.x + rx, self.y + self.TITLE_H + ry, w, h, label, on_click)
        self.widgets.append(btn)
        return btn

    def clickables(self) -> list[Widget]:
        return [self.close_btn, *self.widgets]

    def draw(self, surface, active: bool = True):
        theme.drop_shadow(surface, self.rect, radius=10, spread=14, strength=11 if active else 6)
        theme.rrect(surface, theme.SURFACE, self.rect, 10, theme.BORDER if active else (52, 58, 100), 2)
        bar_color = (58, 70, 150) if active else (44, 50, 96)
        pygame.draw.rect(surface, bar_color, (self.x + 1, self.y + 1, self.w - 2, self.TITLE_H),
                         border_top_left_radius=9, border_top_right_radius=9)
        theme.text(surface, self.title, (self.x + 14, self.y + 9), 22, theme.TEXT)
        for i, line in enumerate(self.body_lines):
            theme.text(surface, line, (self.x + 18, self.y + self.TITLE_H + 18 + i * 24), 20, theme.TEXT_DIM)
        if self.on_draw:
            self.on_draw(surface, self)
        for widget in self.widgets:
            widget.draw(surface)
        self.close_btn.draw(surface, fill=bar_color, hover_fill=(200, 70, 84), border=None,
                            text_color=theme.TEXT, size=24)
