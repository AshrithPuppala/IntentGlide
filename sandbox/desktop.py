"""A simulated desktop home screen: wallpaper, icons, windows, taskbar and start menu.

Every clickable widget is exposed as a Target, so the intent controller works on
it unchanged. Only widgets that are actually reachable count as targets:
 - menu open:   the menu items and the Start button
 - otherwise:   the top window's controls, taskbar buttons, icons not hidden
                behind a window, and the show-desktop sliver.
The simulated cursor is drawn by the caller (so the side panel can sit under it).
"""
from __future__ import annotations

import math
import time
from typing import Optional

import pygame

from sandbox.benchmark_window import attach_benchmark
from sandbox.calculator import attach_calculator
from sandbox.models import BenchmarkModel, SettingsModel
from sandbox.scatter import scatter_icon_positions
from sandbox.settings_window import attach_settings
from sandbox.widgets import Widget, Window
from shared.contracts import Target
from visualization import theme
from visualization.cursor_canvas import draw_polyline
from visualization.icons import draw_app_icon, draw_start_glyph
from visualization.trial_hud import hud_rect

WINDOW_SPECS = {
    "notepad": ("Notepad", 460, 320, ["Untitled - Notepad", "",
                                      "Try the X button or the taskbar", "with the tremor switched on."]),
    "calculator": ("Calculator", 300, 380, []),
    "settings": ("Settings", 500, 372, []),                  # drawn by sandbox.settings_window
    "benchmark": ("Benchmark", 660, 470, []),                # drawn by sandbox.benchmark_window
    "files": ("Files", 420, 300, ["Documents", "Downloads", "Pictures"]),
    "browser": ("Browser", 480, 320, ["Demo window - nothing loads here.", "Use it as one more click target."]),
    "photos": ("Photos", 420, 300, ["No photos yet.", "Demo window."]),
}
ICON_ORDER = [("notepad", "Notepad"), ("calculator", "Calculator"), ("settings", "Settings"),
              ("benchmark", "Benchmark")]
# Extra demo icons, placed at random (unaligned) positions. They are not in the Start menu.
SCATTER_ICONS = [("files", "Files"), ("browser", "Browser"), ("photos", "Photos")]


class Desktop:
    TASKBAR_H = 48
    TILE = 64

    def __init__(self, width: int = 820, height: int = 700, icon_seed: Optional[int] = 7):
        self.width, self.height = width, height
        self.windows: list[Window] = []           # z-order, last = topmost
        self._taskbar_order: list[str] = []       # order windows were opened
        self.menu_open = False
        self.settings = SettingsModel()           # edited by the Settings window
        self.benchmark = BenchmarkModel()         # shown by the Benchmark window
        self.raw_trail: list[tuple[float, float]] = []
        self.output_trail: list[tuple[float, float]] = []
        self.last_clicked: Optional[str] = None
        self.goal_id: Optional[str] = None        # widget id highlighted as the current trial goal
        self._wallpaper: Optional[pygame.Surface] = None

        # Icon widgets include the caption, like a real desktop icon.
        self.icons = [
            Widget(f"icon:{key}", 18, 20 + i * 108, 84, 100, name,
                   on_click=lambda k=key: self.open_window(k))
            for i, (key, name) in enumerate(ICON_ORDER)
        ]
        self.icons += self._scattered_icons(icon_seed)      # appended, so icons[0:3] stay the originals
        self.start_btn = Widget("taskbar:start", 0, 0, 44, 36, on_click=self.toggle_start)
        self.show_desktop_btn = Widget("taskbar:desktop", width - 14, height - self.TASKBAR_H, 14,
                                       self.TASKBAR_H, "", on_click=self.close_all)
        self.task_buttons: list[Widget] = []
        self.menu_items: list[Widget] = []
        self.menu_rect = pygame.Rect(0, 0, 0, 0)
        self._layout_taskbar()

    def _scattered_icons(self, seed: Optional[int]) -> list[Widget]:
        """Create the extra icons at random positions, kept clear of everything that must stay reachable."""
        w, h = 84, 100
        _, cw, ch, _ = WINDOW_SPECS["calculator"]
        avoid = [
            (0, 0, 120, self.height),                              # the original icon column
            (0, self.height - self.TASKBAR_H - 8, self.width, self.TASKBAR_H + 8),   # taskbar
            tuple(hud_rect(self.width)),                           # trial HUD, top right
            (160, 40, cw, ch),                                     # where the calculator opens
        ]
        area = (124, 16, self.width - 124 - 12, self.height - self.TASKBAR_H - 16 - 12)
        spots = scatter_icon_positions(len(SCATTER_ICONS), area, (w, h), avoid, seed=seed)
        return [Widget(f"icon:{key}", x, y, w, h, name, on_click=lambda k=key: self.open_window(k))
                for (key, name), (x, y) in zip(SCATTER_ICONS, spots)]

    # ---- windows -----------------------------------------------------
    def open_window(self, key: str):
        self.menu_open = False
        for win in self.windows:
            if win.id == key:
                self.raise_window(key)
                return
        title, w, h, lines = WINDOW_SPECS[key]
        n = len(self.windows)
        x = max(0, min(160 + 34 * n, self.width - w - 6))        # keep big windows on screen
        y = max(0, min(40 + 34 * n, self.height - self.TASKBAR_H - h - 6))
        win = Window(key, title, x, y, w, h,
                     on_close=lambda k=key: self.close_window(k), body_lines=lines)
        if key == "calculator":
            attach_calculator(win)               # keypad = 20 closely spaced targets
        elif key == "settings":
            attach_settings(win, self.settings)
        elif key == "benchmark":
            attach_benchmark(win, self.benchmark)
        self.windows.append(win)
        self._taskbar_order.append(key)
        self._layout_taskbar()

    def raise_window(self, key: str):
        for win in self.windows:
            if win.id == key:
                self.windows.remove(win)
                self.windows.append(win)
                break
        self._layout_taskbar()

    def close_window(self, key: str):
        self.windows = [w for w in self.windows if w.id != key]
        self._taskbar_order = [k for k in self._taskbar_order if k != key]
        self._layout_taskbar()

    def close_all(self):
        self.menu_open = False
        self.windows.clear()
        self._taskbar_order.clear()
        self._layout_taskbar()

    def toggle_start(self):
        self.menu_open = not self.menu_open

    # ---- layout ------------------------------------------------------
    def _layout_taskbar(self):
        n = len(self._taskbar_order)
        total = 44 + (6 + n * 126 - 6 if n else 0)
        left = (self.width - total) / 2
        top = self.height - self.TASKBAR_H + 6
        self.start_btn.x, self.start_btn.y = left, top
        titles = {**dict(ICON_ORDER), **dict(SCATTER_ICONS)}
        self.task_buttons = [
            Widget(f"taskbar:win:{key}", left + 50 + i * 126, top, 120, 36, titles[key],
                   on_click=lambda k=key: self.raise_window(k))
            for i, key in enumerate(self._taskbar_order)
        ]
        mw, mh = 224, 52 + len(ICON_ORDER) * 42 + 46
        self.menu_rect = pygame.Rect(left, self.height - self.TASKBAR_H - mh - 8, mw, mh)
        self.menu_items = [
            Widget(f"start:{key}", self.menu_rect.x + 12, self.menu_rect.y + 46 + i * 42, mw - 24, 36,
                   name, on_click=lambda k=key: self.open_window(k))
            for i, (key, name) in enumerate(ICON_ORDER)
        ]
        self.menu_items.append(
            Widget("start:close_all", self.menu_rect.x + 12, self.menu_rect.bottom - 48, mw - 24, 36,
                   "Close all windows", on_click=self.close_all))

    # ---- targets / hit testing --------------------------------------
    def _icon_covered(self, icon: Widget) -> bool:
        cx, cy = icon.center
        return any(win.contains(cx, cy) for win in self.windows)

    def _reachable(self) -> list[Widget]:
        if self.menu_open:
            return [*self.menu_items, self.start_btn]
        items: list[Widget] = []
        if self.windows:
            items += self.windows[-1].clickables()
        items += self.task_buttons
        items += [i for i in self.icons if not self._icon_covered(i)]
        items += [self.start_btn, self.show_desktop_btn]
        return items

    def keypad_targets(self) -> list[Target]:
        """The calculator keys as Targets (empty if the calculator is closed)."""
        win = next((w for w in self.windows if w.id == "calculator"), None)
        return [w.to_target() for w in win.widgets] if win else []

    @property
    def targets(self):
        return [w.to_target() for w in self._reachable()]

    def _hit(self, x: float, y: float) -> Optional[Widget]:
        for widget in self._reachable():
            if widget.contains(x, y):
                return widget
        return None

    def _all_widgets(self):
        yield from self.icons
        yield from self.task_buttons
        yield from self.menu_items
        yield self.start_btn
        yield self.show_desktop_btn
        for win in self.windows:
            yield from win.clickables()

    def hover(self, x: float, y: float):
        hit = self._hit(x, y)
        for w in self._all_widgets():
            w.hovered = w is hit

    def click(self, x: float, y: float) -> Optional[str]:
        """Click at (x, y). Returns the id of the activated widget, or None."""
        hit = self._hit(x, y)
        if hit is not None:
            self.last_clicked = hit.id
            if hit.on_click:
                hit.on_click()
            return hit.id
        self.last_clicked = None
        if self.menu_open:                      # click-away closes the menu
            self.menu_open = False
            return None
        for win in reversed(self.windows[:-1]):  # click on a lower window raises it
            if win.contains(x, y):
                self.raise_window(win.id)
                break
        return None

    # ---- drawing (same API as SandboxScene) -------------------------
    def set_layout(self, layout: str):
        """Layouts belong to the test scene; the desktop ignores them."""

    def add_positions(self, raw_xy, output_xy, trail_limit: int = 90):
        self.raw_trail.append(tuple(raw_xy))
        self.output_trail.append(tuple(output_xy))
        del self.raw_trail[:-trail_limit]
        del self.output_trail[:-trail_limit]

    def _make_wallpaper(self) -> pygame.Surface:
        surf = pygame.Surface((self.width, self.height))
        top, bottom = theme.BG_TOP, theme.BG_BOTTOM
        for y in range(self.height):
            t = y / max(1, self.height - 1)
            pygame.draw.line(surf, tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)),
                             (0, y), (self.width, y))
        glow = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for (fx, fy, radius, color) in (
            (0.78, 0.26, 260, (96, 165, 250)),
            (0.58, 0.70, 220, (150, 110, 240)),
            (0.95, 0.88, 150, (60, 200, 170)),
        ):
            # Concentric low-alpha discs build a soft radial glow.
            for r in range(radius, 0, -6):
                alpha = int(3 + 22 * (1 - r / radius) ** 1.6)
                pygame.draw.circle(glow, (*color, alpha), (int(self.width * fx), int(self.height * fy)), r)
        surf.blit(glow, (0, 0))
        return surf

    def draw(self, surface, cursor_xy=None):
        if self._wallpaper is None:
            self._wallpaper = self._make_wallpaper()
        surface.blit(self._wallpaper, (0, 0))

        for icon in self.icons:
            if icon.hovered:
                theme.arect(surface, (255, 255, 255, 34), icon.rect, 8)
                pygame.draw.rect(surface, (200, 220, 255), icon.rect, width=1, border_radius=8)
            tile = pygame.Rect(icon.x + (icon.w - self.TILE) / 2, icon.y + 8, self.TILE, self.TILE)
            draw_app_icon(surface, icon.id.split(":")[1], tile)
            theme.text(surface, icon.label, (icon.x + icon.w / 2, tile.bottom + 8), 20,
                       theme.TEXT, anchor="midtop", shadow=True)

        top = self.windows[-1] if self.windows else None
        for win in self.windows:
            win.draw(surface, active=win is top)

        self._draw_goal(surface)
        if self.menu_open:
            self._draw_menu(surface)
        self._draw_taskbar(surface)

        draw_polyline(surface, self.raw_trail, (170, 120, 210), 1)
        draw_polyline(surface, self.output_trail, (80, 225, 190), 2)

    def _draw_goal(self, surface):
        """Pulsing amber ring around the key the current trial asks the user to reach."""
        if not self.goal_id:
            return
        widget = next((w for w in self._reachable() if w.id == self.goal_id), None)
        if widget is None:
            return
        pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 6.0)
        ring = widget.rect.inflate(8 + int(4 * pulse), 8 + int(4 * pulse))
        pygame.draw.rect(surface, theme.AMBER, ring, width=3 + int(pulse * 1.5), border_radius=10)

    def _draw_menu(self, surface):
        theme.drop_shadow(surface, self.menu_rect, radius=12, spread=14, strength=12)
        theme.rrect(surface, (26, 30, 58), self.menu_rect, 12, theme.BORDER, 2)
        theme.text(surface, "Apps", (self.menu_rect.x + 16, self.menu_rect.y + 16), 22, theme.TEXT_DIM)
        for item in self.menu_items:
            item.draw(surface, fill=(26, 30, 58), hover_fill=(60, 72, 140), border=None, size=22)

    def _draw_taskbar(self, surface):
        bar = pygame.Rect(0, self.height - self.TASKBAR_H, self.width, self.TASKBAR_H)
        pygame.draw.rect(surface, (14, 16, 34), bar)
        pygame.draw.line(surface, (60, 70, 130), bar.topleft, bar.topright)

        if self.start_btn.hovered or self.menu_open:
            theme.arect(surface, (255, 255, 255, 30), self.start_btn.rect, 8)
        draw_start_glyph(surface, self.start_btn.rect, self.start_btn.hovered)

        active_id = self.windows[-1].id if self.windows else None
        for btn in self.task_buttons:
            is_active = btn.id.endswith(f":{active_id}")
            fill = (46, 54, 108) if is_active else (28, 32, 64)
            btn.draw(surface, fill=fill, hover_fill=(60, 72, 140), border=None, size=20)
            underline = (96, 165, 250) if is_active else (90, 98, 140)
            pygame.draw.line(surface, underline, (btn.x + 24, btn.y + btn.h - 2),
                             (btn.x + btn.w - 24, btn.y + btn.h - 2), 3)

        theme.text(surface, time.strftime("%H:%M"), (self.width - 26, self.height - 28), 22,
                   theme.TEXT, anchor="midright")
        theme.text(surface, time.strftime("%d %b %Y"), (self.width - 26, self.height - 12), 16,
                   theme.TEXT_DIM, anchor="midright")
        sliver = self.show_desktop_btn
        pygame.draw.line(surface, (110, 120, 180) if sliver.hovered else (60, 70, 130),
                         (sliver.x, sliver.y + 8), (sliver.x, sliver.y + sliver.h - 8))
