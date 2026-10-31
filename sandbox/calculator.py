"""A small calculator: logic plus a dense 4x5 keypad window.

The keypad is deliberately tight (20 keys, 6 px gaps). It is the best place to show
confidence-gated assistance: many plausible neighbouring targets, so the controller
should abstain unless one key is clearly the intended one.
"""
from __future__ import annotations

from sandbox.widgets import Window
from visualization import theme

DIGITS = "0123456789"
OPS = {"plus": "+", "minus": "-", "mul": "x", "div": "/"}
LAYOUT = [
    ["clear", "del", "pct", "div"],
    ["7", "8", "9", "mul"],
    ["4", "5", "6", "minus"],
    ["1", "2", "3", "plus"],
    ["neg", "0", "dot", "eq"],
]
LABELS = {"clear": "C", "del": "DEL", "pct": "%", "neg": "+/-", "dot": ".", "eq": "=", **OPS}

KEY_W, KEY_H, GAP = 62, 46, 6


def _fmt(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):
        raise ArithmeticError
    if value == int(value) and abs(value) < 1e12:
        return str(int(value))
    return f"{value:.10g}"


class Calculator:
    """Immediate-execution calculator (2 + 3 x 4 = 20, like a basic desk calculator)."""

    MAX_DIGITS = 12

    def __init__(self):
        self.reset()

    def reset(self):
        self.entry = "0"
        self.acc: float | None = None
        self.op: str | None = None
        self.fresh = True          # next digit starts a new number
        self.error = False

    @property
    def display(self) -> str:
        return "Error" if self.error else self.entry

    def press(self, key: str):
        if key == "clear":
            self.reset()
            return
        if self.error:
            return
        if key in DIGITS:
            if self.fresh:
                self.entry, self.fresh = key, False
            elif len(self.entry.replace("-", "").replace(".", "")) < self.MAX_DIGITS:
                self.entry = key if self.entry == "0" else self.entry + key
        elif key == "dot":
            if self.fresh:
                self.entry, self.fresh = "0.", False
            elif "." not in self.entry:
                self.entry += "."
        elif key == "neg":
            if self.entry != "0":
                self.entry = self.entry[1:] if self.entry.startswith("-") else "-" + self.entry
        elif key == "pct":
            self._try(lambda: _fmt(float(self.entry) / 100.0))
            self.fresh = True
        elif key == "del":
            if not self.fresh:
                self.entry = self.entry[:-1]
                if self.entry in ("", "-"):
                    self.entry, self.fresh = "0", True
        elif key in OPS:
            if self.op is not None and not self.fresh:
                self._compute()
            elif self.op is None:
                self.acc = float(self.entry)
            self.op, self.fresh = key, True
        elif key == "eq":
            if self.op is not None:
                self._compute()
            self.acc, self.op, self.fresh = None, None, True

    def _try(self, fn):
        try:
            self.entry = fn()
        except (ArithmeticError, ValueError, OverflowError):
            self.error = True

    def _compute(self):
        cur = float(self.entry)
        acc = self.acc if self.acc is not None else cur

        def calc():
            if self.op == "plus":
                result = acc + cur
            elif self.op == "minus":
                result = acc - cur
            elif self.op == "mul":
                result = acc * cur
            else:
                result = acc / cur
            self.acc = result
            return _fmt(result)

        self._try(calc)


def _key_style(name: str) -> dict:
    if name == "eq":
        return dict(fill=(36, 120, 90), hover_fill=(58, 168, 124), border=None)
    if name in OPS:
        return dict(fill=(50, 66, 136), hover_fill=(80, 104, 190), border=None)
    if name in ("clear", "del"):
        return dict(fill=(104, 50, 62), hover_fill=(160, 70, 84), border=None)
    if name in ("pct", "neg"):
        return dict(fill=(44, 50, 88), hover_fill=(66, 78, 142), border=None)
    return dict(fill=(36, 42, 78), hover_fill=(66, 78, 142), border=None)


def attach_calculator(win: Window) -> Calculator:
    """Add the display and keypad to a calculator window. Returns the logic object."""
    calc = Calculator()
    win.calc = calc
    left = (win.w - (4 * KEY_W + 3 * GAP)) / 2
    top = 80
    for r, row in enumerate(LAYOUT):
        for c, name in enumerate(row):
            win.add_button(f"key:{name}", LABELS.get(name, name),
                           left + c * (KEY_W + GAP), top + r * (KEY_H + GAP), KEY_W, KEY_H,
                           on_click=lambda n=name: calc.press(n)).style = {**_key_style(name), "size": 26}

    def draw_display(surface, window):
        import pygame
        box = pygame.Rect(window.x + left, window.y + window.TITLE_H + 12, 4 * KEY_W + 3 * GAP, 56)
        theme.rrect(surface, (12, 14, 30), box, 8, (60, 70, 130))
        if calc.op is not None and not calc.error:
            theme.text(surface, OPS[calc.op], (box.x + 12, box.y + 8), 22, theme.TEXT_DIM)
        size = 40 if len(calc.display) <= 9 else 30
        theme.text(surface, calc.display, (box.right - 12, box.centery + 4), size,
                   theme.RED if calc.error else theme.TEXT, anchor="midright")

    win.on_draw = draw_display
    return calc
