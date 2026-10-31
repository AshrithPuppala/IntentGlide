"""Settings window layout: filtering mode, tremor frequency / amplitude, tremor on/off.

Steppers (- / +) rather than sliders: they are big, separate click targets, which suits the tremor demo.
Controls edit a SettingsModel; connecting them to the pipeline is done by subscribing to that model.
"""
from __future__ import annotations

import pygame

from sandbox.models import MODE_LABELS, MODES, SettingsModel
from sandbox.widgets import Window
from visualization import theme

PAD = 18
MODE_Y, FREQ_Y, AMP_Y, TOGGLE_Y, NOTE_Y = 40, 128, 190, 254, 306
_IDLE = dict(fill=(36, 42, 78), hover_fill=(66, 78, 142), border=None, size=19)
_SELECTED = dict(fill=(52, 112, 196), hover_fill=(76, 140, 226), border=theme.ACCENT, size=19)
_STEP = dict(fill=(44, 50, 88), hover_fill=(66, 78, 142), border=None, size=30)
_ON = dict(fill=(36, 120, 90), hover_fill=(58, 168, 124), border=None, size=21)
_OFF = dict(fill=(104, 50, 62), hover_fill=(160, 70, 84), border=None, size=21)


def attach_settings(win: Window, model: SettingsModel) -> SettingsModel:
    mode_w = (win.w - 2 * PAD - 4 * 8) / 5
    mode_btns = {}
    for i, m in enumerate(MODES):
        mode_btns[m] = win.add_button(f"mode:{m}", MODE_LABELS[m], PAD + i * (mode_w + 8), MODE_Y, mode_w, 40,
                                      on_click=lambda m=m: model.set_mode(m))
    sx = win.w - PAD - 200                                   # left edge of each stepper
    for row_y, name, step in ((FREQ_Y, "freq", model.step_freq), (AMP_Y, "amp", model.step_amp)):
        win.add_button(f"{name}:down", "-", sx, row_y, 44, 40, on_click=lambda s=step: s(-1)).style = _STEP
        win.add_button(f"{name}:up", "+", sx + 156, row_y, 44, 40, on_click=lambda s=step: s(+1)).style = _STEP
    toggle = win.add_button("tremor:toggle", "", PAD, TOGGLE_Y, 170, 40, on_click=model.toggle_tremor)

    def refresh():
        for m, btn in mode_btns.items():
            btn.style = _SELECTED if model.mode == m else _IDLE
        toggle.label = "Tremor: ON" if model.tremor_on else "Tremor: OFF"
        toggle.style = _ON if model.tremor_on else _OFF

    def draw_content(surface, w):
        refresh()                                            # the model may be changed from outside
        top = w.y + w.TITLE_H
        theme.text(surface, "Filtering mode", (w.x + PAD, top + 12), 20, theme.TEXT_DIM)
        for row_y, label, value, text, bounds in (
            (FREQ_Y, "Tremor frequency", model.tremor_freq, f"{model.tremor_freq:.0f} Hz", model.FREQ_RANGE),
            (AMP_Y, "Tremor amplitude", model.tremor_amp, f"{model.tremor_amp:.0f} px", model.AMP_RANGE),
        ):
            theme.text(surface, label, (w.x + PAD, top + row_y + 2), 20, theme.TEXT)
            track = pygame.Rect(w.x + PAD, top + row_y + 30, sx - PAD - 18, 6)
            theme.rrect(surface, (40, 46, 84), track, 3)
            frac = (value - bounds[0]) / (bounds[1] - bounds[0])
            theme.rrect(surface, theme.ACCENT, (track.x, track.y, max(6, track.w * frac), track.h), 3)
            theme.text(surface, text, (w.x + sx + 100, top + row_y + 20), 22, theme.TEXT, anchor="center")
        pygame.draw.line(surface, (60, 70, 130), (w.x + PAD, top + NOTE_Y - 12), (w.x + w.w - PAD, top + NOTE_Y - 12))
        if not model.connected:
            theme.text(surface, "Layout preview: controls are not connected to the live pipeline yet.",
                       (w.x + PAD, top + NOTE_Y), 17, theme.AMBER)

    win.on_draw = draw_content
    refresh()
    return model
