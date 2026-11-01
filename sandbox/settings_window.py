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
MODE_Y, FREQ_Y, AMP_Y, TOGGLE_Y = 32, 102, 148, 190
PROFILE_TITLE_Y = 238
PROFILE_ROWS = {
    "dwell_seconds": (270, "Dwell duration", "s", 1),
    "acquisition_base_margin_px": (320, "Target acquisition margin", "px", 0),
    "maximum_assistance_distance_px": (370, "Magnetic attraction radius", "px", 0),
    "magnetic_strength": (420, "Magnetic strength", "", 2),
    "sticky_grace_ms": (470, "Sticky target grace", "ms", 0),
    "maximum_magnetic_speed_px_s": (520, "Assistance speed cap", "px/s", 0),
}
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
    profile_buttons = {}
    for name, (row_y, _, _, kind) in PROFILE_ROWS.items():
        if kind == 2:
            profile_buttons[name] = win.add_button("profile:magnet", "", win.w-222, row_y, 204, 36,
                                                       on_click=model.cycle_strength)
            profile_buttons[name].style = _STEP
        else:
            step = model.step_profile
            profile_buttons[name] = (
                win.add_button(f"profile:{name}:down", "−", win.w-222, row_y, 42, 36,
                    on_click=lambda name=name, step=step: step(name, -1)),
                win.add_button(f"profile:{name}:up", "+", win.w-66, row_y, 42, 36,
                    on_click=lambda name=name, step=step: step(name, +1)),
            )
            for button in profile_buttons[name]:
                button.style = _STEP

    def refresh():
        for m, btn in mode_btns.items():
            btn.style = _SELECTED if model.mode == m else _IDLE
        toggle.label = "Tremor: ON" if model.tremor_on else "Tremor: OFF"
        toggle.style = _ON if model.tremor_on else _OFF
        profile_buttons["magnetic_strength"].label = model.magnetic_strength.title()

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
        theme.text(surface, "Personalised assistance", (w.x + PAD, top + PROFILE_TITLE_Y), 20, theme.TEXT_DIM)
        for name, (row_y, label, unit, kind) in PROFILE_ROWS.items():
            theme.text(surface, label, (w.x + PAD, top + row_y + 9), 18, theme.TEXT)
            if kind == 2:
                continue
            else:
                value = getattr(model, name)
                text = f"{value:.1f} {unit}" if kind == 1 else f"{value:.0f} {unit}"
                theme.text(surface, text, (w.x + w.w - 123, top + row_y + 18),
                           18, theme.TEXT, anchor="center")
        pygame.draw.line(surface, (60, 70, 130), (w.x + PAD, top + 568),
                         (w.x + w.w - PAD, top + 568))
        theme.text(surface, "3-sec dwell · isolated X controls get a safe +16 px zone.",
                   (w.x + PAD, top + 577), 16, theme.TEXT_DIM)
        theme.text(surface, "F1/F3 magnetic · F4/F5 dwell · F6 target · F8/F9 sticky feedback",
                   (w.x + PAD, top + 598), 13, theme.TEXT_DIM)

    win.on_draw = draw_content
    refresh()
    return model

