"""Docked side panel: live controller status, metrics and key hints."""
from __future__ import annotations

import pygame

from visualization import theme

PANEL_W = 280
MODES = ("raw", "fixed", "adaptive", "full")
MODE_LABELS = {"raw": "RAW", "fixed": "FIXED", "adaptive": "ADAPT", "full": "FULL"}
STATES = (
    ("stable", "Stable movement", theme.ACCENT),
    ("oscillation_detected", "Oscillation detected", theme.AMBER),
    ("target_confident", "Target confident", theme.GREEN),
    ("target_ambiguous", "Target ambiguous", theme.AMBER),
    ("override", "Override", theme.RED),
)


def _status_badge(s):
    decision = s["decision"]
    if s["override"]:
        return "OVERRIDDEN: direct control", theme.RED
    if s["mode"] != "full":
        return f"ASSISTANCE OFF ({s['mode']})", theme.GREY
    if decision is not None and decision.state == "target_confident":
        return f"ASSISTING: {decision.target_id}", theme.GREEN
    if decision is not None and decision.state == "target_ambiguous":
        return "ABSTAINING: target unclear", theme.AMBER
    return "ASSISTANCE ON: watching", theme.ACCENT


def _label(surface, s, x, y):
    theme.text(surface, s, (x, y), 17, theme.TEXT_DIM)


def draw_side_panel(surface, rect, s: dict):
    """s keys: mode, features, decision, override, filter_source, cutoff_hz,
    processing_ms, source, tremor."""
    rect = pygame.Rect(rect)
    pygame.draw.rect(surface, (16, 18, 38), rect)
    pygame.draw.line(surface, (60, 70, 130), rect.topleft, rect.bottomleft, 2)
    x, w = rect.x + 16, rect.w - 32
    y = rect.y + 14

    theme.text(surface, "IntentGlide", (x, y), 34, theme.TEXT)
    theme.text(surface, "Adaptive cursor assistance", (x, y + 30), 18, theme.TEXT_DIM)
    y += 58

    text, color = _status_badge(s)
    theme.rrect(surface, color, (x, y, w, 38), 19)
    theme.text(surface, text, (x + w / 2, y + 19), 22, theme.DARK, anchor="center")
    y += 50

    _label(surface, "FILTER MODE  (keys 1-4)", x, y)
    y += 20
    chip_w = (w - 3 * 6) // 4
    for i, mode in enumerate(MODES):
        r = pygame.Rect(x + i * (chip_w + 6), y, chip_w, 30)
        active = mode == s["mode"]
        theme.rrect(surface, theme.ACCENT if active else theme.SURFACE_2, r, 8)
        theme.text(surface, MODE_LABELS[mode], r.center, 18, theme.DARK if active else theme.TEXT_DIM,
                   anchor="center")
    y += 42

    _label(surface, "LIVE METRICS", x, y)
    y += 20
    f, d = s["features"], s["decision"]
    freq = f.dominant_frequency
    conf = d.confidence if d is not None and s["mode"] == "full" and not s["override"] else None
    cards = [
        ("Oscillation", f"{f.oscillation_score:.2f}", theme.TEXT),
        ("Frequency", f"{freq:.1f} Hz" if freq is not None else "-", theme.TEXT),
        ("Filter cutoff", f"{s['cutoff_hz']:.1f} Hz" if s["cutoff_hz"] is not None else "-", theme.TEXT),
        ("Target conf.", f"{conf:.2f}" if conf is not None else "-",
         theme.GREEN if conf is not None and conf >= 0.7 else theme.TEXT),
        ("Velocity", f"{f.velocity:.0f} px/s", theme.TEXT),
        ("Processing", f"{s['processing_ms']:.2f} ms", theme.TEXT),
    ]
    cw = (w - 8) // 2
    for i, (label, value, vcolor) in enumerate(cards):
        r = pygame.Rect(x + (i % 2) * (cw + 8), y + (i // 2) * 58, cw, 52)
        theme.rrect(surface, theme.SURFACE_2, r, 9)
        theme.text(surface, label, (r.x + 10, r.y + 7), 16, theme.TEXT_DIM)
        theme.text(surface, value, (r.x + 10, r.y + 26), 26, vcolor)
    y += 3 * 58 + 6

    _label(surface, "CONTROLLER STATE", x, y)
    y += 20
    active_state = "override" if s["override"] else (d.state if d is not None and s["mode"] == "full" else None)
    for key, name, color in STATES:
        r = pygame.Rect(x, y, w, 24)
        on = key == active_state
        if on:
            theme.arect(surface, (*color, 50), r, 7)
            pygame.draw.rect(surface, color, r, width=1, border_radius=7)
        pygame.draw.circle(surface, color if on else (70, 76, 112), (r.x + 14, r.centery), 5)
        theme.text(surface, name, (r.x + 28, r.centery), 19, theme.TEXT if on else theme.TEXT_DIM,
                   anchor="midleft")
        y += 26
    if active_state is None:
        theme.text(surface, "Controller idle in this mode", (x, y), 16, theme.TEXT_DIM)
    y += 14
    theme.text(surface, f"Input: {s['source'].upper()}   Tremor {'ON' if s['tremor'] else 'OFF'}", (x, y), 20)
    theme.text(surface, f"Filter policy: {s['filter_source']}", (x, y + 20), 18, theme.TEXT_DIM)

    hints = ["D desktop   T test targets", "S input source   V tremor",
             "Ctrl+Shift+A override   Esc quit"]
    hy = rect.bottom - 18 * (len(hints) + 2) - 8
    for i, line in enumerate(hints):
        theme.text(surface, line, (x, hy + i * 18), 16, theme.TEXT_DIM)
    theme.text(surface, "Engineering prototype. Simulated tremor,", (x, rect.bottom - 40), 15, (110, 118, 156))
    theme.text(surface, "not a medical device.", (x, rect.bottom - 24), 15, (110, 118, 156))
