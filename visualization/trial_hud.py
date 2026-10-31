"""Trial HUD: a small card in the desktop area showing the goal, timer, last result and running stats."""
from __future__ import annotations

import pygame

from visualization import theme

HUD_W, HUD_H = 330, 132
OUTCOME_STYLE = {
    "success": ("SUCCESS", theme.GREEN),
    "wrong_target": ("WRONG TARGET", theme.RED),
    "miss": ("MISSED", theme.AMBER),
    "timeout": ("TIMEOUT", theme.AMBER),
}


def hud_rect(area_w: int) -> pygame.Rect:
    return pygame.Rect(area_w - HUD_W - 14, 12, HUD_W, HUD_H)


def draw_trial_hud(surface, rect, v: dict):
    rect = pygame.Rect(rect)
    theme.drop_shadow(surface, rect, radius=12, spread=12, strength=10)
    theme.rrect(surface, (22, 26, 52), rect, 12, theme.BORDER, 2)
    x, y = rect.x + 14, rect.y + 10
    theme.text(surface, "TRIAL MODE", (x, y), 18, theme.TEXT_DIM)
    if v["auto"]:
        pill = pygame.Rect(rect.right - 112, y - 2, 98, 20)
        theme.rrect(surface, theme.ACCENT, pill, 10)
        theme.text(surface, "AUTO REACH", pill.center, 16, theme.DARK, anchor="center")

    if v["active"]:
        theme.text(surface, "Reach for", (x, y + 26), 20, theme.TEXT)
        key = pygame.Rect(x + 86, y + 20, 44, 34)
        theme.rrect(surface, theme.AMBER, key, 8)
        theme.text(surface, v["goal"], key.center, 28, theme.DARK, anchor="center")
        theme.text(surface, "and click it", (key.right + 10, y + 26), 20, theme.TEXT)
        theme.text(surface, f"{v['elapsed']:.1f} s", (rect.right - 14, y + 24), 28, theme.TEXT, anchor="topright")
    elif v["last"] is not None:
        _mode, r = v["last"]
        label, color = OUTCOME_STYLE[r.outcome]
        theme.text(surface, label, (x, y + 22), 30, color)
        detail = f"{r.elapsed:.2f} s"
        if r.outcome == "wrong_target" and v["last_label"]:
            detail += f"  (clicked {v['last_label']})"
        theme.text(surface, detail, (x, y + 52), 20, theme.TEXT)
    else:
        theme.text(surface, "Press SPACE to start", (x, y + 24), 22, theme.TEXT)
        theme.text(surface, "R = auto reach (synthetic)", (x, y + 48), 18, theme.TEXT_DIM)

    st = v["stats"]
    pygame.draw.line(surface, (60, 70, 130), (x, rect.y + 82), (rect.right - 14, rect.y + 82))
    theme.text(surface, f"Mode: {v['mode'].upper()}", (x, rect.y + 90), 18, theme.TEXT_DIM)
    if st["n"]:
        rate = st["success_rate"] * 100
        med = f"{st['median_time']:.2f} s" if st["median_time"] is not None else "-"
        theme.text(surface, f"{st['success']}/{st['n']} ok ({rate:.0f}%)   median {med}", (x, rect.y + 108), 19,
                   theme.TEXT)
        theme.text(surface, f"wrong {st['wrong_target']}  miss {st['miss']}  timeout {st['timeout']}",
                   (rect.right - 14, rect.y + 91), 16, theme.TEXT_DIM, anchor="topright")
    else:
        theme.text(surface, "No trials in this mode yet", (x, rect.y + 108), 18, theme.TEXT_DIM)
