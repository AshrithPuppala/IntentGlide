"""Compact live dashboard for filter, intent and dwell interaction state."""
from __future__ import annotations

import pygame
from visualization import theme

PANEL_W = 280
MODES = ("raw", "fixed", "adaptive", "full", "ml_butterworth")
MODE_LABELS = {
    "raw": "RAW", "fixed": "FIXED", "adaptive": "EMA",
    "full": "FULL", "ml_butterworth": "ML-BW",
}


def _target_name(target_id):
    if not target_id:
        return "No target yet"
    name = str(target_id).split(":")[-1].replace("target-", "Target ").replace("_", " ")
    return name.replace("close_all", "Close all").title()


def _card(surface, rect, fill=None, border=None, radius=10):
    theme.rrect(surface, fill or theme.SURFACE_2, rect, radius,
                 border if border is not None else (58, 66, 108), 1)


def _progress(surface, rect, value, color):
    pygame.draw.rect(surface, (19, 22, 44), rect, border_radius=5)
    fill = pygame.Rect(rect.x, rect.y, round(rect.w * max(0.0, min(1.0, value))), rect.h)
    if fill.w > 0:
        pygame.draw.rect(surface, color, fill, border_radius=5)


def draw_side_panel(surface, rect, s: dict):
    rect = pygame.Rect(rect)
    pygame.draw.rect(surface, (14, 16, 34), rect)
    pygame.draw.line(surface, (54, 64, 118), rect.topleft, rect.bottomleft, 2)
    x, w = rect.x + 14, rect.w - 28

    theme.text(surface, "IntentGlide", (x, rect.y + 12), 32, theme.TEXT)
    theme.text(surface, "ASSISTIVE POINTER LAB", (x, rect.y + 43), 14, theme.TEXT_DIM)

    decision = s.get("decision")
    diag = s.get("intent_diagnostics") or {}
    dwell = s.get("dwell_status")
    candidate = (decision.target_id if decision and decision.state == "target_confident"
                 else diag.get("candidate_id"))
    name = _target_name(candidate)

    # A single status card makes it clear that filtering and intent are separate.
    y = rect.y + 67
    if s.get("override"):
        status, status_color = "MANUAL OVERRIDE", theme.RED
    elif s.get("calibration") is not None:
        done, total = s["calibration"].progress
        status, status_color = f"CALIBRATION · {done}/{total} · SELECT HIGHLIGHTED TARGET", theme.AMBER
    elif not s.get("inside_scene", True):
        status, status_color = "OUTSIDE TARGET AREA", theme.AMBER
    elif decision and decision.state == "target_confident":
        status = f"{'TARGET LOCKED' if s.get('target_lock') else 'TARGET CONFIRMED'} · {name.upper()}"
        status_color = theme.GREEN if s.get("assist_enabled") else theme.ACCENT
    elif candidate:
        status, status_color = f"CONFIRMING · {name.upper()}", theme.AMBER
    else:
        status, status_color = "SEARCHING FOR INTENT", theme.ACCENT
    _card(surface, (x, y, w, 42), fill=(25, 31, 58), border=status_color)
    pygame.draw.circle(surface, status_color, (x + 14, y + 21), 4)
    theme.text(surface, status, (x + 25, y + 21), 17, theme.TEXT, anchor="midleft")

    # Filter modes occupy two rows to remain readable at this panel width.
    y += 51
    theme.text(surface, "FILTER", (x, y), 15, theme.TEXT_DIM)
    theme.text(surface, f"{MODE_LABELS.get(s['mode'], s['mode'])} ACTIVE",
               (x + w, y), 14, theme.ACCENT, anchor="topright")
    y += 21
    gap, chip_h = 5, 27
    chip_w = (w - gap * 2) // 3
    for i, mode in enumerate(MODES):
        row, col = divmod(i, 3)
        r = pygame.Rect(x + col * (chip_w + gap), y + row * (chip_h + 5), chip_w, chip_h)
        active = mode == s["mode"]
        _card(surface, r, fill=theme.ACCENT if active else (28, 32, 58),
              border=theme.ACCENT if active else (48, 54, 88), radius=7)
        theme.text(surface, MODE_LABELS[mode], r.center, 14,
                   theme.DARK if active else theme.TEXT_DIM, anchor="center")

    y += chip_h * 2 + 14
    toggle_gap = 7
    toggle_w = (w - toggle_gap) // 2
    for i, (label, enabled) in enumerate((
        ("G  ASSIST", s.get("assist_enabled", True)),
        ("H  DWELL", s.get("dwell_enabled", False)),
    )):
        r = pygame.Rect(x + i * (toggle_w + toggle_gap), y, toggle_w, 37)
        color = theme.GREEN if enabled else (78, 84, 112)
        _card(surface, r, fill=(24, 36, 48) if enabled else (27, 30, 52), border=color, radius=8)
        theme.text(surface, label, r.center, 16, color, anchor="center")

    y += 47
    theme.text(surface, "INTENT", (x, y), 15, theme.TEXT_DIM)
    y += 20
    _card(surface, (x, y, w, 75), fill=(22, 26, 49))
    theme.text(surface, name, (x + 10, y + 7), 20, theme.TEXT)
    score = diag.get("score", decision.confidence if decision else 0.0)
    margin = diag.get("margin", 0.0)
    age_ms = diag.get("candidate_age", 0.0) * 1000.0
    theme.text(surface, f"Score {score:.2f}   Lead {margin:.2f}   {age_ms:.0f} ms",
               (x + 10, y + 34), 15, theme.TEXT_DIM)
    reason = str(diag.get("reason", "Waiting for movement"))
    if len(reason) > 34:
        reason = reason[:31] + "..."
    theme.text(surface, reason, (x + 10, y + 55), 13, theme.TEXT_DIM)

    y += 84
    theme.text(surface, "DWELL ACTIVATION", (x, y), 15, theme.TEXT_DIM)
    y += 20
    _card(surface, (x, y, w, 63), fill=(22, 26, 49))
    dwell_state = getattr(dwell, "state", "idle") if dwell else "idle"
    dwell_progress = getattr(dwell, "progress", 0.0) if dwell else 0.0
    if not s.get("dwell_enabled"):
        dwell_label, dwell_color = "Off · press H to enable", theme.TEXT_DIM
    elif dwell_state == "dwelling":
        remaining = s.get("dwell_seconds", 3.0) * (1.0 - dwell_progress)
        extra = s.get("dwell_enlargement_px", 0.0)
        suffix = f" · +{extra:.0f}px" if extra > 0 else ""
        dwell_label = f"Hold · {dwell_progress * 100:.0f}% · {remaining:.1f}s{suffix}"
        dwell_color = theme.GREEN
    elif dwell_state == "ready":
        dwell_label = f"Ready · {getattr(dwell, 'reason', 'return to target to activate')}"
        dwell_color = theme.AMBER
    elif dwell_state == "cancelled":
        dwell_label = getattr(dwell, "reason", "Cancelled")
        dwell_color = theme.AMBER
    elif dwell_state == "activated":
        dwell_label, dwell_color = "Activated · leave target to re-arm", theme.GREEN
    elif dwell_state == "blocked":
        dwell_label = getattr(dwell, "reason", "Action needs an explicit click")
        dwell_color = theme.AMBER
    else:
        dwell_label, dwell_color = f"Hover target · {s.get('dwell_seconds', 3.0):.1f} sec", theme.TEXT_DIM
        extra = s.get("dwell_enlargement_px", 0.0)
        if extra > 0:
            dwell_label += f" · +{extra:.0f}px"
    theme.text(surface, dwell_label, (x + 10, y + 8), 15, dwell_color)
    _progress(surface, pygame.Rect(x + 10, y + 35, w - 20, 9), dwell_progress,
              theme.GREEN if dwell_state in {"dwelling", "ready"} else theme.ACCENT)

    y += 72
    theme.text(surface, "LIVE SIGNALS", (x, y), 15, theme.TEXT_DIM)
    y += 20
    f = s["features"]
    velocity = (diag.get("velocity_x", 0.0) ** 2 + diag.get("velocity_y", 0.0) ** 2) ** 0.5
    cutoff = s.get("cutoff_hz")
    metrics = (
        ("CUTOFF", f"{cutoff:.1f} Hz" if cutoff is not None else "—"),
        ("POINTER SPEED", f"{velocity:.0f} px/s"),
        ("OSCILLATION", f"{f.oscillation_score:.2f}"),
        ("LATENCY", f"{s['processing_ms']:.1f} ms"),
    )
    metric_gap = 6
    metric_w = (w - metric_gap) // 2
    for i, (label, value) in enumerate(metrics):
        row, col = divmod(i, 2)
        r = pygame.Rect(x + col * (metric_w + metric_gap), y + row * 42, metric_w, 38)
        _card(surface, r, fill=(24, 28, 52), radius=7)
        theme.text(surface, label, (r.x + 7, r.y + 4), 12, theme.TEXT_DIM)
        theme.text(surface, value, (r.x + 7, r.y + 18), 16, theme.TEXT)

    # Keep the most useful controls visible without crowding the diagnostics.
    footer_y = rect.bottom - 100
    pygame.draw.line(surface, (48, 55, 94), (x, footer_y - 9), (x + w, footer_y - 9))
    theme.text(surface, "1–5 FILTER   G MAGNET   H DWELL", (x, footer_y), 13, theme.TEXT)
    theme.text(surface, "E TARGET FRAME   J STICKY   P PROFILE", (x, footer_y + 18), 12, theme.TEXT_DIM)
    theme.text(surface, "- / = DWELL   [ / ] STRENGTH   SPACE CANCEL", (x, footer_y + 36), 11, theme.TEXT_DIM)
    theme.text(surface, f"{s['magnetic_profile'].upper()} · {s.get('personalization_source', 'default').upper()} · frame {s.get('acquisition_margin_px', 18):.0f}px · pull {s.get('magnetic_displacement_px', 0):.1f}px",
               (x, footer_y + 54), 11, (125, 134, 170))
    theme.text(surface, "Prototype · simulated tremor", (x, rect.bottom - 18), 13,
               (100, 108, 143))
