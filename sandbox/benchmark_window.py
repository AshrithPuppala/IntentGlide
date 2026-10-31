"""Benchmark window layout: results table (one row per mode) and a bar chart of one chosen metric.

Draws a BenchmarkModel. With no data it shows "-" cells and ghost bars; once the model has results the same
code fills in the table, highlights the best value per column, and draws real bars. The source
(none / simulated / live) is always printed so simulated runs are never mistaken for real ones.
"""
from __future__ import annotations

import pygame

from sandbox.models import METRIC_LABELS, METRICS, MODE_LABELS, MODES, BenchmarkModel, format_metric
from sandbox.widgets import Window
from visualization import theme

PAD = 18
MODE_COL = 120
HEAD_Y, ROWS_Y, ROW_H = 10, 40, 28
CHART_TITLE_Y, CHART_Y, CHART_H = 192, 214, 146
BUTTON_Y = 376
_BTN = dict(fill=(44, 50, 88), hover_fill=(66, 78, 142), border=None, size=19)
_RUN = dict(fill=(36, 120, 90), hover_fill=(58, 168, 124), border=None, size=20)
SOURCE_COLOR = {"none": theme.TEXT_DIM, "simulated": theme.AMBER, "live": theme.GREEN}


def attach_benchmark(win: Window, model: BenchmarkModel) -> BenchmarkModel:
    metric_btn = win.add_button("metric", "", PAD, BUTTON_Y, 250, 36, on_click=model.cycle_metric)
    metric_btn.style = _BTN
    win.add_button("run", "Run benchmark", win.w - PAD - 190, BUTTON_Y, 190, 36,
                   on_click=model.request_run).style = _RUN
    col_w = (win.w - 2 * PAD - MODE_COL) / len(METRICS)

    def draw_content(surface, w):
        top = w.y + w.TITLE_H
        metric_btn.label = f"Chart: {METRIC_LABELS[model.chart_metric]}  >"
        # ---- table
        theme.text(surface, "Mode", (w.x + PAD, top + HEAD_Y + 4), 18, theme.TEXT_DIM)
        for i, (_k, label, _kind, _b) in enumerate(METRICS):
            theme.text(surface, label, (w.x + PAD + MODE_COL + (i + 0.5) * col_w, top + HEAD_Y + 4), 18,
                       theme.TEXT_DIM, anchor="midtop")
        pygame.draw.line(surface, (60, 70, 130), (w.x + PAD, top + ROWS_Y - 6), (w.x + w.w - PAD, top + ROWS_Y - 6))
        best = {k: model.best_mode(k) for k, *_ in METRICS}
        for r, mode in enumerate(MODES):
            y = top + ROWS_Y + r * ROW_H
            if r % 2 == 0:
                theme.arect(surface, (255, 255, 255, 10), (w.x + PAD - 6, y - 2, w.w - 2 * PAD + 12, ROW_H - 2), 5)
            theme.text(surface, MODE_LABELS[mode], (w.x + PAD, y + 3), 20, theme.TEXT)
            for i, (k, *_rest) in enumerate(METRICS):
                value = model.rows.get(mode, {}).get(k)
                color = theme.GREEN if best[k] == mode else (theme.TEXT if value is not None else theme.TEXT_DIM)
                theme.text(surface, format_metric(k, value), (w.x + PAD + MODE_COL + (i + 0.5) * col_w, y + 3),
                           19, color, anchor="midtop")
        # ---- chart
        theme.text(surface, f"{METRIC_LABELS[model.chart_metric]} by mode", (w.x + PAD, top + CHART_TITLE_Y), 20,
                   theme.TEXT)
        label = {"none": "no data yet", "simulated": "SIMULATED data", "live": "live data"}[model.source]
        theme.text(surface, f"Source: {label}", (w.x + w.w - PAD, top + CHART_TITLE_Y + 2), 18,
                   SOURCE_COLOR[model.source], anchor="topright")
        box = pygame.Rect(w.x + PAD, top + CHART_Y, w.w - 2 * PAD, CHART_H)
        theme.rrect(surface, (14, 16, 34), box, 8, (52, 58, 100))
        key = model.chart_metric
        values = {m: model.rows.get(m, {}).get(key) for m in MODES}
        kind = next(kd for mk, _, kd, _ in METRICS if mk == key)
        known = [v for v in values.values() if v is not None]
        scale = 1.0 if kind == "pct" else (max(known) if known and max(known) > 0 else 1.0)
        slot, base, max_h = box.w / len(MODES), box.bottom - 24, box.h - 24 - 26
        for i, mode in enumerate(MODES):
            cx = box.x + (i + 0.5) * slot
            bar_w, value = 46, values[mode]
            theme.text(surface, MODE_LABELS[mode], (cx, box.bottom - 20), 17, theme.TEXT_DIM, anchor="midtop")
            if value is None:                                            # ghost bar: waiting for data
                theme.rrect(surface, (26, 30, 58), (cx - bar_w / 2, base - 0.35 * max_h, bar_w, 0.35 * max_h), 5,
                            (60, 70, 130))
                continue
            h = max(3.0, max_h * min(value / scale, 1.0))
            color = theme.GREEN if best[key] == mode else theme.ACCENT
            theme.rrect(surface, color, (cx - bar_w / 2, base - h, bar_w, h), 5)
            theme.text(surface, format_metric(key, value), (cx, base - h - 4), 17, theme.TEXT, anchor="midbottom")

    win.on_draw = draw_content
    return model
