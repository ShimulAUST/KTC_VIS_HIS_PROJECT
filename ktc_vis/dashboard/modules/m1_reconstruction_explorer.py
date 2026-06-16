"""Module 1: Reconstruction Explorer.

Two-row layout (hero comparison + analysis row) for the currently
selected ``(algorithm, level, sample)`` triple:

    Row 1 (hero):     Ground Truth   |   Reconstruction
    Row 2 (analysis): Segmentation   |   Error Overlay

Backed by :class:`ReferenceOutputAdapter` (precomputed ``.mat`` files).
"""

from __future__ import annotations

import logging

import numpy as np
from dash import Input, Output, dcc, html

from ktc_vis.adapters import ReferenceOutputAdapter
from ktc_vis.data.loader import KTCDataLoader
from ktc_vis.dashboard.theme import (
    ACCENT,
    BORDER,
    CARD_STYLE,
    DANGER,
    MUTED,
    SUCCESS,
    TEXT,
    WARN,
)
from ktc_vis.utils.figures import (
    CLASS_COLORS,
    CLASS_LABELS,
    ERROR_COLORS,
    empty_figure,
    error_overlay_figure,
    segmentation_figure,
)

logger = logging.getLogger(__name__)

# Module-scoped singletons (cheap; no heavy state)
_LOADER = KTCDataLoader()
_ADAPTERS: dict[str, ReferenceOutputAdapter] = {}


def _get_adapter(name: str) -> ReferenceOutputAdapter:
    if name not in _ADAPTERS:
        _ADAPTERS[name] = ReferenceOutputAdapter(name)
    return _ADAPTERS[name]


# ── IDs (prefix all M1 ids with ``m1-`` per coding standards) ──
_GT_ID = "m1-gt-image"
_RECON_ID = "m1-recon-image"
_SEG_ID = "m1-seg-image"
_ERR_ID = "m1-error-image"
_CHIPS_ID = "m1-chips"
_BANNER_ID = "m1-banner"


_PANEL_HEADER_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "space-between",
    "padding": "10px 14px",
    "borderBottom": f"1px solid {BORDER}",
}


def layout() -> html.Div:
    """Two-row layout: hero comparison row + analysis row."""
    return html.Div(
        [
            _header(),
            _chip_row(),
            html.Div(id=_BANNER_ID),
            _row_label("Comparison", "Ground truth vs. algorithm output"),
            _hero_row(),
            _row_label("Analysis", "Segmentation and per-pixel error"),
            _analysis_row(),
            _legend_row(),
        ],
        style={
            "padding": "24px 28px",
            "minHeight": "100%",
            "display": "flex",
            "flexDirection": "column",
            "gap": "14px",
        },
    )


# ── Layout building blocks ────────────────────────────────────


def _header() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "M1",
                        style={
                            "display": "inline-block",
                            "padding": "2px 10px",
                            "borderRadius": "999px",
                            "backgroundColor": ACCENT,
                            "color": "#fff",
                            "fontSize": "11px",
                            "fontWeight": 600,
                            "letterSpacing": "0.5px",
                            "marginRight": "10px",
                        },
                    ),
                    html.Span(
                        "Reconstruction Explorer",
                        style={"color": TEXT, "fontSize": "20px", "fontWeight": 600},
                    ),
                ]
            ),
            html.P(
                "Two-row view: compare the ground truth against the chosen "
                "algorithm's reconstruction, then inspect its segmentation "
                "and per-pixel error map.",
                style={"color": MUTED, "margin": "8px 0 0", "fontSize": "13px",
                       "lineHeight": "1.5"},
            ),
        ],
        style={**CARD_STYLE, "padding": "16px 20px"},
    )


def _chip_row() -> html.Div:
    return html.Div(
        id=_CHIPS_ID,
        children=[_chip("status", "loading…")],
        style={"display": "flex", "flexWrap": "wrap", "gap": "8px"},
    )


def _chip(label: str, value: str, accent: str | None = None) -> html.Div:
    return html.Div(
        [
            html.Span(
                label,
                style={
                    "color": MUTED,
                    "fontSize": "10.5px",
                    "letterSpacing": "0.6px",
                    "textTransform": "uppercase",
                    "marginRight": "8px",
                },
            ),
            html.Span(
                value,
                style={
                    "color": accent or TEXT,
                    "fontSize": "13px",
                    "fontWeight": 600,
                    "fontVariantNumeric": "tabular-nums",
                },
            ),
        ],
        style={
            **CARD_STYLE,
            "padding": "8px 12px",
            "display": "inline-flex",
            "alignItems": "center",
        },
    )


def _row_label(title: str, subtitle: str) -> html.Div:
    return html.Div(
        [
            html.Span(
                title,
                style={
                    "color": TEXT,
                    "fontWeight": 600,
                    "fontSize": "12px",
                    "letterSpacing": "0.8px",
                    "textTransform": "uppercase",
                    "marginRight": "10px",
                },
            ),
            html.Span(subtitle, style={"color": MUTED, "fontSize": "12px"}),
        ],
        style={
            "display": "flex",
            "alignItems": "baseline",
            "padding": "4px 2px 0",
            "borderBottom": f"1px dashed {BORDER}",
            "paddingBottom": "6px",
            "marginTop": "4px",
        },
    )


def _hero_row() -> html.Div:
    """Primary comparison: GT and Reconstruction side-by-side (large)."""
    return html.Div(
        [
            _panel("Ground Truth", "Reference segmentation", _GT_ID,
                   badge="GT", badge_color=SUCCESS, height=440),
            _panel("Reconstruction", "Algorithm output", _RECON_ID,
                   badge="Recon", badge_color="#5b8def", height=440),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))",
            "gap": "16px",
        },
    )


def _analysis_row() -> html.Div:
    """Secondary analysis: Segmentation + Error overlay (smaller)."""
    return html.Div(
        [
            _panel("Segmentation", "Algorithm output (3-class)", _SEG_ID,
                   badge="Seg", badge_color="#a07bff", height=440),
            _panel("Error Overlay", "Per-pixel agreement vs. GT", _ERR_ID,
                   badge="Δ", badge_color=DANGER, height=440),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))",
            "gap": "16px",
        },
    )


def _panel(
    title: str,
    subtitle: str,
    graph_id: str,
    *,
    badge: str = "",
    badge_color: str = ACCENT,
    height: int = 340,
) -> html.Div:
    # 44px header + 16px inner padding + 2px border = 62px overhead
    card_height = height + 62
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                badge,
                                style={
                                    "display": "inline-block",
                                    "padding": "2px 8px",
                                    "borderRadius": "6px",
                                    "backgroundColor": badge_color,
                                    "color": "#fff",
                                    "fontSize": "10px",
                                    "fontWeight": 700,
                                    "letterSpacing": "0.4px",
                                    "marginRight": "10px",
                                },
                            ),
                            html.Span(
                                title,
                                style={"color": TEXT, "fontWeight": 600,
                                       "fontSize": "13.5px"},
                            ),
                        ]
                    ),
                    html.Span(
                        subtitle,
                        style={"color": MUTED, "fontSize": "11.5px"},
                    ),
                ],
                style=_PANEL_HEADER_STYLE,
            ),
            html.Div(
                dcc.Graph(
                    id=graph_id,
                    figure=empty_figure("Loading…"),
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": f"{height}px", "width": "100%"},
                ),
                style={
                    "padding": "6px 6px 10px",
                    "flex": "1",
                    "minHeight": "0",
                    "overflow": "hidden",
                },
            ),
        ],
        style={
            **CARD_STYLE,
            "height": f"{card_height}px",
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden",
        },
    )


def _legend_row() -> html.Div:
    def swatch(color: str) -> dict:
        return {
            "display": "inline-block",
            "width": "10px",
            "height": "10px",
            "borderRadius": "2px",
            "backgroundColor": color,
            "marginRight": "6px",
        }

    item_style = {
        "display": "inline-flex",
        "alignItems": "center",
        "color": TEXT,
        "fontSize": "12px",
        "marginRight": "18px",
    }
    label_style = {
        "color": MUTED,
        "fontSize": "11px",
        "letterSpacing": "0.6px",
        "textTransform": "uppercase",
        "marginRight": "14px",
    }
    classes = [
        html.Span([html.Span(style=swatch(CLASS_COLORS[k])), CLASS_LABELS[k]],
                  style=item_style)
        for k in (0, 1, 2)
    ]
    errors = [
        html.Span([html.Span(style=swatch(ERROR_COLORS["correct"])), "correct"],
                  style=item_style),
        html.Span([html.Span(style=swatch(ERROR_COLORS["false_positive"])),
                   "false positive"], style=item_style),
        html.Span([html.Span(style=swatch(ERROR_COLORS["false_negative"])),
                   "false negative"], style=item_style),
        html.Span([html.Span(style=swatch("#f1c40f")), "wrong class"],
                  style=item_style),
    ]
    return html.Div(
        [
            html.Div(
                [html.Span("Classes", style=label_style), *classes],
                style={"padding": "10px 14px",
                       "borderBottom": f"1px solid {BORDER}"},
            ),
            html.Div(
                [html.Span("Error overlay", style=label_style), *errors],
                style={"padding": "10px 14px"},
            ),
        ],
        style=CARD_STYLE,
    )


def _banner(message: str, kind: str = "info") -> html.Div:
    palette = {
        "info": ("#1f3a5f", "#5b8def", "#cfe0ff"),
        "warn": ("#3a2a1a", WARN, "#f4c870"),
        "error": ("#3a1f25", DANGER, "#ffd1d8"),
    }
    bg, border, fg = palette.get(kind, palette["info"])
    return html.Div(
        message,
        style={
            "backgroundColor": bg,
            "border": f"1px solid {border}",
            "color": fg,
            "padding": "10px 14px",
            "borderRadius": "10px",
            "fontSize": "12.5px",
        },
    )


# ── Callbacks ────────────────────────────────────────────────


def register_callbacks(app) -> None:  # noqa: ANN001
    """Wire the shared sidebar selectors to the four M1 panels."""

    @app.callback(
        Output(_GT_ID, "figure"),
        Output(_RECON_ID, "figure"),
        Output(_SEG_ID, "figure"),
        Output(_ERR_ID, "figure"),
        Output(_CHIPS_ID, "children"),
        Output(_BANNER_ID, "children"),
        Input("sidebar-algorithm-dropdown", "value"),
        Input("sidebar-level-slider", "value"),
        Input("sidebar-sample-radio", "value"),
    )
    def _update_panels(algorithm: str, level: int, sample: str):
        level = int(level)
        try:
            measurement = _LOADER.load(level=level, sample=sample)
            adapter = _get_adapter(algorithm)
            recon = adapter.reconstruct(measurement)
        except FileNotFoundError as exc:
            logger.warning("M1 data missing: %s", exc)
            empty = empty_figure("Data missing")
            chips = [_chip("status", "data missing", accent=DANGER)]
            return empty, empty, empty, empty, chips, _banner(str(exc), "error")
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            logger.exception("M1 update failed")
            empty = empty_figure("Error")
            chips = [_chip("status", "error", accent=DANGER)]
            return empty, empty, empty, empty, chips, _banner(f"Error: {exc}", "error")

        gt = measurement.ground_truth.astype(np.uint8)
        recon = recon.astype(np.uint8)

        n_inj, n_el = measurement.current_matrix.shape
        n_meas = measurement.voltage_matrix.size
        agreement = float((recon == gt).mean()) * 100.0

        chips = [
            _chip("algorithm", algorithm.upper(), accent="#cfe0ff"),
            _chip("level", f"L{level}", accent=WARN),
            _chip("sample", sample.upper(), accent="#cfe0ff"),
            _chip("electrodes", f"{n_el}"),
            _chip("injections", f"{n_inj}"),
            _chip("voltage samples", f"{n_meas:,}"),
            _chip("pixel agreement", f"{agreement:.1f}%",
                  accent=SUCCESS if agreement >= 90 else WARN),
        ]

        gt_fig = segmentation_figure(gt, title="Ground Truth")
        recon_fig = segmentation_figure(
            recon, title=f"Reconstruction · {algorithm.upper()} · L{level}"
        )
        seg_fig = segmentation_figure(recon, title=f"Segmentation · L{level}")
        err_fig = error_overlay_figure(recon, gt, title=f"Error · L{level}")

        return gt_fig, recon_fig, seg_fig, err_fig, chips, None
