"""Module 4: Fingerprint Radar. Owner: Asmita Bhuva."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from ktc_vis.cache.hdf5_store import CacheMiss, load_result
from ktc_vis.dashboard.theme import (
    ACCENT_SOFT,
    BORDER,
    CARD,
    CARD_ALT,
    CARD_STYLE,
    MUTED,
    SUCCESS,
    TEXT,
    WARN,
    DANGER,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CACHE_PATH = _PROJECT_ROOT / "data" / "cache" / "results.h5"

_ALGORITHMS = ["abc1", "cuqi8", "pnpe2e"]
_SAMPLES = ["a", "b", "c"]

_LABEL = {"abc1": "ABC1", "cuqi8": "CUQI8", "pnpe2e": "PNPE2E"}
_COLOR = {"abc1": "#f4a261", "cuqi8": "#a855f7", "pnpe2e": "#10b981"}
_SHAPE = {
    "abc1": "Wide / low",
    "cuqi8": "Narrow / tall",
    "pnpe2e": "Balanced",
}
_EXPECTED = {
    "abc1": "Fast, strong at easy levels, less robust across samples.",
    "cuqi8": "Slow but strong for hard levels and voltage residual.",
    "pnpe2e": "Strong for current sensitivity and voltage residual.",
}

# Nine radar axes in clockwise order
_AXES = [
    "Total SSIM",
    "Easy SSIM",
    "Hard SSIM",
    "IoU Conductive",
    "IoU Resistive",
    "Speed",
    "Robustness",
    "Voltage Residual",
    "Curr. Sensitivity",
]
_AXIS_HINT = [
    "Avg SSIM all levels · selected sample",
    "Avg SSIM levels 1–3 · selected sample",
    "Avg SSIM levels 5–7 · selected sample",
    "IoU conductive inclusion · selected level & sample",
    "IoU resistive inclusion · selected level & sample",
    "Inverted runtime · lower runtime = higher score",
    "Inverted SSIM std across samples at selected level",
    "Inverted voltage residual · lower = better (measurement)",
    "Equal use of all injection patterns (measurement)",
]
# Axes inverted before display (lower raw = better)
_INVERT = {"Speed", "Robustness", "Voltage Residual"}

# Component IDs
_RADAR_ID = "m4-radar"
_STATS_ID = "m4-stats"
_PROFILES_ID = "m4-profiles"
_INFO_ID = "m4-info"


# ── Public API ────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    """M4 layout: header → algorithm profile cards → radar + legend → heatmap."""
    return html.Div(
        [
            _header(),
            html.Div(id=_INFO_ID),
            html.Div(id=_PROFILES_ID, style={"display": "grid",
                                             "gridTemplateColumns": "repeat(3,1fr)",
                                             "gap": "14px"}),
            html.Div(
                [
                    html.Div(
                        [
                            _panel_label("Performance Polygon",
                                         "All algorithms — hover a vertex for the value"),
                            dcc.Graph(
                                id=_RADAR_ID,
                                figure=_empty_radar("Loading…"),
                                config={"displayModeBar": False, "responsive": True},
                                style={"height": "560px"},
                            ),
                        ],
                        style={**CARD_STYLE, "overflow": "hidden", "flex": "1 1 0"},
                    ),
                    _static_legend(),
                ],
                style={"display": "flex", "gap": "14px", "alignItems": "stretch"},
            ),
            html.Div(id=_STATS_ID),
        ],
        style={
            "padding": "24px 28px",
            "display": "flex",
            "flexDirection": "column",
            "gap": "16px",
            "minHeight": "100%",
        },
    )


def register_callbacks(app) -> None:  # noqa: ANN001
    """Wire sidebar controls to the radar, profile cards, and stats heatmap."""

    @app.callback(
        Output(_RADAR_ID, "figure"),
        Output(_STATS_ID, "children"),
        Output(_PROFILES_ID, "children"),
        Output(_INFO_ID, "children"),
        Input("sidebar-algorithm-dropdown", "value"),
        Input("sidebar-level-slider", "value"),
        Input("sidebar-sample-radio", "value"),
    )
    def _update(highlighted: str, level: int, sample: str) -> tuple:
        level = int(level)
        raw_data = _load_cache_data()

        has_data = any(bool(raw_data[alg]) for alg in _ALGORITHMS)
        if not has_data:
            msg = "No cached results found. Run the benchmark to populate the cache."
            empty_cards = [_profile_card(alg, {}, highlighted) for alg in _ALGORITHMS]
            return (
                _empty_radar("No data — run the benchmark first"),
                html.Div(),
                empty_cards,
                _banner(msg, "warn"),
            )

        normalised, raw_axis = _compute_axes(raw_data, level=level, sample=sample)
        fig = _build_radar(normalised, highlighted=highlighted,
                           level=level, sample=sample)
        stats = _build_heatmap(normalised, raw_axis)
        profiles = [_profile_card(alg, normalised[alg], highlighted)
                    for alg in _ALGORITHMS]
        info = _meas_info_banner(raw_axis)
        return fig, stats, profiles, info


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_cache_data(
    cache_path: Path = _CACHE_PATH,
) -> dict[str, dict[tuple[int, str], dict]]:
    """Return ``{alg: {(level, sample): metrics_dict}}`` for all combinations."""
    result: dict[str, dict] = {alg: {} for alg in _ALGORITHMS}
    for alg in _ALGORITHMS:
        for lv in range(1, 8):
            for s in _SAMPLES:
                try:
                    metrics, _ = load_result(alg, lv, s, cache_path)
                    result[alg][(lv, s)] = metrics
                except (CacheMiss, OSError):
                    pass
    return result


def _compute_axes(
    raw_data: dict[str, dict[tuple[int, str], dict]],
    level: int,
    sample: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Compute and normalise all 9 radar axis values.

    SSIM axes aggregate across ALL levels for the selected sample
    (their meaning is level-range based).  IoU, Speed use the selected
    (level, sample) point.  Robustness = inverted std across all samples
    at the selected level (so changing the level slider updates it).
    Measurement-domain axes use the selected (level, sample) when cached.

    Returns (normalised_0_1, raw_values).
    """
    raw: dict[str, dict[str, float]] = {}

    for alg in _ALGORITHMS:
        data = raw_data.get(alg, {})

        def _ssim(lvs: list[int]) -> float:
            vals = [data[(lv, sample)]["ssim"]
                    for lv in lvs if (lv, sample) in data and "ssim" in data[(lv, sample)]]
            return _mean(vals)

        point = data.get((level, sample), {})

        # Robustness: std of SSIM across all samples at the selected level
        ssim_across_samples = [
            data[(level, s)]["ssim"]
            for s in _SAMPLES
            if (level, s) in data and "ssim" in data[(level, s)]
        ]
        robustness_std = (
            float(np.std(ssim_across_samples))
            if len(ssim_across_samples) > 1
            else (0.0 if ssim_across_samples else float("nan"))
        )

        raw[alg] = {
            "Total SSIM": _ssim(list(range(1, 8))),
            "Easy SSIM": _ssim([1, 2, 3]),
            "Hard SSIM": _ssim([5, 6, 7]),
            "IoU Conductive": point.get("iou_conductive", float("nan")),
            "IoU Resistive": point.get("iou_resistive", float("nan")),
            "Speed": point.get("runtime", float("nan")),
            "Robustness": robustness_std,
            "Voltage Residual": point.get("voltage_residual", float("nan")),
            "Curr. Sensitivity": point.get("current_sensitivity", float("nan")),
        }

    # Min-max normalise per axis; apply inversion for "lower=better" axes
    normalised: dict[str, dict[str, float]] = {alg: {} for alg in _ALGORITHMS}
    for axis in _AXES:
        vals = {alg: raw[alg][axis] for alg in _ALGORITHMS}
        valid = [v for v in vals.values() if not np.isnan(v)]

        if not valid:
            for alg in _ALGORITHMS:
                normalised[alg][axis] = 0.0
            continue

        vmin, vmax = min(valid), max(valid)
        span = vmax - vmin if vmax != vmin else 1.0

        for alg in _ALGORITHMS:
            v = vals[alg]
            if np.isnan(v):
                normalised[alg][axis] = 0.0
                continue
            score = (v - vmin) / span
            if axis in _INVERT:
                score = 1.0 - score
            normalised[alg][axis] = round(float(score), 4)

    return normalised, raw


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


# ── Radar figure ──────────────────────────────────────────────────────────────

def _build_radar(
    normalised: dict[str, dict[str, float]],
    highlighted: str | None,
    level: int,
    sample: str,
) -> go.Figure:
    closed = _AXES + [_AXES[0]]
    fig = go.Figure()

    for alg in _ALGORITHMS:
        r = [normalised[alg].get(ax, 0.0) for ax in _AXES]
        r_closed = r + [r[0]]
        is_hi = highlighted is None or alg == highlighted
        fig.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=closed,
                fill="toself",
                fillcolor=_hex_rgba(_COLOR[alg], 0.15 if is_hi else 0.04),
                line=dict(
                    color=_COLOR[alg],
                    width=3.0 if alg == highlighted else 1.6,
                    dash="solid",
                ),
                opacity=1.0 if is_hi else 0.3,
                name=_LABEL[alg],
                hovertemplate=(
                    f"<b>{_LABEL[alg]}</b><br>"
                    "%{theta}: <b>%{r:.3f}</b><extra></extra>"
                ),
            )
        )

    # Axis domain divider: draw a faint arc over the measurement-domain axes
    # (axes 7–8, indices 7 and 8 = "Voltage Residual", "Curr. Sensitivity")
    fig.update_layout(
        polar=dict(
            bgcolor="#14142a",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["", "0.5", "", "1.0"],
                tickfont=dict(color=MUTED, size=9),
                gridcolor="#2a2a4a",
                linecolor="#2a2a4a",
                showticklabels=True,
            ),
            angularaxis=dict(
                tickfont=dict(color=TEXT, size=11.5, family="monospace"),
                linecolor="#2a2a4a",
                gridcolor="#2a2a4a",
                rotation=90,
                direction="clockwise",
            ),
        ),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        title=dict(
            text=f"Sample {sample.upper()} · Level {level}",
            x=0.5,
            font=dict(color=MUTED, size=11),
            pad=dict(t=4),
        ),
        legend=dict(
            font=dict(color=TEXT, size=12),
            bgcolor="rgba(10,10,25,0.7)",
            bordercolor=BORDER,
            borderwidth=1,
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.07,
            itemsizing="constant",
        ),
        margin=dict(l=80, r=80, t=44, b=60),
        height=560,
    )
    return fig


def _empty_radar(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        polar=dict(
            bgcolor="#14142a",
            radialaxis=dict(range=[0, 1], gridcolor="#2a2a4a",
                            linecolor="#2a2a4a", tickfont=dict(color=MUTED, size=9)),
            angularaxis=dict(tickfont=dict(color=MUTED, size=11),
                             linecolor="#2a2a4a", gridcolor="#2a2a4a"),
        ),
        paper_bgcolor=CARD,
        margin=dict(l=80, r=80, t=44, b=60),
        height=560,
        annotations=[dict(text=message, showarrow=False, xref="paper", yref="paper",
                          x=0.5, y=0.5, font=dict(color=MUTED, size=13))],
    )
    return fig


# ── Algorithm profile cards ───────────────────────────────────────────────────

def _profile_card(
    alg: str,
    scores: dict[str, float],
    highlighted: str | None,
) -> html.Div:
    is_hi = highlighted is None or alg == highlighted
    color = _COLOR[alg]

    if scores:
        top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        bars = [_mini_bar(ax, v, color) for ax, v in top3]
    else:
        bars = [html.Div("—", style={"color": MUTED, "fontSize": "12px"})]

    return html.Div(
        [
            # Coloured top accent bar
            html.Div(style={"height": "4px", "backgroundColor": color,
                            "borderRadius": "12px 12px 0 0"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                _LABEL[alg],
                                style={
                                    "color": color,
                                    "fontSize": "18px",
                                    "fontWeight": 700,
                                    "letterSpacing": "0.5px",
                                },
                            ),
                            html.Span(
                                _SHAPE[alg],
                                style={
                                    "display": "block",
                                    "color": MUTED,
                                    "fontSize": "11px",
                                    "fontStyle": "italic",
                                    "marginTop": "2px",
                                },
                            ),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    html.Div(
                        _EXPECTED[alg],
                        style={
                            "color": MUTED,
                            "fontSize": "11.5px",
                            "lineHeight": "1.5",
                            "borderTop": f"1px solid {BORDER}",
                            "paddingTop": "8px",
                            "marginBottom": "10px",
                        },
                    ),
                    html.Div(bars),
                ],
                style={"padding": "14px 16px 16px"},
            ),
        ],
        style={
            **CARD_STYLE,
            "borderRadius": "12px",
            "overflow": "hidden",
            "opacity": "1" if is_hi else "0.5",
            "transition": "opacity 0.2s",
        },
    )


def _mini_bar(label: str, score: float, color: str) -> html.Div:
    bar_pct = f"{max(score * 100, 2):.0f}%"
    return html.Div(
        [
            html.Div(
                [
                    html.Span(label, style={"color": MUTED, "fontSize": "10.5px",
                                            "minWidth": "112px", "display": "inline-block"}),
                    html.Span(f"{score:.2f}", style={"color": TEXT, "fontSize": "11px",
                                                     "fontWeight": 600,
                                                     "fontVariantNumeric": "tabular-nums"}),
                ],
                style={"display": "flex", "justifyContent": "space-between",
                       "marginBottom": "3px"},
            ),
            html.Div(
                html.Div(style={
                    "height": "100%",
                    "width": bar_pct,
                    "backgroundColor": color,
                    "borderRadius": "3px",
                    "opacity": "0.7",
                }),
                style={
                    "height": "5px",
                    "backgroundColor": BORDER,
                    "borderRadius": "3px",
                    "marginBottom": "7px",
                },
            ),
        ]
    )


# ── Heatmap stats table ───────────────────────────────────────────────────────

def _build_heatmap(
    normalised: dict[str, dict[str, float]],
    raw_axis: dict[str, dict[str, float]],
) -> html.Div:
    col_tmpl = "180px " + " ".join(["1fr"] * len(_ALGORITHMS))

    # Column header
    header_cells = [
        html.Div("Axis", style={
            "color": MUTED, "fontSize": "10px", "letterSpacing": "0.8px",
            "fontWeight": 600, "textTransform": "uppercase",
            "padding": "10px 0 10px 16px",
        })
    ]
    for alg in _ALGORITHMS:
        header_cells.append(html.Div(
            [
                html.Span(style={
                    "display": "inline-block", "width": "9px", "height": "9px",
                    "borderRadius": "2px", "backgroundColor": _COLOR[alg],
                    "marginRight": "7px", "verticalAlign": "middle",
                }),
                html.Span(_LABEL[alg], style={
                    "color": TEXT, "fontWeight": 700, "fontSize": "13px",
                }),
            ],
            style={"padding": "10px 8px", "textAlign": "center"},
        ))

    header = html.Div(header_cells, style={
        "display": "grid", "gridTemplateColumns": col_tmpl,
        "backgroundColor": CARD_ALT, "borderBottom": f"1px solid {BORDER}",
    })

    rows = []
    for i, axis in enumerate(_AXES):
        # Domain section label before the measurement axes
        if i == 7:
            rows.append(html.Div(
                "Measurement Domain",
                style={
                    "padding": "6px 16px",
                    "backgroundColor": "#1a1a35",
                    "color": ACCENT_SOFT,
                    "fontSize": "10px",
                    "fontWeight": 600,
                    "letterSpacing": "0.8px",
                    "textTransform": "uppercase",
                    "borderTop": f"2px solid {ACCENT_SOFT}",
                },
            ))

        scores = [normalised[alg].get(axis, 0.0) for alg in _ALGORITHMS]
        best = max(scores)
        hint = _AXIS_HINT[i]
        is_inverted = axis in _INVERT

        cells = [
            html.Div(
                [
                    html.Div(axis, style={"color": TEXT, "fontSize": "12px",
                                          "fontWeight": 500}),
                    html.Div(hint, style={"color": MUTED, "fontSize": "10px",
                                          "lineHeight": "1.4", "marginTop": "2px"}),
                    html.Div("(inverted)" if is_inverted else "",
                             style={"color": ACCENT_SOFT, "fontSize": "9.5px",
                                    "marginTop": "2px"}),
                ],
                style={"padding": "10px 8px 10px 16px"},
            )
        ]

        for alg_idx, alg in enumerate(_ALGORITHMS):
            score = scores[alg_idx]
            raw_v = raw_axis[alg].get(axis, float("nan"))
            is_best = abs(score - best) < 1e-6 and best > 0

            # Cell background tinted by algorithm color at score opacity
            cell_bg = _hex_rgba(_COLOR[alg], score * 0.18)
            score_color = _score_color(score)

            cells.append(html.Div(
                [
                    html.Div(
                        html.Div(style={
                            "height": "100%",
                            "width": f"{max(score * 100, 2):.0f}%",
                            "backgroundColor": _COLOR[alg],
                            "borderRadius": "2px",
                            "opacity": "0.45",
                        }),
                        style={
                            "height": "4px", "backgroundColor": BORDER,
                            "borderRadius": "2px", "marginBottom": "6px",
                        },
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"{score:.3f}",
                                style={
                                    "color": score_color,
                                    "fontWeight": 700 if is_best else 500,
                                    "fontSize": "14px",
                                    "fontVariantNumeric": "tabular-nums",
                                },
                            ),
                            html.Span(" ★" if is_best else "",
                                      style={"color": WARN, "fontSize": "11px"}),
                        ],
                        style={"marginBottom": "2px"},
                    ),
                    html.Div(
                        _fmt_raw(axis, raw_v),
                        style={"color": MUTED, "fontSize": "10px"},
                    ),
                ],
                style={
                    "padding": "10px 8px",
                    "textAlign": "center",
                    "backgroundColor": cell_bg,
                },
            ))

        row_border_top = f"2px solid {_COLOR['abc1']}22" if i == 0 else f"1px solid {BORDER}"
        rows.append(html.Div(cells, style={
            "display": "grid",
            "gridTemplateColumns": col_tmpl,
            "borderTop": row_border_top,
        }))

    title = html.Div(
        [
            html.Span("PERFORMANCE HEATMAP", style={
                "color": MUTED, "fontSize": "10px", "letterSpacing": "0.9px",
                "fontWeight": 600, "textTransform": "uppercase", "marginRight": "12px",
            }),
            html.Span("Normalised [0–1]. ★ = best per axis. Inverted axes: "
                      "Speed, Robustness, Voltage Residual.",
                      style={"color": MUTED, "fontSize": "11.5px"}),
        ],
        style={"padding": "12px 16px", "borderBottom": f"1px solid {BORDER}",
               "backgroundColor": CARD_ALT},
    )

    return html.Div([title, header, *rows], style={**CARD_STYLE, "overflow": "hidden"})


# ── Layout helpers ────────────────────────────────────────────────────────────

def _header() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span("M4", style={
                        "display": "inline-block", "padding": "2px 10px",
                        "borderRadius": "999px", "backgroundColor": "#7b61ff",
                        "color": "#fff", "fontSize": "11px", "fontWeight": 600,
                        "letterSpacing": "0.5px", "marginRight": "10px",
                    }),
                    html.Span("Fingerprint Radar", style={
                        "color": TEXT, "fontSize": "20px", "fontWeight": 600,
                    }),
                ]
            ),
            html.P(
                "Nine-axis radar comparing each algorithm's performance polygon. "
                "SSIM axes aggregate across all levels for the selected sample; "
                "IoU and Speed respond to the selected level. "
                "Higher is always better — Speed, Robustness, and Voltage Residual "
                "are inverted.",
                style={"color": MUTED, "margin": "8px 0 0",
                       "fontSize": "13px", "lineHeight": "1.5"},
            ),
        ],
        style={**CARD_STYLE, "padding": "16px 20px", "borderLeft": "3px solid #7b61ff"},
    )


def _panel_label(title: str, subtitle: str) -> html.Div:
    return html.Div(
        [
            html.Span(title, style={
                "color": TEXT, "fontWeight": 600, "fontSize": "13.5px",
            }),
            html.Span(subtitle, style={
                "color": MUTED, "fontSize": "11.5px",
            }),
        ],
        style={
            "display": "flex", "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "10px 16px",
            "borderBottom": f"1px solid {BORDER}",
        },
    )


def _static_legend() -> html.Div:
    """Non-reactive axis guide displayed next to the radar."""
    items = []
    for i, (ax, hint) in enumerate(zip(_AXES, _AXIS_HINT)):
        inv = ax in _INVERT
        domain_tag = html.Span(
            "meas.",
            style={"color": ACCENT_SOFT, "fontSize": "9.5px",
                   "fontWeight": 600, "marginLeft": "6px"},
        ) if i >= 7 else None

        items.append(html.Div(
            [
                html.Div(
                    [
                        html.Span(ax, style={
                            "color": TEXT, "fontSize": "12px", "fontWeight": 500,
                        }),
                        domain_tag,
                        html.Span(" ↕" if inv else "",
                                  style={"color": ACCENT_SOFT, "fontSize": "9.5px"}),
                    ],
                    style={"display": "flex", "alignItems": "center",
                           "gap": "4px", "marginBottom": "2px"},
                ),
                html.Div(hint, style={
                    "color": MUTED, "fontSize": "10.5px", "lineHeight": "1.4",
                }),
            ],
            style={
                "padding": "8px 0",
                "borderBottom": f"1px solid {BORDER}",
            },
        ))
    items[-1].style["borderBottom"] = "none"

    return html.Div(
        [
            html.Div("AXES GUIDE", style={
                "color": MUTED, "fontSize": "10px", "letterSpacing": "0.8px",
                "fontWeight": 600, "textTransform": "uppercase",
                "padding": "12px 14px 8px",
                "borderBottom": f"1px solid {BORDER}",
            }),
            html.Div(items, style={"padding": "4px 14px 14px", "overflowY": "auto"}),
        ],
        style={
            **CARD_STYLE,
            "width": "230px",
            "flexShrink": 0,
        },
    )


def _banner(message: str, kind: str = "info") -> html.Div:
    palette = {
        "info": ("#1f3a5f", "#5b8def", "#cfe0ff"),
        "warn": ("#3a2a1a", WARN, "#f4c870"),
        "error": ("#3a1f25", DANGER, "#ffd1d8"),
    }
    bg, border, fg = palette.get(kind, palette["info"])
    return html.Div(message, style={
        "backgroundColor": bg, "border": f"1px solid {border}",
        "color": fg, "padding": "10px 14px",
        "borderRadius": "10px", "fontSize": "12.5px",
    })


def _meas_info_banner(raw: dict[str, dict[str, float]]) -> html.Div | None:
    missing = all(
        np.isnan(raw[alg].get("Voltage Residual", float("nan")))
        for alg in _ALGORITHMS
    )
    if missing:
        return _banner(
            "Voltage Residual and Current Sensitivity Balance are not yet cached "
            "and show 0 on the radar. They will appear once measurement-domain "
            "metrics are computed by the benchmark runner.",
            "info",
        )
    return None


# ── Colour & formatting utilities ─────────────────────────────────────────────

def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _score_color(score: float) -> str:
    if score >= 0.67:
        return SUCCESS
    if score >= 0.34:
        return WARN
    return DANGER


def _fmt_raw(axis: str, value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if axis == "Speed":
        return f"{value:.2f} s"
    if axis == "Robustness":
        return f"std {value:.3f}"
    return f"raw {value:.3f}"
