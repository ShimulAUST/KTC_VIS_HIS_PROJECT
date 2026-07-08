"""Module 3: Side-by-Side Comparison Grid.

Three-column grid showing **all three algorithms** (ABC1, CUQI8, PNPE2E) for
the currently selected ``(level, sample)`` triple simultaneously.

Layout
------
    Row 1  — Algorithm reconstructions (3-column grid)
    Row 2  — Pairwise pixel-difference images (3 pairs)
    Row 3  — Voltage measurement chart  +  Metrics scorecard
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from ktc_vis.adapters import ReferenceOutputAdapter
from ktc_vis.data.loader import KTCDataLoader
from ktc_vis.dashboard.theme import (
    ACCENT,
    BORDER,
    CARD,
    CARD_STYLE,
    DANGER,
    MUTED,
    SUCCESS,
    TEXT,
    WARN,
)
from ktc_vis.utils.figures import (
    CLASS_COLORS,
    empty_figure,
)

logger = logging.getLogger(__name__)

# ── Project paths ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "ktc2023"
_CACHE_PATH = _PROJECT_ROOT / "data" / "cache" / "results.h5"
_SAMPLE_MAP = {"a": 1, "b": 2, "c": 3, "d": 4}

# ── Algorithm registry ─────────────────────────────────────────────────────────
_ALGORITHMS: list[str] = ["abc1", "cuqi8", "pnpe2e"]
_ALG_COLORS: dict[str, str] = {
    "abc1": "#5b8def",   # blue
    "cuqi8": "#e85d75",   # pink
    "pnpe2e": "#f4c870",   # gold
}

# ── Singletons ─────────────────────────────────────────────────────────────────
_LOADER = KTCDataLoader()
_ADAPTERS: dict[str, ReferenceOutputAdapter] = {}


def _get_adapter(name: str) -> ReferenceOutputAdapter:
    if name not in _ADAPTERS:
        _ADAPTERS[name] = ReferenceOutputAdapter(name)
    return _ADAPTERS[name]


# ── IDs (all prefixed with ``m3-``) ───────────────────────────────────────────
_CHIPS_ID = "m3-chips"
_BANNER_ID = "m3-banner"
# Reconstruction row
_RECON_IDS = {alg: f"m3-recon-{alg}" for alg in _ALGORITHMS}
# Pairwise diff row
_DIFF_PAIRS = [("abc1", "cuqi8"), ("abc1", "pnpe2e"), ("cuqi8", "pnpe2e")]
_DIFF_IDS = {(a, b): f"m3-diff-{a}-{b}" for a, b in _DIFF_PAIRS}
# Bottom row
_VOLTAGE_ID = "m3-voltage-chart"
_SCORE_ID = "m3-scorecard"

# ── Shared style helpers ───────────────────────────────────────────────────────
_PANEL_HDR = {
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "space-between",
    "padding": "10px 14px",
    "borderBottom": f"1px solid {BORDER}",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════════════

def layout() -> html.Div:
    """Three-section layout: reconstructions → diffs → voltage + scorecard."""
    return html.Div(
        [
            _header(),
            html.Div(id=_CHIPS_ID, children=[_chip("status", "loading…")],
                     style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}),
            html.Div(id=_BANNER_ID),

            # ── Row 1: reconstructions ────────────────────────────────────────
            _section_label(
                "Algorithm Reconstructions",
                "All three algorithms at the same level and sample — select any "
                "tab to inspect the highlighted algorithm in detail via M1.",
            ),
            _recon_grid(),

            # ── Row 2: pairwise differences ───────────────────────────────────
            _section_label(
                "Pairwise Pixel Differences",
                "Diverging map: purple = A classifies higher, gold = B classifies higher, "
                "dark = both agree.",
            ),
            _diff_grid(),

            # ── Row 3: voltage chart + metrics scorecard ──────────────────────
            _section_label(
                "Measurement Data & Metrics",
                "Left: shows how much voltage was measured at each electrode channel — "
                "each bar colour is a different current injection. "
                "Right: how well each algorithm performed, based on saved test results.",
            ),
            _bottom_row(),
        ],
        style={
            "padding": "24px 28px",
            "minHeight": "100%",
            "display": "flex",
            "flexDirection": "column",
            "gap": "14px",
        },
    )


# ── Section building blocks ────────────────────────────────────────────────────

def _header() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "M3",
                        style={
                            "display": "inline-block",
                            "padding": "2px 10px",
                            "borderRadius": "999px",
                            "backgroundColor": "#00b894",
                            "color": "#fff",
                            "fontSize": "11px",
                            "fontWeight": 600,
                            "letterSpacing": "0.5px",
                            "marginRight": "10px",
                        },
                    ),
                    html.Span(
                        "Side-by-Side Comparison Grid",
                        style={"color": TEXT, "fontSize": "20px", "fontWeight": 600},
                    ),
                ]
            ),
            html.P(
                "Displays ABC1, CUQI8, and PNPE2E for the same (level, sample) "
                "simultaneously. Pairwise difference images reveal where algorithms "
                "disagree. The voltage heatmap shows the raw measurement data each "
                "algorithm received, and the scorecard compares quality metrics "
                "from the benchmark cache.",
                style={"color": MUTED, "margin": "8px 0 0", "fontSize": "13px",
                       "lineHeight": "1.5"},
            ),
        ],
        style={**CARD_STYLE, "padding": "16px 20px",
               "borderLeft": "3px solid #00b894"},
    )


def _section_label(title: str, subtitle: str) -> html.Div:
    return html.Div(
        [
            html.Span(title, style={
                "color": TEXT, "fontWeight": 600, "fontSize": "12px",
                "letterSpacing": "0.8px", "textTransform": "uppercase",
                "marginRight": "10px",
            }),
            html.Span(subtitle, style={"color": MUTED, "fontSize": "12px"}),
        ],
        style={
            "display": "flex", "alignItems": "baseline",
            "padding": "4px 2px 0", "borderBottom": f"1px dashed {BORDER}",
            "paddingBottom": "6px", "marginTop": "4px",
        },
    )


def _recon_grid() -> html.Div:
    panels = [
        _image_panel(
            title=alg.upper(),
            subtitle=f"{alg.upper()} segmentation output",
            graph_id=_RECON_IDS[alg],
            badge_color=_ALG_COLORS[alg],
        )
        for alg in _ALGORITHMS
    ]
    return html.Div(
        panels,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(3, 1fr)",
            "gap": "14px",
        },
    )


def _diff_grid() -> html.Div:
    panels = []
    for a, b in _DIFF_PAIRS:
        label = f"{a.upper()} − {b.upper()}"
        panels.append(
            _image_panel(
                title=label,
                subtitle="Pixel-class difference (diverging scale −2 … +2)",
                graph_id=_DIFF_IDS[(a, b)],
                badge="Δ",
                badge_color="#a07bff",
                height=300,
            )
        )
    return html.Div(
        panels,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(3, 1fr)",
            "gap": "14px",
        },
    )


def _bottom_row() -> html.Div:
    voltage_panel = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "V",
                                style={
                                    "display": "inline-block",
                                    "padding": "2px 8px",
                                    "borderRadius": "6px",
                                    "backgroundColor": "#00b894",
                                    "color": "#fff",
                                    "fontSize": "10px",
                                    "fontWeight": 700,
                                    "marginRight": "10px",
                                },
                            ),
                            html.Span("Voltage Measurement Pattern",
                                      style={"color": TEXT, "fontWeight": 600,
                                             "fontSize": "13.5px"}),
                        ]
                    ),
                    html.Span("voltage at each channel · one colour per injection · up to 8 injections shown",
                              style={"color": MUTED, "fontSize": "11.5px"}),
                ],
                style=_PANEL_HDR,
            ),
            html.Div(
                dcc.Graph(
                    id=_VOLTAGE_ID,
                    figure=empty_figure("Loading…"),
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": "500px", "width": "100%"},
                ),
                style={"padding": "6px 6px 4px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("X-axis", style={
                                "color": ACCENT, "fontWeight": 700,
                                "fontSize": "10.5px", "marginRight": "4px",
                            }),
                            html.Span(
                                "Measurement channel — each electrode position around the phantom.",
                                style={"color": MUTED, "fontSize": "10.5px"},
                            ),
                        ],
                        style={"marginBottom": "4px"},
                    ),
                    html.Div(
                        [
                            html.Span("Y-axis", style={
                                "color": ACCENT, "fontWeight": 700,
                                "fontSize": "10.5px", "marginRight": "4px",
                            }),
                            html.Span(
                                "Voltage (V) recorded at that channel during the injection.",
                                style={"color": MUTED, "fontSize": "10.5px"},
                            ),
                        ],
                        style={"marginBottom": "4px"},
                    ),
                    html.Div(
                        [
                            html.Span("Colour", style={
                                "color": ACCENT, "fontWeight": 700,
                                "fontSize": "10.5px", "marginRight": "4px",
                            }),
                            html.Span(
                                "Each bar colour represents one injection — purple = early, "
                                "yellow = late. Up to 8 injections are evenly sampled from "
                                "the full set so the chart stays readable.",
                                style={"color": MUTED, "fontSize": "10.5px"},
                            ),
                        ],
                    ),
                ],
                style={
                    "padding": "8px 14px 12px",
                    "borderTop": f"1px dashed {BORDER}",
                    "lineHeight": "1.6",
                },
            ),
        ],
        style={
            **CARD_STYLE,
            "flex": "2",
            "display": "flex",
            "flexDirection": "column",
        },
    )

    scorecard_panel = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "★",
                                style={
                                    "display": "inline-block",
                                    "padding": "2px 8px",
                                    "borderRadius": "6px",
                                    "backgroundColor": WARN,
                                    "color": "#111",
                                    "fontSize": "10px",
                                    "fontWeight": 700,
                                    "marginRight": "10px",
                                },
                            ),
                            html.Span("Metrics Scorecard",
                                      style={"color": TEXT, "fontWeight": 600,
                                             "fontSize": "13.5px"}),
                        ]
                    ),
                    html.Span("from benchmark cache",
                              style={"color": MUTED, "fontSize": "11.5px"}),
                ],
                style=_PANEL_HDR,
            ),
            html.Div(
                id=_SCORE_ID,
                children=_scorecard_placeholder(),
                style={
                    "padding": "12px 14px",
                    "flex": "1",
                    "overflowY": "auto",
                },
            ),
        ],
        style={
            **CARD_STYLE,
            "flex": "1",
            "display": "flex",
            "flexDirection": "column",
        },
    )

    return html.Div(
        [voltage_panel, scorecard_panel],
        style={"display": "flex", "gap": "14px", "alignItems": "stretch"},
    )


def _scorecard_placeholder() -> list:
    return [
        html.P(
            "Run scripts/run_benchmark.py to populate the cache.",
            style={"color": MUTED, "fontSize": "12px", "fontStyle": "italic",
                   "marginTop": "20px", "textAlign": "center"},
        )
    ]


def _image_panel(
    title: str,
    subtitle: str,
    graph_id: str,
    *,
    badge: str = "",
    badge_color: str = ACCENT,
    height: int = 340,
) -> html.Div:
    """Reusable card containing a single dcc.Graph."""
    badge_text = badge if badge else title[:4]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                badge_text,
                                style={
                                    "display": "inline-block",
                                    "padding": "2px 8px",
                                    "borderRadius": "6px",
                                    "backgroundColor": badge_color,
                                    "color": "#fff" if badge_color != WARN else "#111",
                                    "fontSize": "10px",
                                    "fontWeight": 700,
                                    "letterSpacing": "0.4px",
                                    "marginRight": "10px",
                                },
                            ),
                            html.Span(title, style={"color": TEXT, "fontWeight": 600,
                                                    "fontSize": "13.5px"}),
                        ]
                    ),
                    html.Span(subtitle, style={"color": MUTED, "fontSize": "11.5px"}),
                ],
                style=_PANEL_HDR,
            ),
            html.Div(
                dcc.Graph(
                    id=graph_id,
                    figure=empty_figure("Loading…"),
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": f"{height}px", "width": "100%"},
                ),
                style={"padding": "6px 6px 10px", "flex": "1"},
            ),
        ],
        style={
            **CARD_STYLE,
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden",
        },
    )


def _chip(label: str, value: str, accent: str | None = None) -> html.Div:
    return html.Div(
        [
            html.Span(label, style={
                "color": MUTED, "fontSize": "10.5px", "letterSpacing": "0.6px",
                "textTransform": "uppercase", "marginRight": "8px",
            }),
            html.Span(value, style={
                "color": accent or TEXT, "fontSize": "13px",
                "fontWeight": 600, "fontVariantNumeric": "tabular-nums",
            }),
        ],
        style={
            **CARD_STYLE,
            "padding": "8px 12px",
            "display": "inline-flex",
            "alignItems": "center",
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


# ═══════════════════════════════════════════════════════════════════════════════
# Figure builders
# ═══════════════════════════════════════════════════════════════════════════════

_SEG_COLORSCALE = [
    [0.00, CLASS_COLORS[0]],
    [0.33, CLASS_COLORS[0]],
    [0.34, CLASS_COLORS[1]],
    [0.66, CLASS_COLORS[1]],
    [0.67, CLASS_COLORS[2]],
    [1.00, CLASS_COLORS[2]],
]

_DIFF_COLORSCALE = [
    [0.0, "#7b61ff"],   # −2: A much higher (purple)
    [0.25, "#a07bff"],   # −1
    [0.5, "#1e1e2f"],   # 0: identical (dark background)
    [0.75, "#f4c870"],   # +1
    [1.0, "#e8a030"],   # +2: B much higher (gold)
]


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, x=0.5, font=dict(color="#ddd", size=11)),
        paper_bgcolor=CARD,
        plot_bgcolor="#1a1a2e",
        margin=dict(l=8, r=8, t=28, b=8),
        autosize=True,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, autorange="reversed"),
        uirevision="constant",
    )


def _recon_figure(recon: np.ndarray, alg: str, level: int, sample: str) -> go.Figure:
    """3-class segmentation heatmap for one algorithm."""
    fig = go.Figure(
        go.Heatmap(
            z=recon.astype(float),
            colorscale=_SEG_COLORSCALE,
            zmin=0, zmax=2,
            showscale=False,
            hovertemplate="x:%{x}<br>y:%{y}<br>class:%{z}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(f"{alg.upper()} · L{level}/{sample.upper()}"))
    return fig


def _diff_figure(
    recon_a: np.ndarray, recon_b: np.ndarray,
    alg_a: str, alg_b: str, level: int, sample: str,
) -> go.Figure:
    """Diverging pixel-class difference: A − B, range −2…+2."""
    diff = recon_a.astype(np.int16) - recon_b.astype(np.int16)
    n_disagree = int(np.count_nonzero(diff))
    pct = n_disagree / diff.size * 100

    fig = go.Figure(
        go.Heatmap(
            z=diff,
            colorscale=_DIFF_COLORSCALE,
            zmin=-2, zmax=2,
            showscale=True,
            colorbar=dict(
                thickness=8, len=0.7,
                tickvals=[-2, -1, 0, 1, 2],
                tickfont=dict(color="#ccc", size=9),
            ),
            hovertemplate=(
                "x:%{x}<br>y:%{y}<br>A−B:%{z}"
                f"<extra>{alg_a.upper()}−{alg_b.upper()}</extra>"
            ),
        )
    )
    fig.update_layout(
        **_base_layout(
            f"{alg_a.upper()}−{alg_b.upper()} · L{level}/{sample.upper()} "
            f"· {pct:.1f}% disagree"
        )
    )
    return fig


_MAX_INJ_SHOWN = 8   # max injections shown per channel group
_MAX_CH_SHOWN = 32   # max channels on x-axis before range-slider kicks in


def _voltage_figure(measurement) -> go.Figure:
    """Grouped bar chart — x = channel, bars per channel = one per injection.

    Each channel gets a cluster of bars, one bar per (sampled) injection.
    Injections are colour-coded blue→red via the Plasma palette so you can
    immediately see how the voltage pattern shifts with the injection index.
    A range-slider appears when there are more than _MAX_CH_SHOWN channels.
    """
    V = measurement.voltage_matrix  # (n_inj, n_ch)
    n_inj, n_ch = V.shape

    # ── Sample injections when there are more than the cap ────────────────────
    if n_inj <= _MAX_INJ_SHOWN:
        inj_indices = list(range(n_inj))
    else:
        step = n_inj / _MAX_INJ_SHOWN
        inj_indices = sorted(
            set(min(int(round(i * step)), n_inj - 1) for i in range(_MAX_INJ_SHOWN))
        )

    # ── Sample channels when there are more than the cap ──────────────────────
    if n_ch <= _MAX_CH_SHOWN:
        ch_indices = list(range(n_ch))
    else:
        step = n_ch / _MAX_CH_SHOWN
        ch_indices = sorted(
            set(min(int(round(i * step)), n_ch - 1) for i in range(_MAX_CH_SHOWN))
        )

    x_labels = [f"Ch {ch + 1}" for ch in ch_indices]
    palette = pc.sample_colorscale("Plasma", max(len(inj_indices), 2))

    fig = go.Figure()

    # One trace per injection — all sharing the same channel x-axis positions
    for slot, inj_idx in enumerate(inj_indices):
        y_values = [float(V[inj_idx, ch]) for ch in ch_indices]
        color = palette[slot]
        fig.add_trace(
            go.Bar(
                name=f"Inj {inj_idx + 1}",
                x=x_labels,
                y=y_values,
                marker=dict(
                    color=color,
                    line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
                    opacity=0.88,
                ),
                hovertemplate=(
                    f"<b>Injection {inj_idx + 1}</b><br>"
                    "Channel: %{x}<br>"
                    "Voltage: %{y:.4f} V"
                    "<extra></extra>"
                ),
            )
        )

    inj_note = (
        f"{len(inj_indices)} of {n_inj} inj sampled"
        if n_inj > _MAX_INJ_SHOWN else f"{n_inj} injections"
    )
    ch_note = (
        f", {len(ch_indices)} of {n_ch} ch sampled"
        if n_ch > _MAX_CH_SHOWN else f" × {n_ch} channels"
    )

    show_rangeslider = n_ch > _MAX_CH_SHOWN

    fig.update_layout(
        barmode="group",
        bargap=0.18,
        bargroupgap=0.06,
        paper_bgcolor=CARD,
        plot_bgcolor="#1a1a2e",
        margin=dict(l=60, r=110, t=44, b=50),
        legend=dict(
            title=dict(text="Injection", font=dict(color=MUTED, size=9)),
            font=dict(color=TEXT, size=9),
            bgcolor="rgba(20,20,40,0.7)",
            bordercolor=BORDER,
            borderwidth=1,
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.01,
        ),
        title=dict(
            text=(
                f"Voltage Measurement Pattern · "
                f"L{measurement.level}/{measurement.sample.upper()} · "
                f"{inj_note}{ch_note}"
            ),
            x=0.5,
            xanchor="center",
            font=dict(color="#ddd", size=11),
        ),
        xaxis=dict(
            title=dict(text="Measurement Channel", font=dict(color=MUTED, size=10)),
            tickfont=dict(color=MUTED, size=8),
            tickangle=-40,
            gridcolor="#2e2e44",
            zeroline=False,
            rangeslider=dict(visible=show_rangeslider, thickness=0.05),
        ),
        yaxis=dict(
            title=dict(text="Voltage (V)", font=dict(color=MUTED, size=10)),
            tickfont=dict(color=MUTED, size=9),
            gridcolor="#2e2e44",
            zeroline=True,
            zerolinecolor="#555",
            zerolinewidth=1,
        ),
        uirevision="constant",
    )
    return fig


# ── Metrics scorecard ──────────────────────────────────────────────────────────

# (metric_key, display_label, higher_is_better, format_spec)
_METRIC_KEYS: list[tuple[str, str, bool, str]] = [
    # Image Quality
    ("ssim",                   "SSIM Score",           True,  ".3f"),
    ("ssim_min",               "Spatial SSIM (min)",   True,  ".3f"),
    # Shape Matching
    ("hausdorff",              "Hausdorff Dist (px)",  False, ".1f"),
    ("position_error",         "Position Error (px)",  False, ".1f"),
    ("resolution",             "Resolution (px)",      False, ".1f"),
    # Class Specific
    ("confusion_accuracy",     "Confusion Accuracy",   True,  ".3f"),
    ("iou_mean",               "Mean IoU",             True,  ".3f"),
    ("iou_water",              "IoU Water",            True,  ".3f"),
    ("iou_resistive",          "IoU Resistive",        True,  ".3f"),
    ("iou_conductive",         "IoU Conductive",       True,  ".3f"),
    ("dice_mean",              "Mean Dice",            True,  ".3f"),
    ("dice_water",             "Dice Water",           True,  ".3f"),
    ("dice_resistive",         "Dice Resistive",       True,  ".3f"),
    ("dice_conductive",        "Dice Conductive",      True,  ".3f"),
    # Data Efficiency
    ("runtime",                "Runtime (s)",          False, ".4f"),
    # Measurement Domain
    ("voltage_residual",       "Voltage Residual",     False, ".4f"),
    ("resistance_consistency", "Resistance Consist.",  True,  ".3f"),
    ("current_sensitivity",    "Current Sensitivity",  True,  ".3f"),
]

# Row indices (into _METRIC_KEYS) where a new section header should be rendered.
_METRIC_SECTIONS: dict[int, str] = {
    0:  "Image Quality",
    2:  "Shape Matching",
    5:  "Class Specific",
    14: "Data Efficiency",
    15: "Measurement Domain",
}

_EXPECTED_METRIC_KEYS: set[str] = {key for key, *_ in _METRIC_KEYS}

# Measurement-domain metrics were rewritten to be reconstruction-dependent
# (Born forward surrogate). Any cached value that still matches the legacy
# saturated pattern is treated as stale and recomputed.
_MEASUREMENT_DOMAIN_KEYS: tuple[str, ...] = (
    "voltage_residual", "resistance_consistency", "current_sensitivity",
)


def _measurement_domain_is_stale(metrics: dict) -> bool:
    """True if the cached measurement-domain triple looks like an older formula.

    Legacy formula  : voltage_residual >= 0.98, resistance_consistency == 0,
                      current_sensitivity == 0  (clip-to-zero saturation).
    First rewrite   : resistance_consistency and current_sensitivity used
                      ``clip(r, 0, 1)`` so values landed in [0, 1] with many
                      exact 0.0s. The current version rescales correlations
                      via ``(r+1)/2`` so no honest cached value should be
                      < 0.05 unless something is severely wrong.
    """
    vr = metrics.get("voltage_residual")
    rc = metrics.get("resistance_consistency")
    cs = metrics.get("current_sensitivity")
    if vr is None or rc is None or cs is None:
        return True
    if vr >= 0.98 and abs(rc) < 1e-9 and abs(cs) < 1e-9:
        return True
    if rc < 0.05 and cs < 0.05:
        return True
    return False


def _try_load_metrics(algorithm: str, level: int, sample: str) -> dict | None:
    """Return metrics from cache, augmenting any keys missing from older cache entries.

    Flow:
      1. Cache hit → use cached metrics, but if the entry pre-dates a newer
         metric (Dice, measurement-domain, etc.) recompute the missing scalars
         from the cached reconstruction and persist them back. The adapter is
         never re-run.
      2. Cache miss → fall back to the full MetricsEngine.compute_all path,
         which will invoke the adapter if necessary.
    """
    metrics: dict | None = None
    reconstruction = None

    try:
        from ktc_vis.cache.hdf5_store import load_result
        metrics, reconstruction = load_result(
            algorithm, level, sample, cache_path=_CACHE_PATH
        )
    except Exception:
        metrics = None

    # Full cache miss → compute from scratch
    if metrics is None:
        try:
            measurement = _LOADER.load(level=level, sample=sample)
            adapter = _get_adapter(algorithm)
            from ktc_vis.metrics.engine import MetricsEngine
            engine = MetricsEngine(adapter, cache_path=_CACHE_PATH)
            return engine.compute_all(measurement)
        except Exception as exc:
            logger.debug(
                "M3 metrics computation failed for %s L%d/%s: %s",
                algorithm, level, sample, exc,
            )
            return None

    # Cache hit → backfill any missing keys without re-running the adapter.
    # Also drop legacy saturated measurement-domain values so the new
    # surrogate-based formula overwrites them.
    missing = _EXPECTED_METRIC_KEYS - set(metrics.keys())
    if _measurement_domain_is_stale(metrics):
        missing |= set(_MEASUREMENT_DOMAIN_KEYS)
    if missing and reconstruction is not None:
        try:
            measurement = _LOADER.load(level=level, sample=sample)
            from ktc_vis.metrics.engine import MetricsEngine
            engine = MetricsEngine(_get_adapter(algorithm), cache_path=_CACHE_PATH)
            recomputed = engine._compute_metrics_from_reconstruction(
                measurement,
                reconstruction.astype(np.uint8),
                runtime=float(metrics.get("runtime", 0.0)),
            )
            for key in missing:
                if key in recomputed:
                    metrics[key] = recomputed[key]
            try:
                from ktc_vis.cache.hdf5_store import save_result
                save_result(
                    algorithm, level, sample, metrics, reconstruction,
                    cache_path=_CACHE_PATH,
                )
            except Exception as exc:
                logger.debug(
                    "M3 cache write-back failed for %s L%d/%s: %s",
                    algorithm, level, sample, exc,
                )
        except Exception as exc:
            logger.debug(
                "M3 metric backfill failed for %s L%d/%s: %s",
                algorithm, level, sample, exc,
            )

    return metrics


def _scorecard_children(
    level: int, sample: str, selected_alg: str
) -> list:
    """Build the metrics scorecard table rows."""
    all_metrics: dict[str, dict | None] = {
        alg: _try_load_metrics(alg, level, sample) for alg in _ALGORITHMS
    }

    if all(v is None for v in all_metrics.values()):
        return _scorecard_placeholder()

    rows: list = []

    # ── Header row ────────────────────────────────────────────────────────────
    header_cells = [
        html.Th("Metric", style=_th_style()),
    ] + [
        html.Th(
            alg.upper(),
            style={
                **_th_style(),
                "color": _ALG_COLORS[alg],
                "borderBottom": f"2px solid {_ALG_COLORS[alg]}",
                "fontWeight": 700 if alg == selected_alg else 500,
            },
        )
        for alg in _ALGORITHMS
    ]
    rows.append(html.Tr(header_cells))

    # ── Data rows ─────────────────────────────────────────────────────────────
    for idx, (key, label, higher_better, fmt) in enumerate(_METRIC_KEYS):
        # Section header
        if idx in _METRIC_SECTIONS:
            rows.append(html.Tr(html.Td(
                _METRIC_SECTIONS[idx].upper(),
                colSpan=len(_ALGORITHMS) + 1,
                style={
                    "padding": "10px 10px 4px",
                    "color": ACCENT,
                    "fontSize": "9.5px",
                    "fontWeight": 700,
                    "letterSpacing": "0.8px",
                    "borderTop": f"1px solid {BORDER}",
                },
            )))

        # Collect values
        vals: dict[str, float | None] = {
            alg: (all_metrics[alg].get(key) if all_metrics[alg] else None)
            for alg in _ALGORITHMS
        }
        numeric = [v for v in vals.values() if v is not None]
        if not numeric:
            dash_cell = html.Td("—", style=_td_style(False, False))
            rows.append(html.Tr(
                [html.Td(label, style=_td_label_style())] + [dash_cell for _ in _ALGORITHMS],
                style={"borderBottom": f"1px solid {BORDER}"},
            ))
            continue

        best = max(numeric) if higher_better else min(numeric)

        cells = [html.Td(label, style=_td_label_style())]
        for alg in _ALGORITHMS:
            v = vals[alg]
            if v is None:
                cells.append(html.Td("—", style=_td_style(False, False)))
            else:
                is_best = abs(v - best) < 1e-9
                cells.append(
                    html.Td(
                        [
                            html.Span(
                                format(v, fmt),
                                style={"fontVariantNumeric": "tabular-nums"},
                            ),
                            html.Span(
                                " ★",
                                style={"color": SUCCESS, "fontSize": "9px"},
                            ) if is_best else None,
                        ],
                        style=_td_style(is_best, alg == selected_alg),
                    )
                )
        rows.append(html.Tr(cells, style={"borderBottom": f"1px solid {BORDER}"}))

    if not rows:
        return _scorecard_placeholder()

    return [
        html.Table(
            rows,
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "fontSize": "12px",
            },
        ),
        html.P(
            f"★ = best for that metric · highlighted column = sidebar selection ({selected_alg.upper()})",
            style={"color": MUTED, "fontSize": "10px", "marginTop": "10px"},
        ),
    ]


def _th_style() -> dict:
    return {
        "padding": "8px 10px",
        "textAlign": "left",
        "color": MUTED,
        "fontWeight": 600,
        "fontSize": "10.5px",
        "letterSpacing": "0.5px",
        "textTransform": "uppercase",
        "borderBottom": f"1px solid {BORDER}",
    }


def _td_label_style() -> dict:
    return {
        "padding": "8px 10px",
        "color": MUTED,
        "fontSize": "11.5px",
    }


def _td_style(is_best: bool, is_selected: bool) -> dict:
    return {
        "padding": "8px 10px",
        "color": SUCCESS if is_best else TEXT,
        "fontWeight": 700 if is_best else 400,
        "backgroundColor": "#1f2a3a" if is_selected else "transparent",
        "borderRadius": "4px",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_reconstruction(alg: str, level: int, sample: str) -> np.ndarray | None:
    """Try cache first, then fall back to ReferenceOutputAdapter."""
    # 1. Cache
    try:
        from ktc_vis.cache.hdf5_store import load_result
        _, recon = load_result(alg, level, sample, cache_path=_CACHE_PATH)
        return recon.astype(np.uint8)
    except Exception:
        pass

    # 2. Live adapter
    try:
        measurement = _LOADER.load(level=level, sample=sample)
        adapter = _get_adapter(alg)
        return adapter.reconstruct(measurement).astype(np.uint8)
    except Exception as exc:
        logger.warning("M3 could not load %s L%s/%s: %s", alg, level, sample, exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════════

def register_callbacks(app) -> None:  # noqa: ANN001
    """Wire sidebar selectors to all M3 panels."""

    # Flat output list: recon figures + diff figures + voltage + scorecard + chips + banner
    recon_outputs = [Output(_RECON_IDS[alg], "figure") for alg in _ALGORITHMS]
    diff_outputs = [Output(_DIFF_IDS[pair], "figure") for pair in _DIFF_PAIRS]
    other_outputs = [
        Output(_VOLTAGE_ID, "figure"),
        Output(_SCORE_ID, "children"),
        Output(_CHIPS_ID, "children"),
        Output(_BANNER_ID, "children"),
    ]

    @app.callback(
        *recon_outputs,
        *diff_outputs,
        *other_outputs,
        Input("sidebar-level-slider", "value"),
        Input("sidebar-sample-radio", "value"),
        Input("sidebar-algorithm-dropdown", "value"),
    )
    def _update_all(level: int, sample: str, selected_alg: str):
        level = int(level)

        # ── 1. Load measurement (needed for voltage chart) ────────────────────
        measurement = None
        banner = None
        try:
            measurement = _LOADER.load(level=level, sample=sample)
        except FileNotFoundError as exc:
            logger.warning("M3 measurement missing: %s", exc)
            banner = _banner(str(exc), "error")
        except Exception as exc:
            logger.exception("M3 measurement load failed")
            banner = _banner(f"Measurement load error: {exc}", "warn")

        # ── 2. Load all three reconstructions ─────────────────────────────────
        recons: dict[str, np.ndarray | None] = {
            alg: _load_reconstruction(alg, level, sample)
            for alg in _ALGORITHMS
        }
        n_loaded = sum(1 for v in recons.values() if v is not None)

        # ── 3. Build reconstruction figures ───────────────────────────────────
        recon_figs = []
        for alg in _ALGORITHMS:
            r = recons[alg]
            if r is not None:
                recon_figs.append(_recon_figure(r, alg, level, sample))
            else:
                recon_figs.append(
                    empty_figure(f"No data for {alg.upper()}\nRun benchmark")
                )

        # ── 4. Build pairwise difference figures ──────────────────────────────
        diff_figs = []
        for alg_a, alg_b in _DIFF_PAIRS:
            ra = recons[alg_a]
            rb = recons[alg_b]
            if ra is not None and rb is not None:
                diff_figs.append(
                    _diff_figure(ra, rb, alg_a, alg_b, level, sample)
                )
            else:
                missing = alg_a if ra is None else alg_b
                diff_figs.append(
                    empty_figure(f"Missing: {missing.upper()}")
                )

        # ── 5. Voltage figure ─────────────────────────────────────────────────
        if measurement is not None:
            try:
                voltage_fig = _voltage_figure(measurement)
            except Exception as exc:
                logger.exception("M3 voltage figure failed")
                voltage_fig = empty_figure(f"Voltage chart error: {exc}")
        else:
            voltage_fig = empty_figure("Measurement data unavailable")

        # ── 6. Scorecard ──────────────────────────────────────────────────────
        scorecard = _scorecard_children(level, sample, selected_alg)

        # ── 7. Chips ──────────────────────────────────────────────────────────
        chips = [
            _chip("level", f"L{level}", accent=WARN),
            _chip("sample", sample.upper(), accent="#cfe0ff"),
            _chip("loaded", f"{n_loaded}/3",
                  accent=SUCCESS if n_loaded == 3 else DANGER),
        ]
        if measurement is not None:
            n_inj, n_ch = measurement.voltage_matrix.shape
            chips += [
                _chip("injections", str(n_inj)),
                _chip("V channels", str(n_ch)),
            ]
        # Per-algorithm agreement chips
        if measurement is not None:
            gt = measurement.ground_truth.astype(np.uint8)
            for alg in _ALGORITHMS:
                r = recons[alg]
                if r is not None:
                    agr = float((r == gt).mean()) * 100.0
                    chips.append(
                        _chip(
                            alg.upper(),
                            f"{agr:.1f}%",
                            accent=SUCCESS if agr >= 90 else (
                                WARN if agr >= 70 else DANGER
                            ),
                        )
                    )

        return (*recon_figs, *diff_figs, voltage_fig, scorecard, chips, banner)
