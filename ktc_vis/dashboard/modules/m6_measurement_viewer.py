"""Module 6: Measurement Domain Viewer. Owner: Asmita Bhuva.

Explores the raw electrical measurements independently of any
reconstruction, for the selected ``(level, sample)``:

    Current panel    — polar bar of the injected current pattern for one
                       injection step (angle = electrode position 1–32,
                       bar height = |I|). Steppable / animatable.
    Voltage panel    — polar plot of measured electrode voltages for the
                       selected injection (with empty-tank reference
                       overlay) + a difference chart (Uel − Uelref), the
                       signal ABC1 takes as input and CUQI8 fits.
    Resistance panel — R = V/I scatter for the selected injection + a 2D
                       injection × measurement-pair resistance heatmap.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.io
from dash import Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from ktc_vis.adapters.base import KTCMeasurement
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
from ktc_vis.data.loader import KTCDataLoader
from ktc_vis.data.subsampler import subsample_electrodes, subsample_measurement
from ktc_vis.utils.figures import empty_figure

logger = logging.getLogger(__name__)

_LOADER = KTCDataLoader()
_REF_MAT = Path(_LOADER.measurements_dir) / "ref.mat"

_N_TANK_ELECTRODES = 32  # physical electrodes on the KTC2023 tank rim

# ── IDs (all prefixed ``m6-`` per coding standards) ───────────
_CURRENT_ID = "m6-current-polar"
_VOLTAGE_POLAR_ID = "m6-voltage-polar"
_VOLTAGE_DIFF_ID = "m6-voltage-diff"
_RES_SCATTER_ID = "m6-resistance-scatter"
_RES_HEATMAP_ID = "m6-resistance-heatmap"
_INJ_SLIDER_ID = "m6-injection-slider"
_PLAY_BTN_ID = "m6-play-btn"
_INTERVAL_ID = "m6-interval"
_PLAYING_ID = "m6-playing"
_CHIPS_ID = "m6-chips"
_BANNER_ID = "m6-banner"

_POLAR_BG = "#14142a"
_GRID = "#2a2a4a"

_PANEL_HEADER_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "space-between",
    "padding": "10px 14px",
    "borderBottom": f"1px solid {BORDER}",
}


# ── Data access ───────────────────────────────────────────────


@lru_cache(maxsize=32)
def _load_measurement(level: int, sample: str) -> KTCMeasurement:
    """Load one (level, sample) measurement, cached for the app lifetime."""
    return _LOADER.load(level=level, sample=sample)


@lru_cache(maxsize=8)
def _load_reference(level: int) -> KTCMeasurement | None:
    """Load the empty-tank reference, subsampled exactly like a measurement.

    The reference shares the injection protocol (``Injref``) and measurement
    pattern (``Mpat``) with the sample measurements, so running it through
    :func:`subsample_measurement` keeps it row/column-aligned with the
    measurement at every level — an exact difference, not a sliced
    approximation.

    Args:
        level: Difficulty level, 1–7.

    Returns:
        Reference :class:`KTCMeasurement`, or ``None`` if ``ref.mat`` is
        missing or malformed.
    """
    if not _REF_MAT.exists():
        return None
    try:
        ref = scipy.io.loadmat(str(_REF_MAT))
        inj = ref["Injref"]
        mpat = ref["Mpat"].astype(np.int16)
        uel = np.asarray(ref["Uelref"], dtype=np.float64).flatten()
    except Exception:  # pragma: no cover - corrupt file surfaced as overlay-less UI
        logger.exception("Failed to read %s", _REF_MAT)
        return None

    n_inj = inj.shape[1]
    voltage = uel.reshape(n_inj, len(uel) // n_inj)
    current = inj.T
    magnitude = np.abs(current).max(axis=1, keepdims=True)
    magnitude = np.where(magnitude == 0, 1.0, magnitude)

    measurement = KTCMeasurement(
        current_matrix=current,
        voltage_matrix=voltage,
        resistance_matrix=voltage / magnitude,
        ground_truth=np.zeros((256, 256), dtype=np.uint8),
        level=1,
        sample="ref",
    )
    measurement.__dict__["mpat"] = mpat
    if level != 1:
        measurement = subsample_measurement(measurement, level)
    return measurement


def _electrode_angles(level: int) -> tuple[list[int], list[float]]:
    """Return (active electrode indices, their angles on the 32-slot rim)."""
    active = subsample_electrodes(level)
    return active, [(360.0 * e / _N_TANK_ELECTRODES) % 360 for e in active]


def _pair_angles(measurement: KTCMeasurement, level: int) -> list[float]:
    """Circular-mean angle of each measurement pair (Mpat column).

    Each voltage column corresponds to one Mpat column whose two non-zero
    rows identify the measuring electrode pair (in active-electrode space).

    Args:
        measurement: Measurement carrying an ``mpat`` attribute.
        level: Difficulty level, used to map active rows → rim positions.

    Returns:
        One angle in degrees per voltage column.
    """
    n_cols = measurement.voltage_matrix.shape[1]
    mpat = measurement.__dict__.get("mpat")
    if mpat is None or mpat.shape[1] != n_cols:
        return list(np.linspace(0.0, 360.0, n_cols, endpoint=False))

    active = subsample_electrodes(level)
    angles: list[float] = []
    for c in range(n_cols):
        rows = np.nonzero(mpat[:, c])[0]
        rad = np.deg2rad([360.0 * active[r] / _N_TANK_ELECTRODES for r in rows])
        mean = np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))
        angles.append(float(mean % 360))
    return angles


# ── Figure builders (pure; unit-tested) ───────────────────────


def _polar_layout(title: str, legend: bool = True) -> dict:
    """Shared dark polar layout matching the M5 polar styling."""
    return dict(
        title=dict(text=title, x=0.5, font=dict(color="#ddd", size=13)),
        paper_bgcolor=CARD,
        polar=dict(
            bgcolor=_POLAR_BG,
            radialaxis=dict(tickfont=dict(color=MUTED, size=9),
                            gridcolor=_GRID, linecolor=_GRID,
                            angle=90, tickangle=90),
            angularaxis=dict(tickfont=dict(color=TEXT, size=10),
                             gridcolor=_GRID, linecolor=_GRID,
                             direction="counterclockwise", rotation=0),
        ),
        showlegend=legend,
        legend=dict(font=dict(color=TEXT, size=11),
                    bgcolor="rgba(0,0,0,0.45)",
                    bordercolor=BORDER, borderwidth=1,
                    orientation="h", x=0.5, xanchor="center", y=-0.12),
        margin=dict(l=30, r=30, t=40, b=46),
        uirevision="m6",
    )


def _cartesian_layout(title: str, xtitle: str, ytitle: str) -> dict:
    """Shared dark cartesian layout for the difference / scatter charts."""
    axis = dict(
        gridcolor=_GRID, linecolor=_GRID, zerolinecolor="#3a3a55",
        tickfont=dict(color=MUTED, size=10),
    )
    return dict(
        title=dict(text=title, x=0.5, font=dict(color="#ddd", size=13)),
        paper_bgcolor=CARD,
        plot_bgcolor=_POLAR_BG,
        xaxis={**axis, "title": dict(text=xtitle, font=dict(color=MUTED, size=11))},
        yaxis={**axis, "title": dict(text=ytitle, font=dict(color=MUTED, size=11))},
        showlegend=True,
        legend=dict(font=dict(color=TEXT, size=11),
                    bgcolor="rgba(0,0,0,0.45)",
                    bordercolor=BORDER, borderwidth=1,
                    orientation="h", x=0.5, xanchor="center", y=-0.28),
        margin=dict(l=50, r=20, t=40, b=40),
        uirevision="m6",
    )


def current_polar_figure(
    measurement: KTCMeasurement, level: int, inj_idx: int
) -> go.Figure:
    """Polar bar chart of one injection pattern.

    Angle = electrode position on the 32-slot rim; bar height = |I|.
    Source electrode (+) drawn red, sink (−) blue; electrodes removed at
    this level are marked with gray crosses so coverage gaps are visible.
    """
    active, angles = _electrode_angles(level)
    currents = np.asarray(measurement.current_matrix[inj_idx], dtype=np.float64)

    colors = [DANGER if c > 0 else ACCENT if c < 0 else "rgba(90,90,120,0.25)"
              for c in currents]
    rmax = float(np.abs(currents).max()) or 1.0

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=np.abs(currents),
        theta=angles,
        width=[360.0 / _N_TANK_ELECTRODES * 0.85] * len(angles),
        marker=dict(color=colors, line=dict(color="#1e1e2f", width=0.5)),
        customdata=np.stack([np.array(active) + 1, currents], axis=-1),
        hovertemplate=("electrode %{customdata[0]}<br>"
                       "I = %{customdata[1]:.3f} mA<extra></extra>"),
        name="injected current",
    ))

    removed = [e for e in range(_N_TANK_ELECTRODES) if e not in set(active)]
    if removed:
        fig.add_trace(go.Scatterpolar(
            r=[rmax * 1.12] * len(removed),
            theta=[(360.0 * e / _N_TANK_ELECTRODES) % 360 for e in removed],
            mode="markers",
            marker=dict(size=9, symbol="x-thin",
                        line=dict(color="#9aa0b4", width=1.4),
                        color="rgba(154,160,180,0.55)"),
            name=f"removed electrode ({len(removed)})",
            hovertemplate="removed electrode<extra></extra>",
        ))

    n_inj = measurement.current_matrix.shape[0]
    fig.update_layout(**_polar_layout(
        f"Injection {inj_idx + 1} / {n_inj} · current pattern"))
    fig.update_polars(radialaxis_range=[0, rmax * 1.25])
    return fig


def voltage_polar_figure(
    measurement: KTCMeasurement,
    reference: KTCMeasurement | None,
    level: int,
    inj_idx: int,
) -> go.Figure:
    """Polar plot of measured voltages for one injection, with reference.

    Each point is one measurement pair, placed at the circular-mean angle
    of its two electrodes. The dashed overlay is the empty-tank reference.
    """
    angles = _pair_angles(measurement, level)
    v = np.asarray(measurement.voltage_matrix[inj_idx], dtype=np.float64)

    order = np.argsort(angles)
    theta = [angles[i] for i in order]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=v[order],
        theta=theta,
        mode="lines+markers",
        marker=dict(size=5, color=ACCENT),
        line=dict(color=ACCENT, width=1.6),
        name="measured (with inclusions)",
        hovertemplate="θ=%{theta:.0f}°<br>V = %{r:.4f}<extra></extra>",
    ))
    if reference is not None and reference.voltage_matrix.shape == \
            measurement.voltage_matrix.shape:
        v_ref = np.asarray(reference.voltage_matrix[inj_idx], dtype=np.float64)
        fig.add_trace(go.Scatterpolar(
            r=v_ref[order],
            theta=theta,
            mode="lines",
            line=dict(color=WARN, width=1.4, dash="dash"),
            name="empty-tank reference",
            hovertemplate="θ=%{theta:.0f}°<br>V_ref = %{r:.4f}<extra></extra>",
        ))

    fig.update_layout(**_polar_layout(
        f"Injection {inj_idx + 1} · electrode voltages"))
    return fig


def voltage_diff_figure(
    measurement: KTCMeasurement,
    reference: KTCMeasurement | None,
    inj_idx: int,
) -> go.Figure:
    """Bar chart of ΔV = Uel − Uelref per measurement pair for one injection.

    This difference is the main input to ABC1 and the signal CUQI8 fits;
    large |ΔV| marks pairs whose current path passes near an inclusion.
    """
    if reference is None or reference.voltage_matrix.shape != \
            measurement.voltage_matrix.shape:
        return empty_figure("Empty-tank reference (ref.mat) unavailable")

    v = np.asarray(measurement.voltage_matrix[inj_idx], dtype=np.float64)
    v_ref = np.asarray(reference.voltage_matrix[inj_idx], dtype=np.float64)
    delta = v - v_ref
    x = np.arange(1, len(delta) + 1)
    colors = [DANGER if d < 0 else SUCCESS for d in delta]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=delta,
        marker=dict(color=colors, line=dict(width=0)),
        name="ΔV = V − V_ref",
        hovertemplate="pair %{x}<br>ΔV = %{y:.5f}<extra></extra>",
    ))
    fig.update_layout(**_cartesian_layout(
        f"Injection {inj_idx + 1} · inclusion-induced voltage change",
        "measurement pair", "ΔV"))
    return fig


def resistance_scatter_figure(
    measurement: KTCMeasurement, inj_idx: int
) -> go.Figure:
    """Scatter of R = V/I per measurement pair for the selected injection."""
    r = np.asarray(measurement.resistance_matrix[inj_idx], dtype=np.float64)
    x = np.arange(1, len(r) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=r,
        mode="lines+markers",
        marker=dict(size=6, color=r, colorscale="Viridis", showscale=False),
        line=dict(color="rgba(123,97,255,0.45)", width=1.2),
        name="R = V / I",
        hovertemplate="pair %{x}<br>R = %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**_cartesian_layout(
        f"Injection {inj_idx + 1} · resistance per pair",
        "measurement pair", "R = V / I"))
    return fig


def resistance_heatmap_figure(
    measurement: KTCMeasurement, inj_idx: int | None = None
) -> go.Figure:
    """2D resistance map: rows = injections, columns = measurement pairs.

    Bright rows/columns are the injection–measurement combinations carrying
    the most information about the inclusions. The currently selected
    injection row is outlined.
    """
    z = np.asarray(measurement.resistance_matrix, dtype=np.float64)
    n_inj, n_pairs = z.shape

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(range(1, n_pairs + 1)),
        y=list(range(1, n_inj + 1)),
        colorscale="Viridis",
        colorbar=dict(thickness=10, len=0.85,
                      tickfont=dict(color="#ccc", size=10),
                      title=dict(text="R", font=dict(color=MUTED, size=11))),
        hovertemplate=("injection %{y}<br>pair %{x}<br>"
                       "R = %{z:.4f}<extra></extra>"),
    ))
    layout = _cartesian_layout(
        "Resistance map · all injections × measurement pairs",
        "measurement pair", "injection")
    layout["showlegend"] = False
    fig.update_layout(**layout)
    if inj_idx is not None and 0 <= inj_idx < n_inj:
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=0.5, x1=n_pairs + 0.5,
            y0=inj_idx + 0.5, y1=inj_idx + 1.5,
            line=dict(color=WARN, width=1.6),
        )
    return fig


# ── Layout ────────────────────────────────────────────────────


def layout() -> html.Div:
    """M6 layout: header, chips, injection controls, 3 measurement panels."""
    return html.Div(
        [
            _header(),
            html.Div(
                id=_CHIPS_ID,
                children=[_chip("status", "loading…")],
                style={"display": "flex", "flexWrap": "wrap", "gap": "8px"},
            ),
            html.Div(id=_BANNER_ID),
            _injection_controls(),
            dcc.Interval(id=_INTERVAL_ID, interval=900, n_intervals=0,
                         disabled=True),
            dcc.Store(id=_PLAYING_ID, data=False),
            _row_label("Current", "How charge is injected at each step"),
            html.Div(
                [
                    _panel("Current Pattern", "Polar bar · red = source, "
                           "blue = sink", _CURRENT_ID, badge="I",
                           badge_color=DANGER, height=400),
                    _panel("Electrode Voltages", "Measured vs. empty-tank "
                           "reference", _VOLTAGE_POLAR_ID, badge="V",
                           badge_color=ACCENT, height=400),
                ],
                style=_grid(420),
            ),
            _row_label("Voltage Difference",
                       "Uel − Uelref · the signal the algorithms actually use"),
            html.Div(
                [_panel("Inclusion Signal", "Per measurement pair for the "
                        "selected injection", _VOLTAGE_DIFF_ID, badge="ΔV",
                        badge_color=SUCCESS, height=320)],
                style=_grid(420),
            ),
            _row_label("Resistance", "R = V/I · scatter and full 2D map"),
            html.Div(
                [
                    _panel("Resistance Scatter", "Selected injection",
                           _RES_SCATTER_ID, badge="R", badge_color=WARN,
                           height=360),
                    _panel("Resistance Map", "All injections × pairs · "
                           "selected row outlined", _RES_HEATMAP_ID,
                           badge="R²ᴰ", badge_color="#a07bff", height=360),
                ],
                style=_grid(420),
            ),
        ],
        style={
            "padding": "24px 28px",
            "minHeight": "100%",
            "display": "flex",
            "flexDirection": "column",
            "gap": "14px",
        },
    )


def _grid(min_px: int) -> dict:
    return {
        "display": "grid",
        "gridTemplateColumns": f"repeat(auto-fit, minmax({min_px}px, 1fr))",
        "gap": "16px",
    }


def _header() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "M6",
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
                        "Measurement Domain Viewer",
                        style={"color": TEXT, "fontSize": "20px",
                               "fontWeight": 600},
                    ),
                ]
            ),
            html.P(
                "Explore the raw KTC2023 electrical measurements — injected "
                "currents, electrode voltages, and derived resistance — "
                "independent of any reconstruction. Step through injection "
                "patterns to see how measurement coverage shrinks at higher "
                "difficulty levels.",
                style={"color": MUTED, "margin": "8px 0 0", "fontSize": "13px",
                       "lineHeight": "1.5"},
            ),
        ],
        style={**CARD_STYLE, "padding": "16px 20px"},
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


def _injection_controls() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Injection Step", style={
                                "color": TEXT, "fontSize": "12px",
                                "fontWeight": 600, "marginRight": "10px",
                            }),
                            html.Span(
                                "Drag to inspect one injection pattern, or "
                                "press ▶ Play to animate through all of them.",
                                style={"color": MUTED, "fontSize": "11px"},
                            ),
                        ],
                        style={"marginBottom": "8px"},
                    ),
                    dcc.Slider(
                        id=_INJ_SLIDER_ID,
                        min=0, max=75, step=1, value=0,
                        marks=_injection_marks(76),
                        tooltip={"placement": "top", "always_visible": False},
                        updatemode="drag",
                    ),
                ],
                style={"flex": "1", "marginRight": "24px"},
            ),
            html.Div(
                [
                    html.Button(
                        "▶  Play", id=_PLAY_BTN_ID, n_clicks=0,
                        style={
                            "backgroundColor": ACCENT, "color": "#fff",
                            "border": "none", "borderRadius": "8px",
                            "padding": "8px 18px", "fontSize": "13px",
                            "fontWeight": 600, "cursor": "pointer",
                        },
                    ),
                    html.Div("0.9 s / step", style={
                        "color": MUTED, "fontSize": "10px",
                        "textAlign": "center", "marginTop": "4px",
                    }),
                ],
                style={"paddingTop": "20px", "textAlign": "center"},
            ),
        ],
        style={
            **CARD_STYLE,
            "display": "flex",
            "alignItems": "flex-start",
            "padding": "16px 20px",
        },
    )


def _injection_marks(n_inj: int) -> dict:
    step = 10 if n_inj > 30 else 5
    marks = {
        i: {"label": str(i + 1),
            "style": {"color": MUTED, "fontSize": "10px"}}
        for i in range(0, n_inj, step)
    }
    marks[n_inj - 1] = {"label": str(n_inj),
                        "style": {"color": MUTED, "fontSize": "10px"}}
    return marks


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


# ── Callbacks ─────────────────────────────────────────────────


def register_callbacks(app) -> None:  # noqa: ANN001
    """Wire sidebar selectors + injection stepper to the five M6 panels."""

    @app.callback(
        Output(_INTERVAL_ID, "disabled"),
        Output(_PLAYING_ID, "data"),
        Output(_PLAY_BTN_ID, "children"),
        Input(_PLAY_BTN_ID, "n_clicks"),
        State(_PLAYING_ID, "data"),
        prevent_initial_call=True,
    )
    def toggle_play(n_clicks, is_playing):  # noqa: ANN001, ANN202
        playing = not is_playing
        label = "⏸  Pause" if playing else "▶  Play"
        return not playing, playing, label

    @app.callback(
        Output(_INJ_SLIDER_ID, "max"),
        Output(_INJ_SLIDER_ID, "marks"),
        Output(_INJ_SLIDER_ID, "value"),
        Input("sidebar-level-slider", "value"),
        Input("sidebar-sample-radio", "value"),
        Input(_INTERVAL_ID, "n_intervals"),
        State(_INJ_SLIDER_ID, "value"),
        State(_PLAYING_ID, "data"),
    )
    def sync_injection_slider(level, sample, n_intervals, current, playing):  # noqa: ANN001, ANN202
        level = int(level)
        try:
            n_inj = _load_measurement(level, sample).current_matrix.shape[0]
        except Exception:
            n_inj = 76

        trigger = (callback_context.triggered[0]["prop_id"]
                   if callback_context.triggered else "")
        current = int(current or 0)
        if _INTERVAL_ID in trigger:
            if not playing:
                raise PreventUpdate
            return n_inj - 1, _injection_marks(n_inj), (current + 1) % n_inj
        return n_inj - 1, _injection_marks(n_inj), min(current, n_inj - 1)

    @app.callback(
        Output(_CURRENT_ID, "figure"),
        Output(_VOLTAGE_POLAR_ID, "figure"),
        Output(_VOLTAGE_DIFF_ID, "figure"),
        Output(_RES_SCATTER_ID, "figure"),
        Output(_RES_HEATMAP_ID, "figure"),
        Output(_CHIPS_ID, "children"),
        Output(_BANNER_ID, "children"),
        Input(_INJ_SLIDER_ID, "value"),
        Input("sidebar-level-slider", "value"),
        Input("sidebar-sample-radio", "value"),
    )
    def update_panels(inj_idx, level, sample):  # noqa: ANN001, ANN202
        level = int(level)
        try:
            measurement = _load_measurement(level, sample)
        except FileNotFoundError as exc:
            logger.warning("M6 data missing: %s", exc)
            empty = empty_figure("Data missing")
            chips = [_chip("status", "data missing", accent=DANGER)]
            return (empty, empty, empty, empty, empty, chips,
                    _banner(str(exc), "error"))
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            logger.exception("M6 update failed")
            empty = empty_figure("Error")
            chips = [_chip("status", "error", accent=DANGER)]
            return (empty, empty, empty, empty, empty, chips,
                    _banner(f"Error: {exc}", "error"))

        reference = _load_reference(level)
        n_inj, n_el = measurement.current_matrix.shape
        inj_idx = min(int(inj_idx or 0), n_inj - 1)

        current_fig = current_polar_figure(measurement, level, inj_idx)
        vpolar_fig = voltage_polar_figure(measurement, reference, level, inj_idx)
        vdiff_fig = voltage_diff_figure(measurement, reference, inj_idx)
        rscatter_fig = resistance_scatter_figure(measurement, inj_idx)
        rheat_fig = resistance_heatmap_figure(measurement, inj_idx)

        n_meas = measurement.voltage_matrix.size
        chips = [
            _chip("level", f"L{level}", accent=WARN),
            _chip("sample", sample.upper(), accent="#cfe0ff"),
            _chip("electrodes", f"{n_el} / {_N_TANK_ELECTRODES}"),
            _chip("injections", f"{n_inj}"),
            _chip("pairs / injection",
                  f"{measurement.voltage_matrix.shape[1]}"),
            _chip("voltage samples", f"{n_meas:,}"),
            _chip("reference",
                  "loaded" if reference is not None else "missing",
                  accent=SUCCESS if reference is not None else DANGER),
        ]
        banner = None
        if reference is None:
            banner = _banner(
                "Empty-tank reference (measurements/ref.mat) not found — "
                "voltage difference panel disabled.", "warn")
        return (current_fig, vpolar_fig, vdiff_fig, rscatter_fig, rheat_fig,
                chips, banner)
