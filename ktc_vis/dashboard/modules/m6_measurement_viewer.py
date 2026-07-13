"""Module 6: Measurement Domain Viewer.

Explores the raw electrical measurements independently of any
reconstruction, for the selected ``(level, sample)``:

    Current       — polar bar of the injected current pattern for one
                    injection step.  Steppable / animatable.
    Voltage       — polar overlay (measured + empty-tank reference) +
                    ΔV difference bar chart.
    Resistance    — R per pair (bar), all-injections overlay, mean±std
                    summary, ΔR difference, SNR per pair.
    Anomaly       — per-injection anomaly score, measurement stability CV.
    Electrode     — contact quality estimate, cross-level coverage chart.
    Coverage map  — polar showing active vs. removed electrodes + pair arcs.
    Export        — CSV download of R / V / I for the selected injection.
"""

from __future__ import annotations

import io
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

_N_TANK_ELECTRODES = 32

# component IDs
_CURRENT_ID          = "m6-current-polar"
_VOLTAGE_POLAR_ID    = "m6-voltage-polar"
_VOLTAGE_DIFF_ID     = "m6-voltage-diff"
# resistance — per injection
_RES_BAR_ID          = "m6-resistance-bar"
_RES_DELTA_ID        = "m6-resistance-delta"
_SNR_ID              = "m6-snr-pairs"
# resistance — across injections
_RES_OVERLAY_ID      = "m6-resistance-overlay"
_RES_SUMMARY_ID      = "m6-resistance-summary"
# anomaly & stability
_ANOMALY_ID          = "m6-anomaly-score"
_STABILITY_ID        = "m6-measurement-stability"
# electrode & coverage
_IMPEDANCE_ID        = "m6-electrode-impedance"
_LEVEL_COVERAGE_ID   = "m6-level-coverage"
# coverage polar
_COVERAGE_POLAR_ID   = "m6-coverage-polar"
# controls / state
_INJ_SLIDER_ID       = "m6-injection-slider"
_PLAY_BTN_ID         = "m6-play-btn"
_INTERVAL_ID         = "m6-interval"
_PLAYING_ID          = "m6-playing"
_CHIPS_ID            = "m6-chips"
_BANNER_ID           = "m6-banner"
_DOWNLOAD_BTN_ID     = "m6-download-btn"
_DOWNLOAD_ID         = "m6-download"

_POLAR_BG = "#14142a"
_GRID     = "#2a2a4a"

_PANEL_HEADER_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "space-between",
    "padding": "10px 14px",
    "borderBottom": f"1px solid {BORDER}",
}


# loads and caches measurement data for a given level/sample pair
@lru_cache(maxsize=32)
def _load_measurement(level: int, sample: str) -> KTCMeasurement:
    return _LOADER.load(level=level, sample=sample)


_REF_CACHE: dict[int, KTCMeasurement] = {}


# loads the empty-tank reference from ref.mat, subsamples it to the requested level, and caches the result
def _load_reference(level: int) -> KTCMeasurement | None:
    if level in _REF_CACHE:
        return _REF_CACHE[level]
    if not _REF_MAT.exists():
        return None
    try:
        ref = scipy.io.loadmat(str(_REF_MAT))
        inj  = ref["Injref"]
        mpat = ref["Mpat"].astype(np.int16)
        uel  = np.asarray(ref["Uelref"], dtype=np.float64).flatten()
    except (KeyError, ValueError, OSError):
        logger.exception("Failed to read %s", _REF_MAT)
        return None

    n_inj = inj.shape[1]
    if n_inj == 0 or uel.size == 0 or uel.size % n_inj != 0:
        return None
    voltage  = uel.reshape(n_inj, len(uel) // n_inj)
    current  = inj.T
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
    _REF_CACHE[level] = measurement
    return measurement


# replaces any NaN or Inf values with zero and flags whether the array needed cleaning
def _sanitize(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.all():
        return arr, False
    return np.where(finite, arr, 0.0), True


# returns the list of active electrode indices and their corresponding angles around the tank
def _electrode_angles(level: int) -> tuple[list[int], list[float]]:
    active = subsample_electrodes(level)
    return active, [(360.0 * e / _N_TANK_ELECTRODES) % 360 for e in active]


# maps each measurement pair column to an angle on the tank circumference using the mpat matrix
def _pair_angles(measurement: KTCMeasurement, level: int) -> list[float]:
    n_cols = measurement.voltage_matrix.shape[1]
    mpat   = measurement.__dict__.get("mpat")
    if mpat is None or mpat.shape[1] != n_cols:
        return list(np.linspace(0.0, 360.0, n_cols, endpoint=False))
    active = subsample_electrodes(level)
    angles: list[float] = []
    for c in range(n_cols):
        rows = [r for r in np.nonzero(mpat[:, c])[0] if r < len(active)]
        if not rows:
            angles.append(360.0 * c / max(n_cols, 1))
            continue
        rad  = np.deg2rad([360.0 * active[r] / _N_TANK_ELECTRODES for r in rows])
        mean = np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))
        angles.append(float(mean % 360))
    return angles


# returns a Plotly layout dict for polar charts, with the dark background and optional legend
def _polar_layout(title: str, legend: bool = True) -> dict:
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


# returns a Plotly layout dict for bar and line charts with shared axis and legend styling
def _cartesian_layout(title: str, xtitle: str, ytitle: str) -> dict:
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


# builds the polar bar chart showing injected current for one step, with removed electrodes marked
def current_polar_figure(
    measurement: KTCMeasurement, level: int, inj_idx: int
) -> go.Figure:
    if measurement.current_matrix.size == 0:
        return empty_figure("No current data at this level")
    active, angles = _electrode_angles(level)
    currents, _ = _sanitize(measurement.current_matrix[inj_idx])
    colors = [DANGER if c > 0 else ACCENT if c < 0 else "rgba(90,90,120,0.25)"
              for c in currents]
    rmax = float(np.abs(currents).max()) or 1.0
    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=np.abs(currents), theta=angles,
        width=[360.0 / _N_TANK_ELECTRODES * 0.85] * len(angles),
        marker=dict(color=colors, line=dict(color="#1e1e2f", width=0.5)),
        customdata=np.stack([np.array(active) + 1, currents], axis=-1),
        hovertemplate="electrode %{customdata[0]}<br>I = %{customdata[1]:.3f} mA<extra></extra>",
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
        f"Injection {inj_idx + 1} of {n_inj}, current pattern"))
    fig.update_polars(radialaxis_range=[0, rmax * 1.25])
    return fig


# overlays measured electrode voltages and the empty-tank reference on a polar chart
def voltage_polar_figure(
    measurement: KTCMeasurement,
    reference: KTCMeasurement | None,
    level: int,
    inj_idx: int,
) -> go.Figure:
    if measurement.voltage_matrix.size == 0:
        return empty_figure("No voltage data at this level")
    angles = _pair_angles(measurement, level)
    v, _ = _sanitize(measurement.voltage_matrix[inj_idx])
    order = np.argsort(angles)
    theta = [angles[i] for i in order]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=v[order], theta=theta, mode="lines+markers",
        marker=dict(size=5, color=ACCENT), line=dict(color=ACCENT, width=1.6),
        name="measured (with inclusions)",
        hovertemplate="θ=%{theta:.0f}°<br>V = %{r:.4f}<extra></extra>",
    ))
    if reference is not None and \
            reference.voltage_matrix.shape == measurement.voltage_matrix.shape:
        v_ref, _ = _sanitize(reference.voltage_matrix[inj_idx])
        fig.add_trace(go.Scatterpolar(
            r=v_ref[order], theta=theta, mode="lines",
            line=dict(color=WARN, width=1.4, dash="dash"),
            name="empty-tank reference",
            hovertemplate="θ=%{theta:.0f}°<br>V_ref = %{r:.4f}<extra></extra>",
        ))
    fig.update_layout(**_polar_layout(f"Injection {inj_idx + 1}, electrode voltages"))
    return fig


# bar chart of ΔV = V − V_ref for the selected injection, the direct input to reconstruction
def voltage_diff_figure(
    measurement: KTCMeasurement,
    reference: KTCMeasurement | None,
    inj_idx: int,
) -> go.Figure:
    if reference is None or \
            reference.voltage_matrix.shape != measurement.voltage_matrix.shape:
        return empty_figure("Empty-tank reference (ref.mat) unavailable")
    if measurement.voltage_matrix.size == 0:
        return empty_figure("No voltage data at this level")
    v, _     = _sanitize(measurement.voltage_matrix[inj_idx])
    v_ref, _ = _sanitize(reference.voltage_matrix[inj_idx])
    delta    = v - v_ref
    x        = np.arange(1, len(delta) + 1)
    colors   = [DANGER if d < 0 else SUCCESS for d in delta]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=delta, marker=dict(color=colors, line=dict(width=0)),
        name="ΔV = V − V_ref",
        hovertemplate="pair %{x}<br>ΔV = %{y:.5f}<extra></extra>",
    ))
    fig.update_layout(**_cartesian_layout(
        f"Injection {inj_idx + 1}, voltage change from inclusion",
        "measurement pair", "ΔV"))
    return fig


def resistance_bar_figure(measurement: KTCMeasurement, inj_idx: int) -> go.Figure:
    """Bar chart of resistance per measurement pair for the selected injection. Green = positive, red = negative."""
    if measurement.resistance_matrix.size == 0:
        return empty_figure("No resistance data at this level")
    r, _   = _sanitize(measurement.resistance_matrix[inj_idx])
    x      = list(range(1, len(r) + 1))
    colors = [SUCCESS if v >= 0 else DANGER for v in r]
    fig = go.Figure(go.Bar(
        x=x, y=r, marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="pair %{x}<br>R = %{y:.4f}<extra></extra>",
        name="R = V / I",
    ))
    fig.add_hline(y=0, line=dict(color="#555", width=1, dash="dot"))
    fig.update_layout(**_cartesian_layout(
        f"Injection {inj_idx + 1}, R per pair",
        "measurement pair", "R = V / I"))
    return fig


def resistance_delta_figure(
    measurement: KTCMeasurement,
    reference: KTCMeasurement | None,
    inj_idx: int,
) -> go.Figure:
    # ΔR = R − R_ref per pair — direct pre-reconstruction anomaly signal.
    if reference is None or \
            reference.resistance_matrix.shape != measurement.resistance_matrix.shape:
        return empty_figure("Reference (ref.mat) unavailable")
    r, _     = _sanitize(measurement.resistance_matrix[inj_idx])
    r_ref, _ = _sanitize(reference.resistance_matrix[inj_idx])
    delta    = r - r_ref
    x        = list(range(1, len(delta) + 1))
    colors   = [SUCCESS if d >= 0 else DANGER for d in delta]
    fig = go.Figure(go.Bar(
        x=x, y=delta.tolist(),
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="pair %{x}<br>ΔR = %{y:.4f}<extra></extra>",
        name="ΔR = R − R_ref",
    ))
    fig.add_hline(y=0, line=dict(color="#555", width=1, dash="dot"))
    fig.update_layout(**_cartesian_layout(
        f"Injection {inj_idx + 1}, resistance difference ΔR",
        "measurement pair", "ΔR = R − R_ref"))
    return fig


def snr_per_pair_figure(
    measurement: KTCMeasurement,
    reference: KTCMeasurement | None,
    inj_idx: int,
) -> go.Figure:
    # Signal-to-noise |ΔV|/|V_ref| per pair — higher = more informative pair.
    if reference is None or \
            reference.voltage_matrix.shape != measurement.voltage_matrix.shape:
        return empty_figure("Reference (ref.mat) unavailable")
    v, _     = _sanitize(measurement.voltage_matrix[inj_idx])
    v_ref, _ = _sanitize(reference.voltage_matrix[inj_idx])
    snr      = np.abs(v - v_ref) / (np.abs(v_ref) + 1e-12)
    x        = list(range(1, len(snr) + 1))
    fig = go.Figure(go.Bar(
        x=x, y=snr.tolist(),
        marker=dict(
            color=snr.tolist(), colorscale="YlOrRd", showscale=True,
            colorbar=dict(
                thickness=8, len=0.7,
                title=dict(text="|ΔV|/|V_ref|", font=dict(color=MUTED, size=10)),
                tickfont=dict(color=MUTED, size=9),
            ),
            line=dict(width=0),
        ),
        hovertemplate="pair %{x}<br>SNR = %{y:.4f}<extra></extra>",
        name="|ΔV| / |V_ref|",
    ))
    fig.update_layout(**_cartesian_layout(
        f"Injection {inj_idx + 1}, signal-to-noise per pair",
        "measurement pair", "|ΔV| / |V_ref|"))
    return fig


def resistance_overlay_figure(measurement: KTCMeasurement, inj_idx: int) -> go.Figure:
    # All injections as faint lines, selected one highlighted.
    #
    # All background traces are merged into one None-separated Scatter so
    # Plotly handles a single object instead of up to 76 individual traces —
    # keeps the play animation smooth.
    if measurement.resistance_matrix.size == 0:
        return empty_figure("No resistance data at this level")
    z, _ = _sanitize(measurement.resistance_matrix)
    n_inj, n_pairs = z.shape
    x = list(range(1, n_pairs + 1))

    # Build one trace for all background injections using None separators.
    x_bg: list = []
    y_bg: list = []
    for i in range(n_inj):
        if i == inj_idx:
            continue
        x_bg.extend(x)
        x_bg.append(None)
        y_bg.extend(z[i].tolist())
        y_bg.append(None)

    fig = go.Figure()
    if x_bg:
        fig.add_trace(go.Scatter(
            x=x_bg, y=y_bg, mode="lines",
            line=dict(color="rgba(160,123,255,0.15)", width=1),
            name="other injections", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=x, y=z[inj_idx].tolist(), mode="lines+markers",
        line=dict(color=WARN, width=2.2),
        marker=dict(size=5, color=WARN),
        name=f"injection {inj_idx + 1}",
        hovertemplate="pair %{x}<br>R = %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**_cartesian_layout(
        f"All injections, #{inj_idx + 1} highlighted",
        "measurement pair", "R = V / I"))
    return fig


def resistance_summary_figure(measurement: KTCMeasurement) -> go.Figure:
    # Mean R per pair ±1 std — bright marker = high variability across injections.
    if measurement.resistance_matrix.size == 0:
        return empty_figure("No resistance data at this level")
    z, _  = _sanitize(measurement.resistance_matrix)
    mean  = z.mean(axis=0)
    std   = z.std(axis=0)
    x     = list(range(1, z.shape[1] + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=(mean + std).tolist() + (mean - std).tolist()[::-1],
        fill="toself", fillcolor="rgba(160,123,255,0.18)",
        line=dict(width=0), showlegend=True, name="± 1 std", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=mean.tolist(), mode="lines+markers",
        line=dict(color="#a07bff", width=2),
        marker=dict(size=4, color=std.tolist(), colorscale="Oranges",
                    showscale=True,
                    colorbar=dict(thickness=8, len=0.6,
                                  title=dict(text="std", font=dict(color=MUTED, size=10)),
                                  tickfont=dict(color=MUTED, size=9))),
        name="mean R",
        hovertemplate="pair %{x}<br>mean R = %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**_cartesian_layout(
        "Mean R per pair (marker color shows variability)",
        "measurement pair", "mean R = V / I"))
    return fig


def anomaly_score_figure(
    measurement: KTCMeasurement,
    reference: KTCMeasurement | None,
    inj_idx: int,
) -> go.Figure:
    # Per-injection anomaly score Σ|ΔR| — taller bar = more inclusion influence.
    if reference is None or \
            reference.resistance_matrix.shape != measurement.resistance_matrix.shape:
        return empty_figure("Reference (ref.mat) unavailable")
    r, _     = _sanitize(measurement.resistance_matrix)
    r_ref, _ = _sanitize(reference.resistance_matrix)
    scores   = np.abs(r - r_ref).sum(axis=1)
    x        = list(range(1, len(scores) + 1))
    colors   = [WARN if i == inj_idx else "#5b8def" for i in range(len(scores))]
    fig = go.Figure(go.Bar(
        x=x, y=scores.tolist(),
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="injection %{x}<br>score = %{y:.4f}<extra></extra>",
        name="Σ|ΔR|",
    ))
    fig.update_layout(**_cartesian_layout(
        f"Anomaly score per injection, #{inj_idx + 1} highlighted",
        "injection", "Σ|ΔR|"))
    return fig


def measurement_stability_figure(measurement: KTCMeasurement) -> go.Figure:
    # Coefficient of variation per pair — red bars signal unstable measurements.
    if measurement.resistance_matrix.size == 0:
        return empty_figure("No resistance data at this level")
    z, _   = _sanitize(measurement.resistance_matrix)
    mean   = np.abs(z.mean(axis=0))
    std    = z.std(axis=0)
    cv     = std / (mean + 1e-12)
    x      = list(range(1, z.shape[1] + 1))
    thresh = float(np.median(cv) * 2)
    colors = [DANGER if v > thresh else SUCCESS for v in cv]
    fig = go.Figure(go.Bar(
        x=x, y=cv.tolist(),
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="pair %{x}<br>CV = %{y:.3f}<extra></extra>",
        name="CV = std / |mean|",
    ))
    fig.add_hline(
        y=thresh, line=dict(color=WARN, width=1.2, dash="dash"),
        annotation_text="2× median threshold",
        annotation_font=dict(color=WARN, size=10),
    )
    fig.update_layout(**_cartesian_layout(
        "Measurement stability: coefficient of variation per pair",
        "measurement pair", "CV = std / |mean|"))
    return fig


def electrode_impedance_figure(measurement: KTCMeasurement, level: int) -> go.Figure:
    # Mean |R| per electrode across all pairs it participates in.
    #
    # Electrodes with unusually high mean |R| may have poor contact or high
    # contact impedance — they appear as red bars above the 2× median line.
    if measurement.resistance_matrix.size == 0:
        return empty_figure("No resistance data at this level")
    z, _   = _sanitize(measurement.resistance_matrix)
    mpat   = measurement.__dict__.get("mpat")
    active = subsample_electrodes(level)

    mean_r: list[float] = []
    if mpat is not None and mpat.shape[1] == z.shape[1]:
        for e_idx in range(len(active)):
            pair_cols = (np.where(mpat[e_idx, :] != 0)[0]
                         if e_idx < mpat.shape[0] else np.array([], dtype=int))
            if len(pair_cols):
                mean_r.append(float(np.abs(z[:, pair_cols]).mean()))
            else:
                mean_r.append(0.0)
    else:
        mean_r = [float(np.abs(z).mean())] * len(active)

    labels = [str(active[i] + 1) for i in range(len(active))]
    thresh = float(np.median(mean_r) * 2)
    colors = [DANGER if v > thresh else SUCCESS for v in mean_r]

    fig = go.Figure(go.Bar(
        x=labels, y=mean_r,
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="electrode %{x}<br>mean |R| = %{y:.4f}<extra></extra>",
        name="mean |R|",
    ))
    fig.add_hline(
        y=thresh, line=dict(color=WARN, width=1.2, dash="dash"),
        annotation_text="2× median", annotation_font=dict(color=WARN, size=10),
    )
    fig.update_layout(**_cartesian_layout(
        "Electrode contact quality, mean |R| across all pairs",
        "electrode", "mean |R|"))
    return fig


def level_coverage_figure(current_level: int, sample: str) -> go.Figure:
    # Electrode count and measurement-pair count at each difficulty level 1–7.
    levels      = list(range(1, 8))
    n_electrodes = [len(subsample_electrodes(lv)) for lv in levels]
    n_pairs: list[int | None] = []
    for lv in levels:
        try:
            n_pairs.append(_load_measurement(lv, sample).voltage_matrix.shape[1])
        except Exception:
            n_pairs.append(None)

    bar_colors = [WARN if lv == current_level else "#5b8def" for lv in levels]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=levels, y=n_electrodes, name="active electrodes",
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertemplate="level %{x}<br>electrodes = %{y}<extra></extra>",
    ))
    valid = [(lv, p) for lv, p in zip(levels, n_pairs) if p is not None]
    if valid:
        lv_list, p_list = zip(*valid)
        fig.add_trace(go.Scatter(
            x=list(lv_list), y=list(p_list), mode="lines+markers",
            line=dict(color=SUCCESS, width=2),
            marker=dict(size=7, color=SUCCESS),
            name="meas. pairs", yaxis="y2",
            hovertemplate="level %{x}<br>pairs = %{y}<extra></extra>",
        ))

    layout = _cartesian_layout(
        f"Coverage per level, currently at L{current_level}",
        "difficulty level", "active electrodes")
    axis = dict(gridcolor=_GRID, linecolor=_GRID, tickfont=dict(color=MUTED, size=10))
    layout["yaxis2"] = {
        **axis,
        "overlaying": "y", "side": "right", "showgrid": False,
        "title": dict(text="meas. pairs", font=dict(color=MUTED, size=11)),
    }
    layout["legend"]["y"] = -0.22
    fig.update_layout(**layout)
    return fig


def coverage_polar_figure(
    measurement: KTCMeasurement, level: int, inj_idx: int
) -> go.Figure:
    """Polar map of the electrode ring showing active vs. removed positions, source/sink for the current injection, and a sample of pair arcs."""
    active     = subsample_electrodes(level)
    active_set = set(active)
    removed    = [e for e in range(_N_TANK_ELECTRODES) if e not in active_set]

    # Identify source (+) and sink (–) for the current injection.
    source_electrodes: list[int] = []
    sink_electrodes:   list[int] = []
    if measurement.current_matrix.size > 0:
        currents, _ = _sanitize(measurement.current_matrix[inj_idx])
        for idx, c in enumerate(currents):
            if idx < len(active):
                if c > 0.01:
                    source_electrodes.append(active[idx])
                elif c < -0.01:
                    sink_electrodes.append(active[idx])

    inj_set = set(source_electrodes + sink_electrodes)

    # Background active electrodes (exclude injection pair)
    bg_active = [e for e in active if e not in inj_set]
    if bg_active:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[1.0] * len(bg_active),
            theta=[(360.0 * e / _N_TANK_ELECTRODES) % 360 for e in bg_active],
            mode="markers",
            marker=dict(size=11, color=ACCENT, symbol="circle",
                        line=dict(color="#fff", width=1)),
            name="active electrode",
            text=[f"E{e + 1}" for e in bg_active],
            hovertemplate="%{text}<extra></extra>",
        ))
    else:
        fig = go.Figure()

    # Removed electrodes
    if removed:
        fig.add_trace(go.Scatterpolar(
            r=[1.0] * len(removed),
            theta=[(360.0 * e / _N_TANK_ELECTRODES) % 360 for e in removed],
            mode="markers",
            marker=dict(size=10, symbol="x-thin",
                        line=dict(color="#9aa0b4", width=1.5),
                        color="rgba(154,160,180,0.5)"),
            name=f"removed ({len(removed)})",
            text=[f"E{e + 1}" for e in removed],
            hovertemplate="%{text} (removed)<extra></extra>",
        ))

    # Source electrode(s) — bright red, larger
    if source_electrodes:
        fig.add_trace(go.Scatterpolar(
            r=[1.0] * len(source_electrodes),
            theta=[(360.0 * e / _N_TANK_ELECTRODES) % 360
                   for e in source_electrodes],
            mode="markers",
            marker=dict(size=16, color=DANGER, symbol="circle",
                        line=dict(color="#fff", width=2)),
            name="source (+)",
            text=[f"E{e + 1} source" for e in source_electrodes],
            hovertemplate="%{text}<extra></extra>",
        ))

    # Sink electrode(s) — bright blue, larger
    if sink_electrodes:
        fig.add_trace(go.Scatterpolar(
            r=[1.0] * len(sink_electrodes),
            theta=[(360.0 * e / _N_TANK_ELECTRODES) % 360
                   for e in sink_electrodes],
            mode="markers",
            marker=dict(size=16, color="#5b8def", symbol="circle",
                        line=dict(color="#fff", width=2)),
            name="sink (−)",
            text=[f"E{e + 1} sink" for e in sink_electrodes],
            hovertemplate="%{text}<extra></extra>",
        ))

    # Measurement pair arcs — single trace with None separators
    mpat = measurement.__dict__.get("mpat")
    if mpat is not None:
        n_pairs = mpat.shape[1]
        step    = max(1, n_pairs // 12)
        r_arcs: list  = []
        th_arcs: list = []
        for c in range(0, n_pairs, step):
            rows = [r for r in np.where(mpat[:, c] != 0)[0] if r < len(active)]
            if len(rows) >= 2:
                a1 = (360.0 * active[rows[0]] / _N_TANK_ELECTRODES) % 360
                a2 = (360.0 * active[rows[1]] / _N_TANK_ELECTRODES) % 360
                r_arcs  += [0.90, 0.90, None]
                th_arcs += [a1,   a2,   None]
        if r_arcs:
            fig.add_trace(go.Scatterpolar(
                r=r_arcs, theta=th_arcs, mode="lines",
                line=dict(color="rgba(160,123,255,0.25)", width=1.3),
                showlegend=False, hoverinfo="skip",
            ))

    n_inj = measurement.current_matrix.shape[0]
    fig.update_layout(**_polar_layout(
        f"Injection {inj_idx + 1} of {n_inj}: red = source, blue = sink"))
    fig.update_polars(radialaxis_range=[0, 1.35], radialaxis_visible=False)
    return fig


# builds the full page layout: header, chips, injection controls, and all the panel rows
def layout() -> html.Div:
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
            dcc.Interval(id=_INTERVAL_ID, interval=900, n_intervals=0, disabled=True),
            dcc.Store(id=_PLAYING_ID, data=False),
            dcc.Download(id=_DOWNLOAD_ID),

            _row_label("Current", "How charge is injected at each step"),
            html.Div(
                [
                    _panel("Current Pattern",
                           "Red bars = source electrode, blue = sink",
                           _CURRENT_ID, badge="I", badge_color=DANGER, height=380,
                           interpretation=(
                               "This shows which electrodes are injecting current and how much. "
                               "The tall red bars are where current enters (source) and the blue bars are where it exits (sink); bar height is the current magnitude in mA. "
                               "Grey X marks are electrodes that aren't available at this difficulty level. "
                               "Try stepping through all the injections. At higher levels you'll notice gaps where electrodes have been removed."
                           )),
                    _panel("Electrode Voltages",
                           "Measured vs. empty-tank reference",
                           _VOLTAGE_POLAR_ID, badge="V", badge_color=ACCENT, height=380,
                           interpretation=(
                               "The solid line is what we actually measured; the dashed line is what the empty tank looks like with nothing inside. "
                               "Where the two lines pull apart, something inside the tank is redirecting current. That's where an inclusion is. "
                               "If you see matching divergence on both sides, the inclusion probably sits along that diameter. "
                               "If the lines track each other perfectly for this step, try a different injection direction."
                           )),
                ],
                style=_grid(420),
            ),

            _row_label("Voltage Difference",
                       "Uel minus Uelref, the signal reconstruction algorithms actually work with"),
            html.Div(
                [_panel("Inclusion Signal",
                        "Per measurement pair for the selected injection",
                        _VOLTAGE_DIFF_ID, badge="ΔV", badge_color=SUCCESS, height=280,
                        interpretation=(
                            "This is literally the signal reconstruction algorithms work with. It's the difference between what we measured and what we'd expect from an empty tank. "
                            "Green bars mean the inclusion pushed voltage up for that pair; red means it pulled it down. "
                            "A bunch of tall bars clustered together tells you roughly where the inclusion is angularly. "
                            "A single tall bar with nothing around it is usually noise, not a real inclusion signal."
                        ))],
                style=_grid(420),
            ),

            _row_label("Resistance for This Injection",
                       "Absolute R, difference from reference, and SNR per pair"),
            html.Div(
                [
                    _panel("R per Pair",
                           "Green = positive resistance, red = sign reversal",
                           _RES_BAR_ID, badge="R", badge_color=WARN, height=260,
                           interpretation=(
                               "Resistance per pair for this injection. Green is the normal case; red means the sign flipped, "
                               "which can happen with conductive inclusions or a bad electrode contact. "
                               "Bars that stick out above or below their neighbours are the ones whose current paths got disrupted by the inclusion. "
                               "If everything looks flat and uniform, this injection direction probably isn't sensitive to whatever's inside."
                           )),
                    _panel("ΔR = R − R_ref",
                           "Pre-reconstruction anomaly signal",
                           _RES_DELTA_ID, badge="ΔR", badge_color=DANGER, height=260,
                           interpretation=(
                               "Subtracting the reference takes out the baseline and leaves just the inclusion's fingerprint. "
                               "Green bars show pairs where resistance went up (resistive anomaly); red bars where it went down (conductive). "
                               "Pairs hovering near zero weren't affected by the inclusion from this angle. "
                               "A tight cluster of large bars suggests a small localised object; if the whole chart lights up, the anomaly is large or diffuse."
                           )),
                    _panel("Signal-to-Noise per Pair",
                           "Brighter bars carry more information about the inclusion",
                           _SNR_ID, badge="SNR", badge_color="#5b8def", height=260,
                           interpretation=(
                               "A simple ratio of signal to background showing how loud the inclusion's effect is relative to the empty-tank voltage. "
                               "Tall bright bars are the most useful pairs for reconstruction; short pale ones are barely above the noise floor. "
                               "If everything is near zero, the inclusion happens to be in a blind spot for this injection and other steps will do better. "
                               "The tall bars are the ones you'd want to trust most if you were weighting the inversion manually."
                           )),
                ],
                style=_grid(340),
            ),

            _row_label("Resistance Across All Injections",
                       "How R patterns vary across the full injection protocol"),
            html.Div(
                [
                    _panel("All Injections Overlay",
                           "Selected injection highlighted in yellow",
                           _RES_OVERLAY_ID, badge="2", badge_color="#a07bff", height=260,
                           interpretation=(
                               "All injection profiles stacked together, with the one you've selected picked out in yellow. "
                               "A wide spread in the grey lines means the injection protocol is exploring diverse paths through the tank, which is what you want for a good reconstruction. "
                               "When the yellow line drifts away from the pack at certain pairs, those are the ones most sensitive to the inclusion from that angle. "
                               "A single grey trace that goes rogue while all the others agree is worth checking. It could be a bad electrode or a dropped injection."
                           )),
                    _panel("Mean ± Std Summary",
                           "Aggregated over all injections, updates when level or sample changes",
                           _RES_SUMMARY_ID, badge="3", badge_color=SUCCESS, height=260,
                           interpretation=(
                               "The purple line is the average resistance per pair over all injections; the shaded band shows how much it varies. "
                               "Pairs with orange markers have high variability. They respond differently depending on which electrodes inject, making them valuable for figuring out what's inside. "
                               "Pairs with a very tight band are consistent but tell you less about spatial structure. "
                               "At higher difficulty levels the whole band tends to narrow as there are fewer injection directions available."
                           )),
                ],
                style=_grid(420),
            ),

            _row_label("Anomaly & Stability",
                       "Which injections carry the most inclusion information "
                       "and which measurement pairs are unstable"),
            html.Div(
                [
                    _panel("Anomaly Score per Injection",
                           "Taller bar means stronger inclusion influence for that direction",
                           _ANOMALY_ID, badge="A", badge_color=DANGER, height=260,
                           interpretation=(
                               "Each bar shows how much the inclusion disturbed measurements for that injection direction, summing up the absolute resistance changes over all pairs. "
                               "Taller bar means that injection angle was looking right at the inclusion. The yellow bar is the step you currently have selected. "
                               "If only a couple of bars are very tall, the inclusion is probably sitting close to those source/sink axes. "
                               "One bar massively taller than everything else and not matching adjacent injections could be a hardware glitch rather than a real anomaly."
                           )),
                    _panel("Measurement Stability",
                           "Coefficient of variation per pair, updates when level or sample changes",
                           _STABILITY_ID, badge="CV", badge_color="#a07bff", height=260,
                           interpretation=(
                               "Coefficient of variation measures how much each pair's resistance jumps around as we step through different injections. "
                               "Green bars are well-behaved; red bars are all over the place. "
                               "A handful of red bars in the region where you'd expect the inclusion is perfectly normal. Those pairs are reacting to it. "
                               "But a long run of red bars where there shouldn't be anything is worth checking; an electrode may have a bad connection."
                           )),
                ],
                style=_grid(420),
            ),

            _row_label("Electrode & Coverage Analysis",
                       "Contact quality per electrode and how coverage shrinks with level"),
            html.Div(
                [
                    _panel("Electrode Contact Quality",
                           "Per-electrode average resistance, updates when level or sample changes",
                           _IMPEDANCE_ID, badge="Z", badge_color=WARN, height=260,
                           interpretation=(
                               "For each electrode, this averages the absolute resistance across every pair it's involved in, giving a rough indicator of how well it's making contact. "
                               "Green is fine; red means its average is more than twice the median, which could be high contact impedance or a loose gel connection. "
                               "A lone red bar here and there is normal. If you see three or four reds in a row (like E5 through E7), the whole connector block for that section might need attention. "
                               "Bad contacts drag down the SNR for every pair those electrodes participate in."
                           )),
                    _panel("Coverage per Level",
                           "Electrode and pair count across all difficulty levels",
                           _LEVEL_COVERAGE_ID, badge="L", badge_color="#5b8def", height=260,
                           interpretation=(
                               "Blue bars show how many electrodes are active at each level; the green line tracks how many measurement pairs that gives us (right axis). "
                               "The highlighted bar is where you currently are. "
                               "Notice that pairs drop much faster than electrodes as you go from L1 to L7. That's because pairs scale roughly with the square of the electrode count. "
                               "By L7 you've lost nearly 80% of the measurement pairs from L1, which is the main reason reconstruction quality degrades so steeply."
                           )),
                ],
                style=_grid(420),
            ),

            _row_label("Coverage Map",
                       "Polar view of active vs. removed electrodes "
                       "and sampled measurement pair connections"),
            html.Div(
                [_panel("Electrode Coverage Polar",
                        "Updates with each injection step",
                        _COVERAGE_POLAR_ID, badge="M", badge_color=ACCENT, height=400,
                        interpretation=(
                            "A top-down view of the electrode ring around the tank for the current injection step. "
                            "Blue dots are active electrodes; grey X marks are ones that have been removed at this difficulty level. "
                            "The big red dot is where current enters and the big blue dot is where it exits. "
                            "The faint purple lines show a sample of the measurement pairs in use. "
                            "Hit Play to watch the injection pair sweep around the ring. At higher levels you'll see obvious gaps where removed electrodes leave the reconstruction blind."
                        ))],
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


# CSS grid style that fits panels responsively at a given minimum column width
def _grid(min_px: int) -> dict:
    return {
        "display": "grid",
        "gridTemplateColumns": f"repeat(auto-fit, minmax({min_px}px, 1fr))",
        "gap": "16px",
    }


# renders the M6 header card with module badge, title, description, and the algorithm-note
def _header() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span("M6", style={
                        "display": "inline-block", "padding": "2px 10px",
                        "borderRadius": "999px", "backgroundColor": ACCENT,
                        "color": "#fff", "fontSize": "11px", "fontWeight": 600,
                        "letterSpacing": "0.5px", "marginRight": "10px",
                    }),
                    html.Span("Measurement Domain Viewer", style={
                        "color": TEXT, "fontSize": "20px", "fontWeight": 600,
                    }),
                ]
            ),
            html.P(
                "A direct look at the raw measurement data before any reconstruction happens. "
                "Currents, voltages, and resistance all laid out for the selected level and sample. "
                "Step through the injection patterns to see how the coverage changes as electrodes "
                "get removed at higher difficulty levels.",
                style={"color": MUTED, "margin": "8px 0 0", "fontSize": "13px",
                       "lineHeight": "1.5"},
            ),
            html.Div(
                "ℹ  Level and Sample control what data is shown. "
                "Algorithm has no effect here. Raw measurements are the same "
                "for all algorithms.",
                style={
                    "marginTop": "10px",
                    "padding": "7px 12px",
                    "backgroundColor": "#1f2a3f",
                    "border": "1px solid #3a4a6a",
                    "borderRadius": "8px",
                    "color": "#8ab4f8",
                    "fontSize": "11.5px",
                },
            ),
        ],
        style={**CARD_STYLE, "padding": "16px 20px"},
    )


# small badge-style chip showing a label and value pair in the status row
def _chip(label: str, value: str, accent: str | None = None) -> html.Div:
    return html.Div(
        [
            html.Span(label, style={
                "color": MUTED, "fontSize": "10.5px", "letterSpacing": "0.6px",
                "textTransform": "uppercase", "marginRight": "8px",
            }),
            html.Span(value, style={
                "color": accent or TEXT, "fontSize": "13px", "fontWeight": 600,
                "fontVariantNumeric": "tabular-nums",
            }),
        ],
        style={**CARD_STYLE, "padding": "8px 12px",
               "display": "inline-flex", "alignItems": "center"},
    )


# section heading row with an uppercase title and a dimmer subtitle, separated by a bottom border
def _row_label(title: str, subtitle: str) -> html.Div:
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
            "padding": "4px 2px 0", "paddingBottom": "6px",
            "borderBottom": f"1px solid {BORDER}", "marginTop": "4px",
        },
    )


# builds the injection slider, play button, and export CSV button in one card
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
                        id=_INJ_SLIDER_ID, min=0, max=75, step=1, value=0,
                        marks=_injection_marks(76),
                        tooltip={"placement": "top", "always_visible": False},
                        updatemode="drag",
                    ),
                ],
                style={"flex": "1", "marginRight": "24px"},
            ),
            html.Div(
                [
                    html.Button("▶  Play", id=_PLAY_BTN_ID, n_clicks=0, style={
                        "backgroundColor": ACCENT, "color": "#fff",
                        "border": "none", "borderRadius": "8px",
                        "padding": "8px 18px", "fontSize": "13px",
                        "fontWeight": 600, "cursor": "pointer",
                    }),
                    html.Div("0.9 s / step", style={
                        "color": MUTED, "fontSize": "10px",
                        "textAlign": "center", "marginTop": "4px",
                    }),
                ],
                style={"paddingTop": "20px", "textAlign": "center"},
            ),
            html.Div(
                [
                    html.Button("⬇  Export CSV", id=_DOWNLOAD_BTN_ID, n_clicks=0,
                                style={
                                    "backgroundColor": "#2a2a4a", "color": TEXT,
                                    "border": f"1px solid {BORDER}", "borderRadius": "8px",
                                    "padding": "8px 16px", "fontSize": "12px",
                                    "fontWeight": 500, "cursor": "pointer",
                                }),
                    html.Div("R, V, I for selected injection",
                             style={"color": MUTED, "fontSize": "10px",
                                    "textAlign": "center", "marginTop": "4px"}),
                ],
                style={"paddingTop": "20px", "textAlign": "center", "marginLeft": "12px"},
            ),
        ],
        style={**CARD_STYLE, "display": "flex", "alignItems": "flex-start",
               "padding": "16px 20px"},
    )


# generates tick mark labels for the injection slider, spacing them so they don't crowd together
def _injection_marks(n_inj: int) -> dict:
    step = 10 if n_inj > 30 else 5
    marks = {
        i: {"label": str(i + 1), "style": {"color": MUTED, "fontSize": "10px"}}
        for i in range(0, n_inj, step)
    }
    marks[n_inj - 1] = {"label": str(n_inj),
                        "style": {"color": MUTED, "fontSize": "10px"}}
    return marks


# card component wrapping a graph with a header, badge, subtitle, and optional interpretation text
def _panel(
    title: str,
    subtitle: str,
    graph_id: str,
    *,
    badge: str = "",
    badge_color: str = ACCENT,
    height: int = 340,
    interpretation: str = "",
) -> html.Div:
    children = [
        html.Div(
            [
                html.Div([
                    html.Span(badge, style={
                        "display": "inline-block", "padding": "2px 8px",
                        "borderRadius": "6px", "backgroundColor": badge_color,
                        "color": "#fff", "fontSize": "10px", "fontWeight": 700,
                        "letterSpacing": "0.4px", "marginRight": "10px",
                    }),
                    html.Span(title, style={"color": TEXT, "fontWeight": 600,
                                            "fontSize": "13.5px"}),
                ]),
                html.Span(subtitle, style={"color": MUTED, "fontSize": "11.5px"}),
            ],
            style=_PANEL_HEADER_STYLE,
        ),
        html.Div(
            dcc.Graph(
                id=graph_id, figure=empty_figure("Loading…"),
                config={"displayModeBar": False, "responsive": True},
                style={"height": f"{height}px", "width": "100%"},
            ),
            style={"padding": "6px 6px 0", "flex": "1",
                   "minHeight": "0", "overflow": "hidden"},
        ),
    ]
    if interpretation:
        children.append(html.Div(
            interpretation,
            style={
                "padding": "8px 14px 14px",
                "fontSize": "11px",
                "lineHeight": "1.65",
                "color": "#8a9ab8",
                "borderTop": f"1px solid {BORDER}",
                "whiteSpace": "pre-line",
            },
        ))
    return html.Div(
        children,
        style={**CARD_STYLE, "minHeight": f"{height + 62}px",
               "display": "flex", "flexDirection": "column", "overflow": "visible"},
    )


# coloured notification banner for info, warning, or error messages at the top of the page
def _banner(message: str, kind: str = "info") -> html.Div:
    palette = {
        "info":  ("#1f3a5f", "#5b8def", "#cfe0ff"),
        "warn":  ("#3a2a1a", WARN, "#f4c870"),
        "error": ("#3a1f25", DANGER, "#ffd1d8"),
    }
    bg, border, fg = palette.get(kind, palette["info"])
    return html.Div(message, style={
        "backgroundColor": bg, "border": f"1px solid {border}",
        "color": fg, "padding": "10px 14px",
        "borderRadius": "10px", "fontSize": "12.5px",
    })


def register_callbacks(app) -> None:  # noqa: ANN001
    # Wire sidebar selectors + injection stepper to all M6 panels.

    # toggles between play and pause, enabling or disabling the interval timer
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
        return not playing, playing, ("⏸  Pause" if playing else "▶  Play")

    # updates slider range when level/sample changes, and steps the index forward during animation
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

    # redraws the injection-dependent panels on every slider tick to keep animation smooth
    # algorithm dropdown is included so the callback re-runs on algorithm change
    # (measurements are algorithm-independent but we want the UI to react visually)
    @app.callback(
        Output(_CURRENT_ID,        "figure"),
        Output(_VOLTAGE_POLAR_ID,  "figure"),
        Output(_VOLTAGE_DIFF_ID,   "figure"),
        Output(_RES_BAR_ID,        "figure"),
        Output(_RES_DELTA_ID,      "figure"),
        Output(_SNR_ID,            "figure"),
        Output(_RES_OVERLAY_ID,    "figure"),
        Output(_ANOMALY_ID,        "figure"),
        Output(_COVERAGE_POLAR_ID, "figure"),
        Output(_CHIPS_ID,          "children"),
        Output(_BANNER_ID,         "children"),
        Input(_INJ_SLIDER_ID,                 "value"),
        Input("sidebar-level-slider",         "value"),
        Input("sidebar-sample-radio",         "value"),
        Input("sidebar-algorithm-dropdown",   "value"),
    )
    def update_panels_fast(inj_idx, level, sample, algorithm):  # noqa: ANN001, ANN202
        _N_FAST = 9  # 8 data figures + coverage polar

        def _all_empty(msg, kind="error"):
            e = empty_figure(msg)
            return (*([e] * _N_FAST),
                    [_chip("status", msg[:20], accent=DANGER)],
                    _banner(msg, kind))

        try:
            level       = int(level)
            measurement = _load_measurement(level, sample)
        except FileNotFoundError as exc:
            logger.warning("M6 data missing: %s", exc)
            return _all_empty(str(exc))
        except (ValueError, TypeError) as exc:
            logger.warning("M6 invalid selection: %s", exc)
            return _all_empty(f"Invalid selection: {exc}")
        except Exception as exc:  # pragma: no cover
            logger.exception("M6 fast update failed")
            return _all_empty(f"Error: {exc}")

        reference    = _load_reference(level)
        n_inj, n_el = measurement.current_matrix.shape
        try:
            inj_idx = int(inj_idx)
        except (TypeError, ValueError):
            inj_idx = 0
        inj_idx = max(0, min(inj_idx, max(n_inj - 1, 0)))

        def _safe(builder, *args):  # noqa: ANN001, ANN202
            try:
                return builder(*args)
            except Exception:  # pragma: no cover
                logger.exception("M6 %s failed", builder.__name__)
                return empty_figure("Panel error, see logs")

        figs = (
            _safe(current_polar_figure,      measurement, level, inj_idx),
            _safe(voltage_polar_figure,      measurement, reference, level, inj_idx),
            _safe(voltage_diff_figure,       measurement, reference, inj_idx),
            _safe(resistance_bar_figure,     measurement, inj_idx),
            _safe(resistance_delta_figure,   measurement, reference, inj_idx),
            _safe(snr_per_pair_figure,       measurement, reference, inj_idx),
            _safe(resistance_overlay_figure, measurement, inj_idx),
            _safe(anomaly_score_figure,      measurement, reference, inj_idx),
            _safe(coverage_polar_figure,     measurement, level, inj_idx),
        )

        n_meas = measurement.voltage_matrix.size
        chips = [
            _chip("level",             f"L{level}",          accent=WARN),
            _chip("sample",            sample.upper(),        accent="#cfe0ff"),
            _chip("algorithm",         (algorithm or "—").upper(),
                  accent=MUTED),
            _chip("electrodes",        f"{n_el} / {_N_TANK_ELECTRODES}"),
            _chip("injections",        f"{n_inj}"),
            _chip("pairs / injection", f"{measurement.voltage_matrix.shape[1]}"),
            _chip("voltage samples",   f"{n_meas:,}"),
            _chip("reference",
                  "loaded" if reference is not None else "missing",
                  accent=SUCCESS if reference is not None else DANGER),
        ]
        banner = None
        if reference is None:
            banner = _banner(
                "Empty-tank reference (measurements/ref.mat) not found. "
                "ΔV, ΔR, SNR, and anomaly panels are disabled.", "warn")
        elif not (np.isfinite(measurement.voltage_matrix).all()
                  and np.isfinite(measurement.current_matrix).all()
                  and np.isfinite(measurement.resistance_matrix).all()):
            banner = _banner(
                "Measurement contains non-finite values (NaN/Inf). "
                "Shown as 0 in all panels. Check the staged .mat files.", "warn")
        return (*figs, chips, banner)

    # redraws the aggregate panels that only need updating when level or sample changes
    @app.callback(
        Output(_RES_SUMMARY_ID,    "figure"),
        Output(_STABILITY_ID,      "figure"),
        Output(_IMPEDANCE_ID,      "figure"),
        Output(_LEVEL_COVERAGE_ID, "figure"),
        Input("sidebar-level-slider",       "value"),
        Input("sidebar-sample-radio",       "value"),
        Input("sidebar-algorithm-dropdown", "value"),
    )
    def update_panels_slow(level, sample, _algorithm):  # noqa: ANN001, ANN202
        def _all_empty_slow(msg):
            e = empty_figure(msg)
            return e, e, e, e

        try:
            level       = int(level)
            measurement = _load_measurement(level, sample)
        except Exception as exc:
            logger.warning("M6 slow update failed: %s", exc)
            return _all_empty_slow("Data unavailable")

        def _safe(builder, *args):  # noqa: ANN001, ANN202
            try:
                return builder(*args)
            except Exception:  # pragma: no cover
                logger.exception("M6 %s failed", builder.__name__)
                return empty_figure("Panel error, see logs")

        return (
            _safe(resistance_summary_figure,    measurement),
            _safe(measurement_stability_figure, measurement),
            _safe(electrode_impedance_figure,   measurement, level),
            _safe(level_coverage_figure,        level, sample),
        )

    # exports R, V, and I for the selected injection as a CSV file
    @app.callback(
        Output(_DOWNLOAD_ID, "data"),
        Input(_DOWNLOAD_BTN_ID, "n_clicks"),
        State(_INJ_SLIDER_ID,         "value"),
        State("sidebar-level-slider", "value"),
        State("sidebar-sample-radio", "value"),
        prevent_initial_call=True,
    )
    def download_csv(n_clicks, inj_idx, level, sample):  # noqa: ANN001, ANN202
        level   = int(level)
        inj_idx = int(inj_idx or 0)
        try:
            measurement = _load_measurement(level, sample)
        except Exception:
            raise PreventUpdate  # noqa: B904

        n_inj = measurement.resistance_matrix.shape[0]
        inj_idx = max(0, min(inj_idx, n_inj - 1))

        r, _ = _sanitize(measurement.resistance_matrix[inj_idx])
        v, _ = _sanitize(measurement.voltage_matrix[inj_idx])
        i, _ = _sanitize(measurement.current_matrix[inj_idx])

        n_pairs = len(r)
        n_el    = len(i)

        buf = io.StringIO()
        buf.write(f"# KTC2023 measurement export  level={level}  sample={sample}"
                  f"  injection={inj_idx + 1}\n")
        buf.write("pair,R_V_per_I,V_measured\n")
        for p in range(n_pairs):
            buf.write(f"{p + 1},{r[p]:.8g},{v[p]:.8g}\n")
        buf.write("\nelectrode,I_injected\n")
        for e in range(n_el):
            buf.write(f"{e + 1},{i[e]:.8g}\n")

        return dict(
            content=buf.getvalue(),
            filename=f"ktc_L{level}_{sample}_inj{inj_idx + 1}.csv",
            type="text/csv",
        )
