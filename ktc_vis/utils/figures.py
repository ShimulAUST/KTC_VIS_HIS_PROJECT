"""Shared Plotly figure helpers used across dashboard modules."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# ── Color palette ──────────────────────────────────────────────
# Categorical colors for the 3 KTC classes.
CLASS_COLORS = {
    0: "#1a1a2e",  # water        — dark background
    1: "#5b8def",  # resistive    — blue
    2: "#e85d75",  # conductive   — red/pink
}
CLASS_LABELS = {0: "water", 1: "resistive", 2: "conductive"}

# Error overlay colors per the implementation guide:
#   green = correct, red = false positive, blue = false negative
ERROR_COLORS = {
    "correct": "#2ecc71",
    "false_positive": "#e74c3c",
    "false_negative": "#3498db",
    "background": "#1a1a2e",
}

_PAPER_BG = "#2a2a3f"
_PLOT_BG = "#1a1a2e"


def _base_layout(title: str, height: int | None = None) -> dict:
    """Default dark-mode layout shared by all image figures.

    height is intentionally None by default so that responsive=True on the
    dcc.Graph container drives the size — no conflict between figure height
    and CSS container height on first render.
    """
    layout = dict(
        title=dict(text=title, x=0.5, font=dict(color="#ddd", size=13)),
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        margin=dict(l=10, r=10, t=36, b=10),
        autosize=True,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, autorange="reversed"),
        uirevision="constant",
    )
    if height is not None:
        layout["height"] = height
    return layout


def segmentation_figure(seg: np.ndarray, title: str = "Segmentation") -> go.Figure:
    """3-class categorical heatmap (water / resistive / conductive)."""
    colorscale = [
        [0.0, CLASS_COLORS[0]],
        [0.33, CLASS_COLORS[0]],
        [0.34, CLASS_COLORS[1]],
        [0.66, CLASS_COLORS[1]],
        [0.67, CLASS_COLORS[2]],
        [1.0, CLASS_COLORS[2]],
    ]
    fig = go.Figure(
        go.Heatmap(
            z=seg,
            colorscale=colorscale,
            zmin=0,
            zmax=2,
            showscale=False,
            hovertemplate="x:%{x}<br>y:%{y}<br>class:%{z}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(title))
    return fig


def reconstruction_figure(recon: np.ndarray, title: str = "Reconstruction") -> go.Figure:
    """Continuous heatmap for a reconstruction map."""
    zmin = float(recon.min())
    zmax = float(recon.max()) if recon.max() > recon.min() else zmin + 1.0
    fig = go.Figure(
        go.Heatmap(
            z=recon,
            colorscale="Viridis",
            zmin=zmin,
            zmax=zmax,
            showscale=True,
            colorbar=dict(thickness=10, len=0.8, tickfont=dict(color="#ccc", size=10)),
            hovertemplate="x:%{x}<br>y:%{y}<br>val:%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(title))
    return fig


def error_overlay_figure(
    pred: np.ndarray, gt: np.ndarray, title: str = "Error Overlay"
) -> go.Figure:
    """Pixel-wise error map: green=correct, red=false-pos, blue=false-neg, yellow=wrong-class."""
    pred = pred.astype(np.int16)
    gt = gt.astype(np.int16)

    code = np.zeros(pred.shape, dtype=np.uint8)            # 0 = water-correct (bg)
    correct = pred == gt
    code[correct & (gt != 0)] = 1                          # 1 = correct inclusion
    code[(pred != 0) & (gt == 0)] = 2                      # 2 = false positive
    code[(pred == 0) & (gt != 0)] = 3                      # 3 = false negative
    code[(pred != 0) & (gt != 0) & (pred != gt)] = 4       # 4 = wrong class

    colorscale = [
        [0.0, ERROR_COLORS["background"]],
        [0.2, ERROR_COLORS["background"]],
        [0.21, ERROR_COLORS["correct"]],
        [0.4, ERROR_COLORS["correct"]],
        [0.41, ERROR_COLORS["false_positive"]],
        [0.6, ERROR_COLORS["false_positive"]],
        [0.61, ERROR_COLORS["false_negative"]],
        [0.8, ERROR_COLORS["false_negative"]],
        [0.81, "#f1c40f"],
        [1.0, "#f1c40f"],
    ]
    fig = go.Figure(
        go.Heatmap(
            z=code,
            colorscale=colorscale,
            zmin=0,
            zmax=4,
            showscale=False,
            hovertemplate=(
                "x:%{x}<br>y:%{y}<br>code:%{z}"
                "<extra>0=bg 1=correct 2=FP 3=FN 4=wrong-class</extra>"
            ),
        )
    )
    fig.update_layout(**_base_layout(title))
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False, align="left",
        bgcolor="rgba(0,0,0,0.55)", bordercolor="#444", borderwidth=1,
        font=dict(color="#eee", size=10),
        text=(
            "<span style='color:#2ecc71'>■</span> correct  "
            "<span style='color:#e74c3c'>■</span> FP  "
            "<span style='color:#3498db'>■</span> FN  "
            "<span style='color:#f1c40f'>■</span> wrong-class"
        ),
    )
    return fig


def empty_figure(message: str = "No data") -> go.Figure:
    """Placeholder figure used when data is unavailable."""
    fig = go.Figure()
    fig.update_layout(**_base_layout(" "))
    fig.add_annotation(
        text=message, showarrow=False,
        font=dict(color="#888", size=13), xref="paper", yref="paper", x=0.5, y=0.5,
    )
    return fig
