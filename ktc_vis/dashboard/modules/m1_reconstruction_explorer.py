"""Module 1: Reconstruction Explorer. Owner: Asmita Bhuva."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.io
from dash import Input, Output, dcc, html

RAW_DIR = Path("data/raw/ktc2023")
SAMPLE_MAP = {"a": 1, "b": 2, "c": 3}

_DARK = "#12121e"
_CARD = "#2a2a3f"
_MUTED = "#888888"
_TEXT = "#eeeeee"

_CS_2 = [[0.0, "#1565c0"], [0.5, "#1565c0"], [0.5, "#c62828"], [1.0, "#c62828"]]
_CS_3 = [
    [0.00, "#1565c0"], [0.33, "#1565c0"],
    [0.33, "#c62828"], [0.66, "#c62828"],
    [0.66, "#2e7d32"], [1.00, "#2e7d32"],
]
_CS_ERROR = [
    [0.00, "#2e7d32"], [0.33, "#2e7d32"],
    [0.33, "#c62828"], [0.66, "#c62828"],
    [0.66, "#1565c0"], [1.00, "#1565c0"],
]


def layout() -> html.Div:
    """4-panel layout: Ground Truth | Reconstruction | Segmentation | Error Overlay."""
    return html.Div([
        html.H3("Reconstruction Explorer", style={"color": _TEXT, "marginBottom": "4px"}),
        html.P(
            "Side-by-side view of ground truth, reconstruction, segmentation, and error overlay "
            "for the selected algorithm / level / sample.",
            style={"color": _MUTED, "marginBottom": "20px", "fontSize": "13px"},
        ),
        html.Div([
            _panel("Ground Truth", "m1-gt-graph",
                   "0=water · 1=resistive · 2=conductive"),
            _panel("Reconstruction", "m1-recon-graph",
                   "From benchmark cache — run run_benchmark.py if empty"),
            _panel("Segmentation", "m1-seg-graph",
                   "Predicted segmentation classes"),
            _panel("Error Overlay", "m1-error-graph",
                   "Green=correct · Red=false positive · Blue=false negative"),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

        # Metrics bar
        html.Div(id="m1-metrics-bar", style={"marginTop": "16px"}),

    ], style={"padding": "20px", "backgroundColor": _DARK, "minHeight": "100vh"})


def register_callbacks(app) -> None:  # noqa: ANN001
    @app.callback(
        Output("m1-gt-graph", "figure"),
        Output("m1-recon-graph", "figure"),
        Output("m1-seg-graph", "figure"),
        Output("m1-error-graph", "figure"),
        Output("m1-metrics-bar", "children"),
        Input("sidebar-algorithm-dropdown", "value"),
        Input("sidebar-level-slider", "value"),
        Input("sidebar-sample-radio", "value"),
    )
    def update_panels(algorithm, level, sample):
        gt_fig = _gt_figure(level, sample)
        recon_fig, seg_fig, error_fig, metrics = _cached_figures(algorithm, level, sample)
        metrics_bar = _build_metrics_bar(metrics)
        return gt_fig, recon_fig, seg_fig, error_fig, metrics_bar


# ── Figure helpers ────────────────────────────────────────────────────────────

def _panel(title: str, graph_id: str, subtitle: str) -> html.Div:
    return html.Div([
        html.Div(title, style={"color": _TEXT, "fontWeight": "bold",
                               "fontSize": "13px", "marginBottom": "2px"}),
        html.Div(subtitle, style={"color": _MUTED, "fontSize": "10px",
                                  "marginBottom": "6px"}),
        dcc.Graph(id=graph_id, config={"displayModeBar": False},
                  style={"height": "280px"}),
    ], style={"flex": "1", "minWidth": "220px", "backgroundColor": _CARD,
              "borderRadius": "8px", "padding": "12px"})


def _heatmap_fig(data: np.ndarray, colorscale: list, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=data.astype(float), colorscale=colorscale,
        zmin=0, zmax=2, showscale=False,
    ))
    fig.update_layout(
        paper_bgcolor=_CARD, plot_bgcolor=_CARD,
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        title={"text": title, "font": {"color": _MUTED, "size": 11}, "x": 0.5},
        xaxis={"visible": False, "scaleanchor": "y"},
        yaxis={"visible": False, "autorange": "reversed"},
        height=256,
    )
    return fig


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_CARD, plot_bgcolor=_CARD,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        xaxis={"visible": False}, yaxis={"visible": False},
        height=256,
        annotations=[{"text": message, "showarrow": False,
                      "font": {"color": _MUTED, "size": 11},
                      "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}],
    )
    return fig


def _gt_figure(level: int, sample: str) -> go.Figure:
    idx = SAMPLE_MAP.get(sample, 1)
    path = RAW_DIR / f"level{level}" / f"{idx}_true.mat"
    if not path.exists():
        return _empty_fig("Ground truth not found")
    truth = scipy.io.loadmat(str(path))["truth"].astype(np.uint8)
    cs = _CS_3 if 2 in np.unique(truth) else _CS_2
    return _heatmap_fig(truth, cs, f"GT · L{level}/{sample.upper()}")


def _cached_figures(
    algorithm: str, level: int, sample: str
) -> tuple[go.Figure, go.Figure, go.Figure, dict]:
    try:
        from ktc_vis.cache.hdf5_store import load_result
        metrics, recon = load_result(algorithm, level, sample)
    except Exception:
        msg = f"Run benchmark for {algorithm.upper()} L{level}/{sample.upper()}"
        empty = _empty_fig(msg)
        return empty, empty, _empty_fig("Requires reconstruction"), {}

    cs = _CS_3 if 2 in np.unique(recon) else _CS_2
    recon_fig = _heatmap_fig(recon, cs, f"{algorithm.upper()} · L{level}/{sample.upper()}")
    seg_fig = _heatmap_fig(recon, cs, "Segmentation")

    # Error overlay
    idx = SAMPLE_MAP.get(sample, 1)
    gt_path = RAW_DIR / f"level{level}" / f"{idx}_true.mat"
    if gt_path.exists():
        gt = scipy.io.loadmat(str(gt_path))["truth"].astype(np.uint8)
        error = np.zeros_like(gt, dtype=np.float32)
        error[np.logical_and(recon > 0, gt == 0)] = 1.0
        error[np.logical_and(recon == 0, gt > 0)] = 2.0
        error_fig = _heatmap_fig(error, _CS_ERROR, "Error Overlay")
    else:
        error_fig = _empty_fig("Ground truth not found")

    return recon_fig, seg_fig, error_fig, metrics


def _build_metrics_bar(metrics: dict) -> html.Div:
    if not metrics:
        return html.Div()
    items = [
        ("SSIM", metrics.get("ssim"), ".3f"),
        ("IoU Mean", metrics.get("iou_mean"), ".3f"),
        ("Hausdorff", metrics.get("hausdorff"), ".1f"),
        ("Position Error", metrics.get("position_error"), ".1f"),
        ("Runtime", metrics.get("runtime"), ".1f"),
    ]
    cards = []
    for label, val, fmt in items:
        if val is None:
            continue
        cards.append(html.Div([
            html.Div(label, style={"color": _MUTED, "fontSize": "10px"}),
            html.Div(f"{val:{fmt}}", style={"color": _TEXT, "fontSize": "18px",
                                            "fontWeight": "bold"}),
        ], style={"backgroundColor": _CARD, "borderRadius": "6px",
                  "padding": "10px 18px", "textAlign": "center"}))
    return html.Div(cards, style={"display": "flex", "gap": "10px", "flexWrap": "wrap"})
