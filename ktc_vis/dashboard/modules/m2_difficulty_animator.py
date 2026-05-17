"""Module 2: Difficulty Animator. Owner: Asmita Bhuva."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.io
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

# ── Constants ─────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw/ktc2023")
LEVELS = list(range(1, 8))
SAMPLE_MAP = {"a": 1, "b": 2, "c": 3}

# Segmentation colour scale: 0=water(blue), 1=resistive(red), 2=conductive(green)
_GT_COLORSCALE = [
    [0.0,  "#1565c0"],
    [0.5,  "#1565c0"],
    [0.5,  "#c62828"],
    [1.0,  "#c62828"],
]
_GT_COLORSCALE_3 = [
    [0.00, "#1565c0"],
    [0.33, "#1565c0"],
    [0.33, "#c62828"],
    [0.66, "#c62828"],
    [0.66, "#2e7d32"],
    [1.00, "#2e7d32"],
]

_DARK = "#12121e"
_PANEL = "#1e1e2f"
_CARD = "#2a2a3f"
_TEXT = "#eeeeee"
_MUTED = "#888888"


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    """Level animator: ground truth images + degradation curves across levels 1–7."""
    return html.Div([
        # ── Title ────────────────────────────────────────────────────────────
        html.H3("Difficulty Animator", style={"color": _TEXT, "marginBottom": "4px"}),
        html.P(
            "Animate how phantom complexity changes across difficulty levels 1–7. "
            "Reconstructions and metric curves appear once the benchmark cache is populated.",
            style={"color": _MUTED, "marginBottom": "20px", "fontSize": "13px"},
        ),

        # ── Controls ─────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Label("Level", style={"color": "#ccc", "fontSize": "12px",
                                           "marginBottom": "4px", "display": "block"}),
                dcc.Slider(
                    id="m2-level-slider",
                    min=1, max=7, step=1, value=1,
                    marks={i: {"label": str(i), "style": {"color": "#ccc"}}
                           for i in LEVELS},
                    tooltip={"placement": "top"},
                ),
            ], style={"flex": "1", "marginRight": "20px"}),

            html.Div([
                html.Button(
                    "▶  Play",
                    id="m2-play-btn",
                    n_clicks=0,
                    style=_btn_style("#7b61ff"),
                ),
            ], style={"paddingTop": "22px"}),
        ], style={"display": "flex", "alignItems": "flex-start",
                  "marginBottom": "20px", "backgroundColor": _PANEL,
                  "padding": "16px", "borderRadius": "8px"}),

        # ── Animation interval (disabled by default) ──────────────────────────
        dcc.Interval(id="m2-interval", interval=1200, n_intervals=0, disabled=True),

        # ── Play state store ─────────────────────────────────────────────────
        dcc.Store(id="m2-playing", data=False),

        # ── Image panels ─────────────────────────────────────────────────────
        html.Div([
            _image_card("Ground Truth", "m2-gt-graph",
                        "Loaded from KTC2023 dataset"),
            _image_card("Reconstruction", "m2-recon-graph",
                        "Requires benchmark cache — run scripts/run_benchmark.py"),
            _image_card("Error Overlay", "m2-error-graph",
                        "Green=correct · Red=false positive · Blue=false negative"),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "20px"}),

        # ── Degradation curves ────────────────────────────────────────────────
        html.Div([
            _curve_card("SSIM", "m2-ssim-graph"),
            _curve_card("Mean IoU", "m2-iou-graph"),
            _curve_card("Hausdorff Distance", "m2-hausdorff-graph"),
        ], style={"display": "flex", "gap": "12px"}),

    ], style={"padding": "20px", "backgroundColor": _DARK, "minHeight": "100vh"})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _btn_style(color: str) -> dict:
    return {
        "backgroundColor": color, "color": "#fff", "border": "none",
        "padding": "8px 20px", "borderRadius": "6px", "cursor": "pointer",
        "fontSize": "14px", "fontWeight": "bold", "minWidth": "100px",
    }


def _image_card(title: str, graph_id: str, subtitle: str) -> html.Div:
    return html.Div([
        html.Div(title, style={"color": _TEXT, "fontWeight": "bold",
                               "fontSize": "13px", "marginBottom": "2px"}),
        html.Div(subtitle, style={"color": _MUTED, "fontSize": "10px",
                                  "marginBottom": "6px"}),
        dcc.Graph(id=graph_id, config={"displayModeBar": False},
                  style={"height": "260px"}),
    ], style={"flex": "1", "backgroundColor": _CARD, "borderRadius": "8px",
              "padding": "12px"})


def _curve_card(title: str, graph_id: str) -> html.Div:
    return html.Div([
        html.Div(title, style={"color": _TEXT, "fontWeight": "bold",
                               "fontSize": "13px", "marginBottom": "6px"}),
        dcc.Graph(id=graph_id, config={"displayModeBar": False},
                  style={"height": "180px"}),
    ], style={"flex": "1", "backgroundColor": _CARD, "borderRadius": "8px",
              "padding": "12px"})


# ── Figure builders ───────────────────────────────────────────────────────────

def _gt_figure(level: int, sample: str) -> go.Figure:
    """Load ground truth .mat and return a Plotly heatmap figure."""
    idx = SAMPLE_MAP.get(sample, 1)
    mat_path = RAW_DIR / f"level{level}" / f"{idx}_true.mat"

    if not mat_path.exists():
        return _empty_figure("Data not found")

    truth = scipy.io.loadmat(str(mat_path))["truth"].astype(float)
    unique_vals = np.unique(truth)
    colorscale = _GT_COLORSCALE_3 if 2 in unique_vals else _GT_COLORSCALE

    fig = go.Figure(go.Heatmap(
        z=truth,
        colorscale=colorscale,
        zmin=0, zmax=2,
        showscale=False,
    ))
    fig.update_layout(**_image_layout(f"Level {level} · Sample {sample.upper()}"))
    return fig


def _placeholder_figure(message: str) -> go.Figure:
    return _empty_figure(message)


def _empty_figure(message: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_CARD, plot_bgcolor=_CARD,
        xaxis={"visible": False}, yaxis={"visible": False},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        annotations=[{"text": message, "showarrow": False,
                      "font": {"color": _MUTED, "size": 12},
                      "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}],
    )
    return fig


def _image_layout(title: str) -> dict:
    return {
        "paper_bgcolor": _CARD, "plot_bgcolor": _CARD,
        "margin": {"l": 4, "r": 4, "t": 28, "b": 4},
        "title": {"text": title, "font": {"color": _MUTED, "size": 11}, "x": 0.5},
        "xaxis": {"visible": False}, "yaxis": {"visible": False, "scaleanchor": "x"},
        "height": 240,
    }


def _curve_figure(metric: str, algorithm: str, sample: str) -> go.Figure:
    """Try to load metric values from HDF5 cache; show placeholder if unavailable."""
    values = _try_load_from_cache(metric, algorithm, sample)

    fig = go.Figure()
    if values is not None:
        fig.add_trace(go.Scatter(
            x=LEVELS, y=values, mode="lines+markers",
            line={"color": "#7b61ff", "width": 2},
            marker={"size": 7, "color": "#7b61ff"},
            name=algorithm.upper(),
        ))
        annotation = []
    else:
        annotation = [{"text": "Run scripts/run_benchmark.py to populate",
                       "showarrow": False, "font": {"color": _MUTED, "size": 10},
                       "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}]

    fig.update_layout(
        paper_bgcolor=_CARD, plot_bgcolor="#1a1a2e",
        margin={"l": 36, "r": 8, "t": 8, "b": 28},
        xaxis={
            "tickvals": LEVELS,
            "ticktext": [str(l) for l in LEVELS],
            "title": {"text": "Level", "font": {"color": _MUTED, "size": 10}},
            "tickfont": {"color": _MUTED, "size": 9},
            "gridcolor": "#333",
        },
        yaxis={
            "tickfont": {"color": _MUTED, "size": 9},
            "gridcolor": "#333",
        },
        annotations=annotation,
        showlegend=False,
        height=160,
    )
    return fig


def _try_load_from_cache(metric: str, algorithm: str, sample: str):
    """Return list of metric values for levels 1-7, or None if cache unavailable."""
    try:
        import h5py
        cache_path = Path("data/cache/results.h5")
        if not cache_path.exists():
            return None
        values = []
        with h5py.File(str(cache_path), "r") as f:
            for level in LEVELS:
                key = f"results/{algorithm}/{level}/{sample}/{metric}"
                if key not in f:
                    return None
                values.append(float(f[key][()]))
        return values
    except Exception:
        return None


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app) -> None:  # noqa: ANN001
    """Register all M2 callbacks."""

    # ── Play / Pause toggle ───────────────────────────────────────────────────
    @app.callback(
        Output("m2-interval", "disabled"),
        Output("m2-playing", "data"),
        Output("m2-play-btn", "children"),
        Input("m2-play-btn", "n_clicks"),
        State("m2-playing", "data"),
        prevent_initial_call=True,
    )
    def toggle_play(n_clicks, is_playing):
        playing = not is_playing
        label = "⏸  Pause" if playing else "▶  Play"
        return not playing, playing, label

    # ── Advance level on each tick ────────────────────────────────────────────
    @app.callback(
        Output("m2-level-slider", "value"),
        Input("m2-interval", "n_intervals"),
        State("m2-level-slider", "value"),
        State("m2-playing", "data"),
        prevent_initial_call=True,
    )
    def advance_level(n_intervals, current_level, is_playing):
        if not is_playing:
            raise PreventUpdate
        return (current_level % 7) + 1  # cycle 1→2→…→7→1

    # ── Update images on level / sample change ────────────────────────────────
    @app.callback(
        Output("m2-gt-graph", "figure"),
        Output("m2-recon-graph", "figure"),
        Output("m2-error-graph", "figure"),
        Input("m2-level-slider", "value"),
        Input("sidebar-sample-radio", "value"),
        Input("sidebar-algorithm-dropdown", "value"),
    )
    def update_images(level, sample, algorithm):
        gt_fig = _gt_figure(level, sample)
        recon_fig = _placeholder_figure(
            f"No cached reconstruction for {algorithm.upper()} L{level}/{sample.upper()}\n"
            "Run scripts/run_benchmark.py"
        )
        error_fig = _placeholder_figure("Requires reconstruction cache")
        return gt_fig, recon_fig, error_fig

    # ── Update degradation curves ─────────────────────────────────────────────
    @app.callback(
        Output("m2-ssim-graph", "figure"),
        Output("m2-iou-graph", "figure"),
        Output("m2-hausdorff-graph", "figure"),
        Input("sidebar-algorithm-dropdown", "value"),
        Input("sidebar-sample-radio", "value"),
    )
    def update_curves(algorithm, sample):
        ssim_fig = _curve_figure("ssim", algorithm, sample)
        iou_fig = _curve_figure("iou_mean", algorithm, sample)
        hausdorff_fig = _curve_figure("hausdorff", algorithm, sample)
        return ssim_fig, iou_fig, hausdorff_fig
