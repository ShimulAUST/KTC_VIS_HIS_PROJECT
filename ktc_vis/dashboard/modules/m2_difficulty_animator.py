"""Module 2: Difficulty Animator. Owner: Asmita Bhuva."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.io
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

# ── Level metadata (KTC2023 electrode / injection protocol per level) ─────────
_LEVEL_META = {
    1: {"electrodes": 32, "injections": 76, "measurements": 2356},
    2: {"electrodes": 30, "injections": 52, "measurements": 1624},
    3: {"electrodes": 28, "injections": 52, "measurements": 1404},
    4: {"electrodes": 26, "injections": 46, "measurements": 1200},
    5: {"electrodes": 24, "injections": 44, "measurements": 1012},
    6: {"electrodes": 22, "injections": 30, "measurements": 630},
    7: {"electrodes": 20, "injections": 27, "measurements": 513},
}

# ── Constants ─────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "ktc2023"
_CACHE_PATH = _PROJECT_ROOT / "data" / "cache" / "results.h5"
LEVELS = list(range(1, 8))
SAMPLE_MAP = {"a": 1, "b": 2, "c": 3, "d": 4}

# Segmentation colour scale: 0=water(blue), 1=resistive(red), 2=conductive(green)
_GT_COLORSCALE = [
    [0.0, "#1565c0"],
    [0.5, "#1565c0"],
    [0.5, "#c62828"],
    [1.0, "#c62828"],
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

# Chip row colors (match M1 theme)
_CHIP_CARD = {"backgroundColor": "#1e1e2f", "border": "1px solid #2e2e44",
              "borderRadius": "12px", "boxShadow": "0 1px 2px rgba(0,0,0,0.25)"}
_WARN = "#f4c870"
_SUCCESS = "#2ecc71"
_DANGER = "#e85d75"


# ── Layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    """Level animator: ground truth images + degradation curves across levels 1–7."""
    return html.Div([

        # ── Module header ─────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("M2", style={
                    "display": "inline-block", "padding": "2px 10px",
                    "borderRadius": "999px", "backgroundColor": "#7b61ff",
                    "color": "#fff", "fontSize": "11px", "fontWeight": 600,
                    "letterSpacing": "0.5px", "marginRight": "10px",
                }),
                html.Span("Difficulty Animator", style={
                    "color": _TEXT, "fontSize": "20px", "fontWeight": 600,
                }),
            ]),
            html.P(
                "The KTC2023 benchmark defines 7 difficulty levels by progressively removing "
                "electrodes (32 → 20) and voltage measurements (2356 → 513). "
                "Use the slider or ▶ Play to animate how the chosen algorithm's reconstruction "
                "quality degrades as measurement data decreases. "
                "The degradation curves below show how each quality metric evolves across all 7 levels — "
                "revealing whether an algorithm fails gradually or collapses suddenly.",
                style={"color": _MUTED, "margin": "8px 0 0", "fontSize": "13px",
                       "lineHeight": "1.6"},
            ),
        ], style={
            "backgroundColor": _PANEL, "borderRadius": "10px",
            "padding": "16px 20px", "marginBottom": "16px",
            "borderLeft": "3px solid #7b61ff",
        }),

        # ── Chip row ─────────────────────────────────────────────────────────
        html.Div(
            id="m2-chips",
            children=[_chip("status", "loading…")],
            style={"display": "flex", "flexWrap": "wrap", "gap": "8px",
                   "marginBottom": "16px"},
        ),

        # ── Level context bar (dynamic) ───────────────────────────────────────
        html.Div(id="m2-level-context", style={"marginBottom": "14px"}),

        # ── Controls ─────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div([
                    html.Span("Difficulty Level", style={
                        "color": _TEXT, "fontSize": "12px", "fontWeight": 600,
                        "marginRight": "10px",
                    }),
                    html.Span(
                        "Drag to jump to any level, or press ▶ Play to animate levels 1 → 7 → 1 in a loop.",
                        style={"color": _MUTED, "fontSize": "11px"},
                    ),
                ], style={"marginBottom": "8px"}),
                dcc.Slider(
                    id="m2-level-slider",
                    min=1, max=7, step=1, value=1,
                    marks={i: {
                        "label": f"L{i}",
                        "style": {"color": "#aaa", "fontSize": "11px"},
                    } for i in LEVELS},
                    tooltip={"placement": "top", "always_visible": False},
                ),
            ], style={"flex": "1", "marginRight": "24px"}),

            html.Div([
                html.Button("▶  Play", id="m2-play-btn", n_clicks=0,
                            style=_btn_style("#7b61ff")),
                html.Div("1.2 s / level", style={
                    "color": _MUTED, "fontSize": "10px", "textAlign": "center",
                    "marginTop": "4px",
                }),
            ], style={"paddingTop": "24px", "textAlign": "center"}),
        ], style={
            "display": "flex", "alignItems": "flex-start",
            "marginBottom": "20px", "backgroundColor": _PANEL,
            "padding": "16px 20px", "borderRadius": "8px",
        }),

        # ── Animation interval (disabled by default) ──────────────────────────
        dcc.Interval(id="m2-interval", interval=1200, n_intervals=0, disabled=True),

        # ── Play state store ─────────────────────────────────────────────────
        dcc.Store(id="m2-playing", data=False),

        # ── Image panels section label ────────────────────────────────────────
        _section_label(
            "Visual Comparison",
            "Ground truth vs algorithm output at the selected level — watch the error overlay grow as level increases",
        ),

        # ── Image panels ─────────────────────────────────────────────────────
        html.Div([
            _image_card("Ground Truth", "m2-gt-graph",
                        "True phantom geometry — does not change with level"),
            _image_card("Segmentation", "m2-recon-graph",
                        "Algorithm's predicted 3-class map (water / resistive / conductive)"),
            _image_card("Error Overlay", "m2-error-graph",
                        "Pixel-wise agreement · Green=correct · Red=FP · Orange=wrong class · Blue=FN"),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "20px"}),

        # ── Degradation curves section label ──────────────────────────────────
        _section_label(
            "Degradation Curves",
            "Each point is one level — vertical dashed line marks the currently selected level",
        ),

        # ── Degradation curves (row 1: quality metrics) ───────────────────────
        html.Div([
            _curve_card("SSIM  ·  higher is better  ·  range 0–1",
                        "m2-ssim-graph",
                        "Structural Similarity: measures luminance, contrast & structure against the ground truth. "
                        "Official KTC2023 ranking metric."),
            _curve_card("Mean IoU  ·  higher is better  ·  range 0–1",
                        "m2-iou-graph",
                        "Intersection-over-Union averaged across all 3 classes (water, resistive, conductive). "
                        "Penalises both false positives and false negatives equally."),
            _curve_card("Dice Score  ·  higher is better  ·  range 0–1",
                        "m2-dice-graph",
                        "Dice = 2·IoU / (1+IoU). Equivalent to F1-score for segmentation. "
                        "More sensitive to small regions than IoU."),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        # ── Degradation curves (row 2: shape metrics) ─────────────────────────
        html.Div([
            _curve_card("Hausdorff Distance (px)  ·  lower is better",
                        "m2-hausdorff-graph",
                        "Worst-case boundary error in pixels: the maximum distance between predicted "
                        "and true inclusion edges. Rises sharply when inclusions are smeared or missed."),
            _curve_card("Position Error (px)  ·  lower is better",
                        "m2-poserr-graph",
                        "Euclidean distance between predicted and true inclusion centroids. "
                        "Indicates whether the algorithm puts inclusions in the right place."),
            _curve_card("Resolution (px)  ·  smaller = finer detail",
                        "m2-resolution-graph",
                        "Diameter of the smallest detected inclusion. "
                        "Grows as measurement sparsity blurs fine structures."),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        # ── Degradation curves (row 3: speed) ─────────────────────────────────
        html.Div([
            _curve_card("Runtime (s)  ·  lower is faster  ·  axis 0–60 s",
                        "m2-runtime-graph",
                        "Wall-clock seconds to produce one reconstruction. "
                        "Y-axis is fixed (0–60 s) so values are directly comparable across algorithms. "
                        "Available only when running live algorithms via scripts/benchmark_runtime_*.py."),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "20px"}),

        # ── Dynamic commentary section label ──────────────────────────────────
        _section_label(
            "Level Analysis",
            "Automatic interpretation of the selected level's metrics and what they mean physically",
        ),

        # ── Dynamic commentary panel ──────────────────────────────────────────
        html.Div(
            id="m2-commentary",
            style={
                "backgroundColor": _PANEL, "borderRadius": "8px",
                "padding": "16px 20px", "borderLeft": "3px solid #7b61ff",
                "minHeight": "60px",
            },
        ),

    ], style={"padding": "20px", "backgroundColor": _DARK, "minHeight": "100vh"})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chip(label: str, value: str, accent: str | None = None) -> html.Div:
    return html.Div(
        [
            html.Span(label, style={
                "color": _MUTED, "fontSize": "10.5px", "letterSpacing": "0.6px",
                "textTransform": "uppercase", "marginRight": "8px",
            }),
            html.Span(value, style={
                "color": accent or _TEXT, "fontSize": "13px",
                "fontWeight": 600, "fontVariantNumeric": "tabular-nums",
            }),
        ],
        style={**_CHIP_CARD, "padding": "8px 12px",
               "display": "inline-flex", "alignItems": "center"},
    )


def _section_label(title: str, subtitle: str) -> html.Div:
    return html.Div([
        html.Span(title, style={
            "color": _TEXT, "fontWeight": 600, "fontSize": "12px",
            "letterSpacing": "0.8px", "textTransform": "uppercase",
            "marginRight": "10px",
        }),
        html.Span(subtitle, style={"color": _MUTED, "fontSize": "12px"}),
    ], style={
        "display": "flex", "alignItems": "baseline",
        "padding": "4px 2px 0", "marginBottom": "10px",
        "borderBottom": "1px dashed #2e2e44", "paddingBottom": "6px",
    })


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


def _curve_card(title: str, graph_id: str, description: str = "") -> html.Div:
    children = [
        html.Div(title, style={"color": _TEXT, "fontWeight": "bold",
                               "fontSize": "12px", "marginBottom": "2px"}),
    ]
    if description:
        children.append(html.Div(description, style={
            "color": _MUTED, "fontSize": "10px", "lineHeight": "1.4",
            "marginBottom": "6px",
        }))
    else:
        children.append(html.Div(style={"marginBottom": "6px"}))
    children.append(
        dcc.Graph(id=graph_id, config={"displayModeBar": False},
                  style={"height": "160px"}),
    )
    return html.Div(children, style={
        "flex": "1", "backgroundColor": _CARD, "borderRadius": "8px", "padding": "12px",
    })


# ── Figure builders ───────────────────────────────────────────────────────────

def _gt_figure(level: int, sample: str) -> go.Figure:
    """Load ground truth .mat and return a Plotly heatmap figure."""
    idx = SAMPLE_MAP.get(sample, 1)
    mat_path = RAW_DIR / "ground_truth" / f"true{idx}.mat"

    if not mat_path.exists():
        return _empty_figure("Data not found")

    truth = scipy.io.loadmat(str(mat_path))["truth"].astype(float)

    fig = go.Figure(go.Heatmap(
        z=truth,
        colorscale=_GT_COLORSCALE_3,
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
        "margin": {"l": 0, "r": 0, "t": 24, "b": 0},
        "title": {"text": title, "font": {"color": _MUTED, "size": 11}, "x": 0.5,
                  "pad": {"t": 4}},
        "xaxis": {"visible": False, "scaleanchor": "y"},
        "yaxis": {"visible": False, "autorange": "reversed"},
        "height": 260,
        "uirevision": "constant",
    }


# Runtime values below this threshold (seconds) are treated as synthetic:
# the reference-output cache fills runtime with 0.0 or with the cost of reading
# a .mat file (~1 ms), neither of which reflects real reconstruction time.
_RUNTIME_REAL_THRESHOLD_S = 0.01

# Fixed Y-axis ranges per metric so users can compare across algorithms without
# the chart auto-rescaling on every algorithm/sample change.
_FIXED_Y_RANGE: dict[str, tuple[float, float]] = {
    "ssim":           (0.0, 1.0),
    "iou_mean":       (0.0, 1.0),
    "dice":           (0.0, 1.0),
    "runtime":        (0.0, 60.0),   # seconds; covers ABC1/PNPE2E (~17-27s) + CUQI8 headroom
}


def _curve_figure(metric: str, algorithm: str, sample: str,
                  current_level: int = 1) -> go.Figure:
    """Try to load metric values from HDF5 cache; show placeholder if unavailable."""
    values = _try_load_from_cache(metric, algorithm, sample)

    if values is not None:
        if metric == "runtime":
            # Reference-output cache stores ~0s; only treat as real if any sample is meaningfully above zero.
            if max(values) < _RUNTIME_REAL_THRESHOLD_S:
                values = None
        elif all(v == 0.0 for v in values):
            values = None

    fig = go.Figure()
    annotation = []

    if values is not None:
        fig.add_trace(go.Scatter(
            x=LEVELS, y=values, mode="lines+markers",
            line={"color": "#7b61ff", "width": 2},
            marker={"size": 7, "color": "#7b61ff"},
            name=algorithm.upper(),
        ))
        # Highlight current level with a vertical dashed line
        fig.add_vline(
            x=current_level,
            line={"color": "#ffb300", "width": 1.5, "dash": "dash"},
        )
    else:
        msg = ("Requires live benchmark — run scripts/run_benchmark.py"
               if metric == "runtime"
               else "Run scripts/run_benchmark.py to populate")
        annotation = [{"text": msg, "showarrow": False,
                       "font": {"color": _MUTED, "size": 10},
                       "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}]

    yaxis_cfg = {
        "tickfont": {"color": _MUTED, "size": 9},
        "gridcolor": "#2a2a3f",
        "zeroline": False,
    }
    if metric in _FIXED_Y_RANGE:
        lo, hi = _FIXED_Y_RANGE[metric]
        yaxis_cfg["range"] = [lo, hi]
        yaxis_cfg["autorange"] = False

    fig.update_layout(
        paper_bgcolor=_CARD, plot_bgcolor="#1a1a2e",
        margin={"l": 36, "r": 8, "t": 8, "b": 36},
        xaxis={
            "range": [0.5, 7.5],
            "tickvals": LEVELS,
            "ticktext": [str(lv) for lv in LEVELS],
            "title": {"text": "Level", "font": {"color": _MUTED, "size": 10}},
            "tickfont": {"color": _MUTED, "size": 9},
            "gridcolor": "#2a2a3f",
            "zeroline": False,
        },
        yaxis=yaxis_cfg,
        annotations=annotation,
        showlegend=False,
        height=160,
    )
    return fig


def _dice_curve_figure(algorithm: str, sample: str, current_level: int = 1) -> go.Figure:
    """Compute Dice from cached IoU (dice = 2·IoU / (1 + IoU)) and plot as a curve."""
    iou_vals = _try_load_from_cache("iou_mean", algorithm, sample)
    if iou_vals is not None:
        dice_vals = [2 * v / (1 + v) if (1 + v) != 0 else 0.0 for v in iou_vals]
    else:
        dice_vals = None
    return _curve_figure_from_values(dice_vals, algorithm, sample, current_level, metric="dice")


def _curve_figure_from_values(
    values: list | None, algorithm: str, sample: str, current_level: int = 1,
    metric: str | None = None,
) -> go.Figure:
    """Like _curve_figure but accepts pre-computed values instead of a metric name."""
    fig = go.Figure()
    annotation = []
    if values is not None:
        fig.add_trace(go.Scatter(
            x=LEVELS, y=values, mode="lines+markers",
            line={"color": "#7b61ff", "width": 2},
            marker={"size": 7, "color": "#7b61ff"},
            name=algorithm.upper(),
        ))
        fig.add_vline(x=current_level,
                      line={"color": "#ffb300", "width": 1.5, "dash": "dash"})
    else:
        annotation = [{"text": "Run scripts/run_benchmark.py to populate",
                       "showarrow": False, "font": {"color": _MUTED, "size": 10},
                       "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}]

    yaxis_cfg = {"tickfont": {"color": _MUTED, "size": 9},
                 "gridcolor": "#2a2a3f", "zeroline": False}
    if metric and metric in _FIXED_Y_RANGE:
        lo, hi = _FIXED_Y_RANGE[metric]
        yaxis_cfg["range"] = [lo, hi]
        yaxis_cfg["autorange"] = False

    fig.update_layout(
        paper_bgcolor=_CARD, plot_bgcolor="#1a1a2e",
        margin={"l": 36, "r": 8, "t": 8, "b": 36},
        xaxis={
            "range": [0.5, 7.5], "tickvals": LEVELS,
            "ticktext": [str(lv) for lv in LEVELS],
            "title": {"text": "Level", "font": {"color": _MUTED, "size": 10}},
            "tickfont": {"color": _MUTED, "size": 9},
            "gridcolor": "#2a2a3f", "zeroline": False,
        },
        yaxis=yaxis_cfg,
        annotations=annotation, showlegend=False, height=160,
        uirevision="constant",
    )
    return fig


def _recon_and_error_figures(
    level: int, sample: str, algorithm: str
) -> tuple[go.Figure, go.Figure]:
    """Return reconstruction and error overlay figures from cache, or placeholders."""
    try:
        from ktc_vis.cache.hdf5_store import load_result
        _, recon = load_result(algorithm, level, sample, cache_path=_CACHE_PATH)
    except Exception:
        msg = f"No cache for {algorithm.upper()} L{level}/{sample.upper()} — run benchmark"
        return _empty_figure(msg), _empty_figure("Requires reconstruction cache")

    recon_fig = go.Figure(go.Heatmap(
        z=recon.astype(float), colorscale=_GT_COLORSCALE_3,
        zmin=0, zmax=2, showscale=False,
    ))
    recon_fig.update_layout(**_image_layout(f"{algorithm.upper()} L{level}/{sample.upper()}"))

    # Error overlay: 0=correct(green), 1=false-pos(red), 2=false-neg(blue)
    idx = SAMPLE_MAP.get(sample, 1)
    gt_path = RAW_DIR / "ground_truth" / f"true{idx}.mat"
    if not gt_path.exists():
        return recon_fig, _empty_figure("Ground truth not found")

    gt = scipy.io.loadmat(str(gt_path))["truth"].astype(np.int16)
    recon_i = recon.astype(np.int16)
    error = np.zeros_like(gt, dtype=np.float32)
    correct_mask = recon_i == gt
    error[correct_mask & (gt != 0)] = 0.0               # correct inclusion — green
    error[(recon_i != 0) & (gt == 0)] = 1.0             # false positive — red
    error[(recon_i == 0) & (gt != 0)] = 2.0             # false negative — blue
    error[(recon_i != 0) & (gt != 0) & ~correct_mask] = 1.5  # wrong class — orange

    error_cs = [
        [0.00, "#2e7d32"], [0.30, "#2e7d32"],   # correct — green
        [0.30, "#c62828"], [0.55, "#c62828"],   # false pos — red
        [0.55, "#f57f17"], [0.75, "#f57f17"],   # wrong class — orange
        [0.75, "#1565c0"], [1.00, "#1565c0"],   # false neg — blue
    ]
    error_fig = go.Figure(go.Heatmap(
        z=error, colorscale=error_cs, zmin=0, zmax=2, showscale=False,
    ))
    error_fig.update_layout(**_image_layout("Error Overlay"))
    return recon_fig, error_fig


def _try_load_from_cache(metric: str, algorithm: str, sample: str):
    """Return list of metric values for levels 1-7, or None if cache unavailable."""
    try:
        import h5py
        if not _CACHE_PATH.exists():
            return None
        values = []
        with h5py.File(str(_CACHE_PATH), "r") as f:
            for level in LEVELS:
                key = f"results/{algorithm}/{level}/{sample}/{metric}"
                if key not in f:
                    return None
                values.append(float(f[key][()]))
        return values
    except Exception:
        return None


# ── Commentary helpers ────────────────────────────────────────────────────────

def _badge(text: str, color: str) -> html.Span:
    return html.Span(text, style={
        "backgroundColor": color, "color": "#fff", "borderRadius": "4px",
        "padding": "2px 8px", "fontSize": "11px", "fontWeight": "bold",
        "marginRight": "6px",
    })


def _ssim_label(v: float) -> tuple[str, str]:
    if v >= 0.85:
        return "Excellent", "#2e7d32"
    if v >= 0.70:
        return "Good", "#558b2f"
    if v >= 0.55:
        return "Moderate", "#f57f17"
    if v >= 0.40:
        return "Poor", "#e65100"
    return "Very Poor", "#c62828"


def _hausdorff_label(v: float) -> tuple[str, str]:
    if v <= 8:
        return "Tight boundaries", "#2e7d32"
    if v <= 15:
        return "Acceptable boundaries", "#558b2f"
    if v <= 25:
        return "Blurred boundaries", "#f57f17"
    return "Severe boundary error", "#c62828"


def _iou_label(v: float) -> tuple[str, str]:
    if v >= 0.75:
        return "High overlap", "#2e7d32"
    if v >= 0.55:
        return "Moderate overlap", "#558b2f"
    if v >= 0.35:
        return "Low overlap", "#f57f17"
    return "Poor segmentation", "#c62828"


def _poserr_label(v: float) -> tuple[str, str]:
    if v <= 10:
        return "Centroid on-target", "#2e7d32"
    if v <= 25:
        return "Centroid offset", "#f57f17"
    return "Centroid mislocated", "#c62828"


def _resolution_label(v: float) -> tuple[str, str]:
    if v <= 20:
        return "Fine detail preserved", "#2e7d32"
    if v <= 40:
        return "Moderate blurring", "#558b2f"
    if v <= 60:
        return "Coarse only", "#f57f17"
    return "Resolution collapse", "#c62828"


def _trend_text(current: float, prev: float, metric: str, higher_is_better: bool) -> str:
    delta = current - prev
    pct = abs(delta / prev * 100) if prev != 0 else 0
    direction = "↑" if delta > 0 else "↓"
    if higher_is_better:
        sentiment = "improved" if delta > 0 else "dropped"
    else:
        sentiment = "worsened" if delta > 0 else "improved"
    if pct < 2:
        return f"{metric} stable ({direction}{pct:.1f}%)"
    return f"{metric} {sentiment} {direction}{pct:.1f}% (was {prev:.3f})"


def _build_commentary(level: int, algorithm: str, sample: str) -> list:
    meta = _LEVEL_META[level]
    prev_meta = _LEVEL_META.get(level - 1)
    data_ratio = meta["measurements"] / _LEVEL_META[1]["measurements"]

    # ── Section 1: Hardware context ───────────────────────────────────────────
    context_parts = [
        _badge(f"Level {level}", "#7b61ff"),
        html.Span(
            f"{meta['electrodes']} electrodes · {meta['injections']} injections · "
            f"{meta['measurements']} measurements "
            f"({data_ratio * 100:.0f}% of Level 1 data)",
            style={"color": _TEXT, "fontSize": "13px"},
        ),
    ]
    if prev_meta:
        elec_lost = prev_meta["electrodes"] - meta["electrodes"]
        meas_lost = prev_meta["measurements"] - meta["measurements"]
        context_parts.append(html.Span(
            f"  ·  {elec_lost} fewer electrode(s), {meas_lost} fewer measurements than Level {level - 1}",
            style={"color": _MUTED, "fontSize": "12px"},
        ))

    rows = [
        html.Div(context_parts, style={"marginBottom": "10px"}),
    ]

    # ── Section 2: Metric commentary from cache ───────────────────────────────
    ssim_vals = _try_load_from_cache("ssim", algorithm, sample)
    iou_vals = _try_load_from_cache("iou_mean", algorithm, sample)
    hd_vals = _try_load_from_cache("hausdorff", algorithm, sample)
    pe_vals = _try_load_from_cache("position_error", algorithm, sample)
    res_vals = _try_load_from_cache("resolution", algorithm, sample)

    if ssim_vals is None and iou_vals is None and hd_vals is None:
        rows.append(html.Div(
            "Benchmark cache not yet available for this combination — "
            "run scripts/run_benchmark.py to see metric commentary.",
            style={"color": _MUTED, "fontSize": "12px", "fontStyle": "italic"},
        ))
        return rows

    metric_items = []
    idx = level - 1  # 0-based index into values list
    prev_idx = idx - 1

    if ssim_vals and idx < len(ssim_vals):
        v = ssim_vals[idx]
        label, color = _ssim_label(v)
        line = [_badge(label, color), html.Span(f"SSIM = {v:.3f}", style={"color": _TEXT, "fontSize": "12px"})]
        if prev_idx >= 0:
            delta = v - ssim_vals[prev_idx]
            pct = delta / ssim_vals[prev_idx] * 100 if ssim_vals[prev_idx] != 0 else 0
            arrow = "↑" if delta > 0 else "↓"
            color2 = "#4caf50" if delta > 0 else "#ef5350"
            line.append(html.Span(f"  {arrow} {abs(pct):.1f}% vs L{level - 1}",
                                  style={"color": color2, "fontSize": "11px"}))
        metric_items.append(html.Li(line, style={"marginBottom": "4px"}))

    if iou_vals and idx < len(iou_vals):
        v = iou_vals[idx]
        label, color = _iou_label(v)
        line = [_badge(label, color), html.Span(f"Mean IoU = {v:.3f}", style={"color": _TEXT, "fontSize": "12px"})]
        if prev_idx >= 0:
            delta = v - iou_vals[prev_idx]
            pct = delta / iou_vals[prev_idx] * 100 if iou_vals[prev_idx] != 0 else 0
            arrow = "↑" if delta > 0 else "↓"
            color2 = "#4caf50" if delta > 0 else "#ef5350"
            line.append(html.Span(f"  {arrow} {abs(pct):.1f}% vs L{level - 1}",
                                  style={"color": color2, "fontSize": "11px"}))
        metric_items.append(html.Li(line, style={"marginBottom": "4px"}))

    if hd_vals and idx < len(hd_vals):
        v = hd_vals[idx]
        label, color = _hausdorff_label(v)
        line = [_badge(label, color), html.Span(f"Hausdorff = {v:.1f} px", style={"color": _TEXT, "fontSize": "12px"})]
        if prev_idx >= 0:
            delta = v - hd_vals[prev_idx]
            pct = delta / hd_vals[prev_idx] * 100 if hd_vals[prev_idx] != 0 else 0
            arrow = "↑" if delta > 0 else "↓"
            color2 = "#ef5350" if delta > 0 else "#4caf50"  # higher Hausdorff = worse
            line.append(html.Span(f"  {arrow} {abs(pct):.1f}% vs L{level - 1}",
                                  style={"color": color2, "fontSize": "11px"}))
        metric_items.append(html.Li(line, style={"marginBottom": "4px"}))

    if pe_vals and idx < len(pe_vals):
        v = pe_vals[idx]
        label, color = _poserr_label(v)
        line = [_badge(label, color), html.Span(f"Position error = {v:.1f} px", style={"color": _TEXT, "fontSize": "12px"})]
        if prev_idx >= 0:
            delta = v - pe_vals[prev_idx]
            pct = delta / pe_vals[prev_idx] * 100 if pe_vals[prev_idx] != 0 else 0
            arrow = "↑" if delta > 0 else "↓"
            color2 = "#ef5350" if delta > 0 else "#4caf50"  # higher offset = worse
            line.append(html.Span(f"  {arrow} {abs(pct):.1f}% vs L{level - 1}",
                                  style={"color": color2, "fontSize": "11px"}))
        metric_items.append(html.Li(line, style={"marginBottom": "4px"}))

    if res_vals and idx < len(res_vals):
        v = res_vals[idx]
        label, color = _resolution_label(v)
        line = [_badge(label, color), html.Span(f"Resolution = {v:.0f} px", style={"color": _TEXT, "fontSize": "12px"})]
        if prev_idx >= 0:
            delta = v - res_vals[prev_idx]
            pct = delta / res_vals[prev_idx] * 100 if res_vals[prev_idx] != 0 else 0
            arrow = "↑" if delta > 0 else "↓"
            color2 = "#ef5350" if delta > 0 else "#4caf50"  # larger smallest-detected = worse detail
            line.append(html.Span(f"  {arrow} {abs(pct):.1f}% vs L{level - 1}",
                                  style={"color": color2, "fontSize": "11px"}))
        metric_items.append(html.Li(line, style={"marginBottom": "4px"}))

    if metric_items:
        rows.append(html.Ul(metric_items, style={"margin": "0 0 10px 0", "paddingLeft": "16px"}))

    # ── Section 3: Plain-English summary ──────────────────────────────────────
    summary = _summary_sentence(level, algorithm, ssim_vals, iou_vals, hd_vals)
    if summary:
        rows.append(html.Div(summary, style={
            "color": _MUTED, "fontSize": "12px", "fontStyle": "italic",
            "borderTop": "1px solid #2a2a3f", "paddingTop": "8px",
        }))

    return rows


def _summary_sentence(
    level: int,
    algorithm: str,
    ssim_vals: list | None,
    iou_vals: list | None,
    hd_vals: list | None,
) -> str:
    idx = level - 1
    alg = algorithm.upper()

    if not ssim_vals or idx >= len(ssim_vals):
        return ""

    ssim = ssim_vals[idx]

    # Find the first level where quality dropped below a threshold
    drop_levels = [i + 1 for i, v in enumerate(ssim_vals) if v < 0.60]
    cliff_str = ""
    if drop_levels:
        cliff = drop_levels[0]
        if level == cliff:
            cliff_str = f" This is the level where {alg} crosses below acceptable quality (SSIM < 0.60)."
        elif level > cliff:
            cliff_str = f" {alg} dropped below acceptable quality at Level {cliff}."

    if ssim >= 0.85:
        base = f"{alg} performs excellently at Level {level} — with {_LEVEL_META[level]['measurements']} measurements, the EIT inverse problem is still well-constrained."
    elif ssim >= 0.70:
        base = f"{alg} maintains reasonable quality at Level {level}, though reduced electrode count ({_LEVEL_META[level]['electrodes']}) starts limiting spatial resolution."
    elif ssim >= 0.55:
        base = f"At Level {level}, {alg} shows visible degradation — fewer measurement paths ({_LEVEL_META[level]['measurements']}) mean the inverse problem becomes under-determined."
    elif ssim >= 0.40:
        base = f"Level {level} is challenging for {alg}: only {_LEVEL_META[level]['electrodes']} electrodes leaves large angular gaps, causing significant reconstruction artifacts."
    else:
        base = f"At Level {level}, {alg} is operating near its limit — with only {_LEVEL_META[level]['measurements']} measurements ({_LEVEL_META[level]['measurements'] / _LEVEL_META[1]['measurements'] * 100:.0f}% of Level 1), reliable reconstruction is very difficult."

    return base + cliff_str


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

    # ── Sync sidebar level slider → m2-level-slider, or advance on tick ─────────
    @app.callback(
        Output("m2-level-slider", "value"),
        Input("sidebar-level-slider", "value"),
        Input("m2-interval", "n_intervals"),
        State("m2-level-slider", "value"),
        State("m2-playing", "data"),
        prevent_initial_call=True,
    )
    def advance_level(sidebar_level, n_intervals, current_level, is_playing):
        from dash import callback_context
        trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
        if "sidebar-level-slider" in trigger:
            return sidebar_level
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
        recon_fig, error_fig = _recon_and_error_figures(level, sample, algorithm)
        return gt_fig, recon_fig, error_fig

    # ── Update degradation curves ─────────────────────────────────────────────
    @app.callback(
        Output("m2-ssim-graph", "figure"),
        Output("m2-iou-graph", "figure"),
        Output("m2-dice-graph", "figure"),
        Output("m2-hausdorff-graph", "figure"),
        Output("m2-poserr-graph", "figure"),
        Output("m2-resolution-graph", "figure"),
        Output("m2-runtime-graph", "figure"),
        Input("sidebar-algorithm-dropdown", "value"),
        Input("sidebar-sample-radio", "value"),
        Input("m2-level-slider", "value"),
    )
    def update_curves(algorithm, sample, level):
        ssim_fig = _curve_figure("ssim", algorithm, sample, level)
        iou_fig = _curve_figure("iou_mean", algorithm, sample, level)
        dice_fig = _dice_curve_figure(algorithm, sample, level)
        hausdorff_fig = _curve_figure("hausdorff", algorithm, sample, level)
        poserr_fig = _curve_figure("position_error", algorithm, sample, level)
        resolution_fig = _curve_figure("resolution", algorithm, sample, level)
        runtime_fig = _curve_figure("runtime", algorithm, sample, level)
        return ssim_fig, iou_fig, dice_fig, hausdorff_fig, poserr_fig, resolution_fig, runtime_fig

    # ── Update dynamic commentary ─────────────────────────────────────────────
    @app.callback(
        Output("m2-commentary", "children"),
        Input("m2-level-slider", "value"),
        Input("sidebar-algorithm-dropdown", "value"),
        Input("sidebar-sample-radio", "value"),
    )
    def update_commentary(level, algorithm, sample):
        return _build_commentary(level, algorithm, sample)

    # ── Update chip row ───────────────────────────────────────────────────────
    @app.callback(
        Output("m2-chips", "children"),
        Input("m2-level-slider", "value"),
        Input("sidebar-algorithm-dropdown", "value"),
        Input("sidebar-sample-radio", "value"),
    )
    def update_chips(level, algorithm, sample):
        meta = _LEVEL_META[level]
        chips = [
            _chip("algorithm", algorithm.upper(), accent="#cfe0ff"),
            _chip("level", f"L{level}", accent=_WARN),
            _chip("sample", sample.upper(), accent="#cfe0ff"),
            _chip("electrodes", str(meta["electrodes"])),
            _chip("injections", str(meta["injections"])),
            _chip("measurements", f"{meta['measurements']:,}"),
        ]

        # pixel agreement from cached reconstruction vs GT
        try:
            from ktc_vis.cache.hdf5_store import load_result
            metrics, recon = load_result(algorithm, level, sample, cache_path=_CACHE_PATH)
            idx = SAMPLE_MAP[sample]
            gt_path = RAW_DIR / "ground_truth" / f"true{idx}.mat"
            gt = scipy.io.loadmat(str(gt_path))["truth"].astype(np.uint8)
            agreement = float((recon == gt).mean()) * 100.0
            chips.append(_chip(
                "pixel agreement", f"{agreement:.1f}%",
                accent=_SUCCESS if agreement >= 90 else _WARN,
            ))

            pos_err = metrics.get("position_error")
            if pos_err is not None:
                accent = _SUCCESS if pos_err <= 10 else (_WARN if pos_err <= 25 else _DANGER)
                chips.append(_chip("position err", f"{pos_err:.1f} px", accent=accent))

            resolution = metrics.get("resolution")
            if resolution is not None:
                accent = _SUCCESS if resolution <= 30 else (_WARN if resolution <= 60 else _DANGER)
                chips.append(_chip("resolution", f"{resolution:.0f} px", accent=accent))

            runtime = metrics.get("runtime")
            if runtime is not None and runtime >= _RUNTIME_REAL_THRESHOLD_S:
                chips.append(_chip("runtime", f"{runtime:.2f} s"))
        except Exception:
            chips.append(_chip("pixel agreement", "n/a", accent=_MUTED))

        return chips

    # ── Level context bar ─────────────────────────────────────────────────────
    @app.callback(
        Output("m2-level-context", "children"),
        Input("m2-level-slider", "value"),
    )
    def update_level_context(level):
        meta = _LEVEL_META[level]
        l1 = _LEVEL_META[1]
        meas_pct = meta["measurements"] / l1["measurements"] * 100
        elec_lost = l1["electrodes"] - meta["electrodes"]
        meas_lost = l1["measurements"] - meta["measurements"]

        if level == 1:
            description = "Full measurement protocol — maximum data, best expected reconstruction quality."
            bar_color = _SUCCESS
        elif level <= 3:
            description = f"{elec_lost} electrodes removed, {meas_lost:,} fewer measurements than Level 1. Mild data reduction — most algorithms still perform well."
            bar_color = "#4caf50"
        elif level <= 5:
            description = f"{elec_lost} electrodes removed, {meas_lost:,} fewer measurements. Moderate data loss — inverse problem becomes under-determined. Quality starts to drop."
            bar_color = _WARN
        else:
            description = f"{elec_lost} electrodes removed, {meas_lost:,} fewer measurements. Severe data loss — only {meas_pct:.0f}% of Level 1 data remains. Most algorithms degrade significantly."
            bar_color = _DANGER

        bar_width = meas_pct

        return html.Div([
            html.Div([
                html.Span(f"Level {level} data availability: {meas_pct:.0f}% of Level 1",
                          style={"color": _TEXT, "fontSize": "12px", "fontWeight": 600}),
                html.Span(f"  ·  {meta['electrodes']} electrodes, {meta['measurements']:,} measurements",
                          style={"color": _MUTED, "fontSize": "11px"}),
            ], style={"marginBottom": "6px", "display": "flex", "alignItems": "baseline", "gap": "4px"}),
            # Progress bar
            html.Div([
                html.Div(style={
                    "width": f"{bar_width:.1f}%", "height": "6px",
                    "backgroundColor": bar_color, "borderRadius": "3px",
                    "transition": "width 0.4s ease, background-color 0.4s ease",
                }),
            ], style={
                "width": "100%", "height": "6px", "backgroundColor": "#2a2a3f",
                "borderRadius": "3px", "marginBottom": "6px",
            }),
            html.Div(description, style={"color": _MUTED, "fontSize": "11px",
                                         "fontStyle": "italic"}),
        ], style={
            "backgroundColor": _PANEL, "borderRadius": "8px",
            "padding": "10px 14px", "border": "1px solid #2e2e44",
        })
