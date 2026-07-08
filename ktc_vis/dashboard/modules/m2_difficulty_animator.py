"""Module 2: Difficulty Animator. Owner: Asmita Bhuva."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.io
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from ktc_vis.dashboard.theme import (
    ACCENT, BG, BORDER, CARD, CARD_ALT, CARD_STYLE, DANGER, MUTED, SUCCESS, SURFACE, TEXT, WARN,
)

# electrode and injection counts per difficulty level (from KTC2023 paper)
_LEVEL_META = {
    1: {"electrodes": 32, "injections": 76, "measurements": 2356},
    2: {"electrodes": 30, "injections": 52, "measurements": 1624},
    3: {"electrodes": 28, "injections": 52, "measurements": 1404},
    4: {"electrodes": 26, "injections": 46, "measurements": 1200},
    5: {"electrodes": 24, "injections": 44, "measurements": 1012},
    6: {"electrodes": 22, "injections": 30, "measurements": 630},
    7: {"electrodes": 20, "injections": 27, "measurements": 513},
}

# file paths and lookup tables
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


def layout() -> html.Div:
    """Builds the full page layout: header, controls, image panels, metric curves, and the live commentary block."""
    return html.Div([

        html.Div([
            html.Div([
                html.Span("M2", style={
                    "display": "inline-block", "padding": "2px 10px",
                    "borderRadius": "999px", "backgroundColor": ACCENT,
                    "color": "#fff", "fontSize": "11px", "fontWeight": 600,
                    "letterSpacing": "0.5px", "marginRight": "10px",
                }),
                html.Span("Difficulty Animator", style={
                    "color": TEXT, "fontSize": "20px", "fontWeight": 600,
                }),
            ]),
            html.P(
                "KTC2023 has seven difficulty levels created by progressively removing electrodes "
                "(from 32 down to 20) and cutting the voltage measurements from 2356 to 513. "
                "Use the slider or hit ▶ Play to step through the levels and see how reconstruction "
                "quality holds up as data gets sparser. "
                "The curves below track each metric across all seven levels so you can tell whether "
                "an algorithm degrades smoothly or falls apart at a particular point.",
                style={"color": MUTED, "margin": "8px 0 0", "fontSize": "13px",
                       "lineHeight": "1.6"},
            ),
        ], style={
            "backgroundColor": CARD, "borderRadius": "10px",
            "padding": "16px 20px", "marginBottom": "16px",
            "borderLeft": f"3px solid {ACCENT}",
        }),

        html.Div(
            id="m2-chips",
            children=[_chip("status", "loading…")],
            style={"display": "flex", "flexWrap": "wrap", "gap": "8px",
                   "marginBottom": "16px"},
        ),

        html.Div(id="m2-level-context", style={"marginBottom": "14px"}),

        html.Div([
            html.Div([
                html.Div([
                    html.Span("Difficulty Level", style={
                        "color": TEXT, "fontSize": "12px", "fontWeight": 600,
                        "marginRight": "10px",
                    }),
                    html.Span(
                        "Drag to jump to any level, or press ▶ Play to animate levels 1 → 7 → 1 in a loop.",
                        style={"color": MUTED, "fontSize": "11px"},
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
                            style=_btn_style(ACCENT)),
                html.Div("1.2 s / level", style={
                    "color": MUTED, "fontSize": "10px", "textAlign": "center",
                    "marginTop": "4px",
                }),
            ], style={"paddingTop": "24px", "textAlign": "center"}),
        ], style={
            "display": "flex", "alignItems": "flex-start",
            "marginBottom": "20px", "backgroundColor": CARD,
            "padding": "16px 20px", "borderRadius": "8px",
        }),

        dcc.Interval(id="m2-interval", interval=1200, n_intervals=0, disabled=True),
        dcc.Store(id="m2-playing", data=False),

        _section_label(
            "Visual Comparison",
            "Side-by-side ground truth and reconstruction at the selected level. The error overlay shows exactly where the algorithm got it wrong.",
        ),

        html.Div([
            _image_card("Ground Truth", "m2-gt-graph",
                        "The real phantom used for this sample, fixed regardless of difficulty level."),
            _image_card("Segmentation", "m2-recon-graph",
                        "The algorithm's segmentation output for this level and sample."),
            _image_card("Error Overlay", "m2-error-graph",
                        "Where it got it right and wrong. Green = correct, red = false positive, orange = wrong class, blue = missed."),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "20px"}),

        _section_label(
            "Degradation Curves",
            "How each metric changes as measurement data is progressively removed. The yellow marker tracks whichever level you have selected.",
        ),

        html.Div([
            _curve_card("SSIM (higher is better, range 0–1)",
                        "m2-ssim-graph",
                        "Measures how structurally similar the reconstruction is to the ground truth "
                        "by looking at luminance, contrast, and local patterns. "
                        "This is the official KTC2023 ranking metric, so it's the one that matters most for comparison."),
            _curve_card("Mean IoU (higher is better, range 0–1)",
                        "m2-iou-graph",
                        "Intersection-over-Union averaged over all three tissue classes (water, resistive, conductive). "
                        "It penalises missing regions and false positives equally, so you can't cheat it by over-segmenting."),
            _curve_card("Dice Score (higher is better, range 0–1)",
                        "m2-dice-graph",
                        "Computed from IoU as 2*IoU / (1+IoU), which is equivalent to the F1 score for segmentation. "
                        "Tends to be more sensitive to small inclusions than raw IoU."),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        html.Div([
            _curve_card("Hausdorff Distance in pixels (lower is better)",
                        "m2-hausdorff-graph",
                        "The worst-case gap between predicted and true inclusion boundaries. "
                        "A single badly placed pixel can spike this number, so it's a good stress test for algorithms "
                        "that occasionally smear or drop inclusions."),
            _curve_card("Position Error in pixels (lower is better)",
                        "m2-poserr-graph",
                        "How far the predicted inclusion centre is from the true centre, in pixels. "
                        "Tells you whether the algorithm finds inclusions in roughly the right place, even if the shape is off."),
            _curve_card("Resolution in pixels (smaller = finer detail)",
                        "m2-resolution-graph",
                        "Smallest inclusion diameter the algorithm manages to resolve. "
                        "Goes up as measurement data gets sparser and fine structures start to blur together."),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        html.Div([
            _curve_card("Runtime (lower is faster, log scale per algorithm)",
                        "m2-runtime-graph",
                        "How long it actually takes to produce one reconstruction, shown on a log scale so differences "
                        "are visible within each algorithm. Typical ballparks: ABC1 around 22 s, PNPE2E around 42 s, "
                        "CUQI8 roughly 100 minutes. CUQI8 numbers are estimated from a 48-hour batch run, see runtime_log.md for details."),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "20px"}),

        _section_label(
            "Level Analysis",
            "A plain-language read of what the numbers actually mean at this level.",
        ),

        html.Div(
            id="m2-commentary",
            style={
                "backgroundColor": CARD, "borderRadius": "8px",
                "padding": "16px 20px", "borderLeft": f"3px solid {ACCENT}",
                "minHeight": "60px",
            },
        ),

    ], style={"padding": "20px", "backgroundColor": BG, "minHeight": "100vh"})


# small badge-style chip showing a label and value pair, used in the status row
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
        style={**CARD_STYLE, "padding": "8px 12px",
               "display": "inline-flex", "alignItems": "center"},
    )


# renders a section heading with a dimmer subtitle underneath, used to break up the page
def _section_label(title: str, subtitle: str) -> html.Div:
    return html.Div([
        html.Span(title, style={
            "color": TEXT, "fontWeight": 600, "fontSize": "12px",
            "letterSpacing": "0.8px", "textTransform": "uppercase",
            "marginRight": "10px",
        }),
        html.Span(subtitle, style={"color": MUTED, "fontSize": "12px"}),
    ], style={
        "display": "flex", "alignItems": "baseline",
        "padding": "4px 2px 0", "marginBottom": "10px",
        "borderBottom": f"1px solid {BORDER}", "paddingBottom": "6px",
    })


# returns the CSS dict for the play/pause button so the style stays in one place
def _btn_style(color: str) -> dict:
    return {
        "backgroundColor": color, "color": "#fff", "border": "none",
        "padding": "8px 20px", "borderRadius": "6px", "cursor": "pointer",
        "fontSize": "14px", "fontWeight": "bold", "minWidth": "100px",
    }


# wraps a Plotly graph in a titled card for the ground truth / segmentation / error panels
def _image_card(title: str, graph_id: str, subtitle: str) -> html.Div:
    return html.Div([
        html.Div(title, style={"color": TEXT, "fontWeight": "bold",
                               "fontSize": "13px", "marginBottom": "2px"}),
        html.Div(subtitle, style={"color": MUTED, "fontSize": "10px",
                                  "marginBottom": "6px"}),
        dcc.Graph(id=graph_id, config={"displayModeBar": False},
                  style={"height": "260px"}),
    ], style={"flex": "1", "backgroundColor": CARD_ALT, "borderRadius": "8px",
              "padding": "12px"})


# wraps a small degradation curve graph with a title and optional description text
def _curve_card(title: str, graph_id: str, description: str = "") -> html.Div:
    children = [
        html.Div(title, style={"color": TEXT, "fontWeight": "bold",
                               "fontSize": "12px", "marginBottom": "2px"}),
    ]
    if description:
        children.append(html.Div(description, style={
            "color": MUTED, "fontSize": "10px", "lineHeight": "1.4",
            "marginBottom": "6px",
        }))
    else:
        children.append(html.Div(style={"marginBottom": "6px"}))
    children.append(
        dcc.Graph(id=graph_id, config={"displayModeBar": False},
                  style={"height": "160px"}),
    )
    return html.Div(children, style={
        "flex": "1", "backgroundColor": CARD_ALT, "borderRadius": "8px", "padding": "12px",
    })


def _gt_figure(level: int, sample: str) -> go.Figure:
    """Reads the ground truth .mat file for the given sample and returns a 3-class heatmap."""
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
    fig.update_layout(**_image_layout(f"Level {level}, Sample {sample.upper()}"))
    return fig


# thin alias kept so callers don't need to know about _empty_figure directly
def _placeholder_figure(message: str) -> go.Figure:
    return _empty_figure(message)


# returns a blank figure with a centred message, used whenever data isn't available yet
def _empty_figure(message: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=CARD_ALT, plot_bgcolor=CARD_ALT,
        xaxis={"visible": False}, yaxis={"visible": False},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        annotations=[{"text": message, "showarrow": False,
                      "font": {"color": MUTED, "size": 12},
                      "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}],
    )
    return fig


# shared Plotly layout dict for the three image panels so they all look the same
def _image_layout(title: str) -> dict:
    return {
        "paper_bgcolor": CARD_ALT, "plot_bgcolor": CARD_ALT,
        "margin": {"l": 0, "r": 0, "t": 24, "b": 0},
        "title": {"text": title, "font": {"color": MUTED, "size": 11}, "x": 0.5,
                  "pad": {"t": 4}},
        "xaxis": {"visible": False, "scaleanchor": "y"},
        "yaxis": {"visible": False, "autorange": "reversed"},
        "height": 260,
        "uirevision": "constant",
    }


# anything below this is a cached placeholder, not real reconstruction time
_RUNTIME_REAL_THRESHOLD_S = 0.01

# fixed y-axis ranges so the charts stay comparable when switching algorithms
_FIXED_Y_RANGE: dict[str, tuple[float, float]] = {
    "ssim": (0.0, 1.0),
    "iou_mean": (0.0, 1.0),
    "dice": (0.0, 1.0),
}

# log-scale metrics; None = auto-range per algorithm (runtime spans very different
# orders of magnitude across algorithms, so a fixed range would squash most curves)
_LOG_Y_RANGE: dict[str, tuple[float, float] | None] = {
    "runtime": None,
}


def _curve_figure(metric: str, algorithm: str, sample: str,
                  current_level: int = 1) -> go.Figure:
    """Loads metric values from the HDF5 cache and plots them as a line across levels 1-7. Falls back to a placeholder if the cache is missing or all-zero."""
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
            line={"color": ACCENT, "width": 2},
            marker={"size": 7, "color": ACCENT},
            name=algorithm.upper(),
        ))
        # Highlight current level with a vertical dashed line
        fig.add_vline(
            x=current_level,
            line={"color": "#ffb300", "width": 1.5, "dash": "dash"},
        )
    else:
        msg = ("Requires live benchmark. Run scripts/run_benchmark.py"
               if metric == "runtime"
               else "Run scripts/run_benchmark.py to populate")
        annotation = [{"text": msg, "showarrow": False,
                       "font": {"color": MUTED, "size": 10},
                       "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}]

    yaxis_cfg = {
        "tickfont": {"color": MUTED, "size": 9},
        "gridcolor": CARD_ALT,
        "zeroline": False,
    }
    if metric in _LOG_Y_RANGE:
        yaxis_cfg["type"] = "log"
        rng = _LOG_Y_RANGE[metric]
        if rng is None:
            yaxis_cfg["autorange"] = True
        else:
            lo, hi = rng
            yaxis_cfg["range"] = [lo, hi]
            yaxis_cfg["autorange"] = False
    elif metric in _FIXED_Y_RANGE:
        lo, hi = _FIXED_Y_RANGE[metric]
        yaxis_cfg["range"] = [lo, hi]
        yaxis_cfg["autorange"] = False

    fig.update_layout(
        paper_bgcolor=CARD_ALT, plot_bgcolor=SURFACE,
        margin={"l": 36, "r": 8, "t": 8, "b": 36},
        xaxis={
            "range": [0.5, 7.5],
            "tickvals": LEVELS,
            "ticktext": [str(lv) for lv in LEVELS],
            "title": {"text": "Level", "font": {"color": MUTED, "size": 10}},
            "tickfont": {"color": MUTED, "size": 9},
            "gridcolor": CARD_ALT,
            "zeroline": False,
        },
        yaxis=yaxis_cfg,
        annotations=annotation,
        showlegend=False,
        height=160,
    )
    return fig


def _dice_curve_figure(algorithm: str, sample: str, current_level: int = 1) -> go.Figure:
    """Computes Dice from cached IoU values (2*IoU / (1+IoU)) and hands off to _curve_figure_from_values."""
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
    """Same as _curve_figure but takes pre-computed values instead of a metric key, useful for derived metrics like Dice."""
    fig = go.Figure()
    annotation = []
    if values is not None:
        fig.add_trace(go.Scatter(
            x=LEVELS, y=values, mode="lines+markers",
            line={"color": ACCENT, "width": 2},
            marker={"size": 7, "color": ACCENT},
            name=algorithm.upper(),
        ))
        fig.add_vline(x=current_level,
                      line={"color": "#ffb300", "width": 1.5, "dash": "dash"})
    else:
        annotation = [{"text": "Run scripts/run_benchmark.py to populate",
                       "showarrow": False, "font": {"color": MUTED, "size": 10},
                       "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}]

    yaxis_cfg = {"tickfont": {"color": MUTED, "size": 9},
                 "gridcolor": CARD_ALT, "zeroline": False}
    if metric and metric in _FIXED_Y_RANGE:
        lo, hi = _FIXED_Y_RANGE[metric]
        yaxis_cfg["range"] = [lo, hi]
        yaxis_cfg["autorange"] = False

    fig.update_layout(
        paper_bgcolor=CARD_ALT, plot_bgcolor=SURFACE,
        margin={"l": 36, "r": 8, "t": 8, "b": 36},
        xaxis={
            "range": [0.5, 7.5], "tickvals": LEVELS,
            "ticktext": [str(lv) for lv in LEVELS],
            "title": {"text": "Level", "font": {"color": MUTED, "size": 10}},
            "tickfont": {"color": MUTED, "size": 9},
            "gridcolor": CARD_ALT, "zeroline": False,
        },
        yaxis=yaxis_cfg,
        annotations=annotation, showlegend=False, height=160,
        uirevision="constant",
    )
    return fig


def _recon_and_error_figures(
    level: int, sample: str, algorithm: str
) -> tuple[go.Figure, go.Figure]:
    """Loads the cached reconstruction for the given algorithm/level/sample and builds both the segmentation heatmap and the pixel-level error overlay."""
    try:
        from ktc_vis.cache.hdf5_store import load_result
        _, recon = load_result(algorithm, level, sample, cache_path=_CACHE_PATH)
    except Exception:
        msg = f"No cache for {algorithm.upper()} L{level}/{sample.upper()}, run benchmark"
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
    """Tries to read a metric's values for all 7 levels from the HDF5 cache. Returns a list of 7 floats, or None if anything is missing."""
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


# coloured pill-shaped badge used inside the commentary panel to label metric quality
def _badge(text: str, color: str) -> html.Span:
    return html.Span(text, style={
        "backgroundColor": color, "color": "#fff", "borderRadius": "4px",
        "padding": "2px 8px", "fontSize": "11px", "fontWeight": "bold",
        "marginRight": "6px",
    })


# maps an SSIM value to a human-readable quality label and its display colour
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


# maps a Hausdorff distance in pixels to a boundary quality label and colour
def _hausdorff_label(v: float) -> tuple[str, str]:
    if v <= 8:
        return "Tight boundaries", "#2e7d32"
    if v <= 15:
        return "Acceptable boundaries", "#558b2f"
    if v <= 25:
        return "Blurred boundaries", "#f57f17"
    return "Severe boundary error", "#c62828"


# maps a mean IoU value to an overlap quality label and colour
def _iou_label(v: float) -> tuple[str, str]:
    if v >= 0.75:
        return "High overlap", "#2e7d32"
    if v >= 0.55:
        return "Moderate overlap", "#558b2f"
    if v >= 0.35:
        return "Low overlap", "#f57f17"
    return "Poor segmentation", "#c62828"


# maps a position error in pixels to a centroid accuracy label and colour
def _poserr_label(v: float) -> tuple[str, str]:
    if v <= 10:
        return "Centroid on-target", "#2e7d32"
    if v <= 25:
        return "Centroid offset", "#f57f17"
    return "Centroid mislocated", "#c62828"


# maps a resolution value in pixels to a detail-preservation label and colour
def _resolution_label(v: float) -> tuple[str, str]:
    if v <= 20:
        return "Fine detail preserved", "#2e7d32"
    if v <= 40:
        return "Moderate blurring", "#558b2f"
    if v <= 60:
        return "Coarse only", "#f57f17"
    return "Resolution collapse", "#c62828"


# formats a one-line trend sentence comparing the current metric value to the previous level
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


# assembles the full Level Analysis panel: hardware context, per-metric badges, and the summary sentence
def _build_commentary(level: int, algorithm: str, sample: str) -> list:
    meta = _LEVEL_META[level]
    prev_meta = _LEVEL_META.get(level - 1)
    data_ratio = meta["measurements"] / _LEVEL_META[1]["measurements"]

    context_parts = [
        _badge(f"Level {level}", ACCENT),
        html.Span(
            f"{meta['electrodes']} electrodes, {meta['injections']} injections, "
            f"{meta['measurements']} measurements "
            f"({data_ratio * 100:.0f}% of Level 1 data)",
            style={"color": TEXT, "fontSize": "13px"},
        ),
    ]
    if prev_meta:
        elec_lost = prev_meta["electrodes"] - meta["electrodes"]
        meas_lost = prev_meta["measurements"] - meta["measurements"]
        context_parts.append(html.Span(
            f"  ({elec_lost} fewer electrode(s), {meas_lost} fewer measurements than Level {level - 1})",
            style={"color": MUTED, "fontSize": "12px"},
        ))

    rows = [
        html.Div(context_parts, style={"marginBottom": "10px"}),
    ]

    ssim_vals = _try_load_from_cache("ssim", algorithm, sample)
    iou_vals = _try_load_from_cache("iou_mean", algorithm, sample)
    hd_vals = _try_load_from_cache("hausdorff", algorithm, sample)
    pe_vals = _try_load_from_cache("position_error", algorithm, sample)
    res_vals = _try_load_from_cache("resolution", algorithm, sample)

    if ssim_vals is None and iou_vals is None and hd_vals is None:
        rows.append(html.Div(
            "Benchmark cache not yet available for this combination. "
            "Run scripts/run_benchmark.py to see metric commentary.",
            style={"color": MUTED, "fontSize": "12px", "fontStyle": "italic"},
        ))
        return rows

    metric_items = []
    idx = level - 1  # 0-based index into values list
    prev_idx = idx - 1

    if ssim_vals and idx < len(ssim_vals):
        v = ssim_vals[idx]
        label, color = _ssim_label(v)
        line = [_badge(label, color), html.Span(f"SSIM = {v:.3f}", style={"color": TEXT, "fontSize": "12px"})]
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
        line = [_badge(label, color), html.Span(f"Mean IoU = {v:.3f}", style={"color": TEXT, "fontSize": "12px"})]
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
        line = [_badge(label, color), html.Span(f"Hausdorff = {v:.1f} px", style={"color": TEXT, "fontSize": "12px"})]
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
        line = [_badge(label, color), html.Span(f"Position error = {v:.1f} px", style={"color": TEXT, "fontSize": "12px"})]
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
        line = [_badge(label, color), html.Span(f"Resolution = {v:.0f} px", style={"color": TEXT, "fontSize": "12px"})]
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

    summary = _summary_sentence(level, algorithm, ssim_vals, iou_vals, hd_vals)
    if summary:
        rows.append(html.Div(summary, style={
            "color": MUTED, "fontSize": "12px", "fontStyle": "italic",
            "borderTop": f"1px solid {CARD_ALT}", "paddingTop": "8px",
        }))

    return rows


# writes a plain English sentence describing overall reconstruction quality at the given level
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
        base = f"{alg} performs excellently at Level {level}. With {_LEVEL_META[level]['measurements']} measurements, the EIT inverse problem is still well-constrained."
    elif ssim >= 0.70:
        base = f"{alg} maintains reasonable quality at Level {level}, though the reduced electrode count ({_LEVEL_META[level]['electrodes']}) is starting to limit spatial resolution."
    elif ssim >= 0.55:
        base = f"At Level {level}, {alg} shows visible degradation. Fewer measurement paths ({_LEVEL_META[level]['measurements']}) mean the inverse problem is becoming under-determined."
    elif ssim >= 0.40:
        base = f"Level {level} is challenging for {alg}. Only {_LEVEL_META[level]['electrodes']} electrodes leaves large angular gaps, causing significant reconstruction artifacts."
    else:
        base = f"At Level {level}, {alg} is operating near its limit. With only {_LEVEL_META[level]['measurements']} measurements ({_LEVEL_META[level]['measurements'] / _LEVEL_META[1]['measurements'] * 100:.0f}% of Level 1), reliable reconstruction is very difficult."

    return base + cliff_str


def register_callbacks(app) -> None:  # noqa: ANN001
    """Register all M2 callbacks."""

    @app.callback(
        Output("m2-interval", "disabled"),
        Output("m2-playing", "data"),
        Output("m2-play-btn", "children"),
        Input("m2-play-btn", "n_clicks"),
        State("m2-playing", "data"),
        prevent_initial_call=True,
    )
    # flips between play and pause, toggling the interval timer and updating the button label
    def toggle_play(n_clicks, is_playing):
        playing = not is_playing
        label = "⏸  Pause" if playing else "▶  Play"
        return not playing, playing, label

    @app.callback(
        Output("m2-level-slider", "value"),
        Input("sidebar-level-slider", "value"),
        Input("m2-interval", "n_intervals"),
        State("m2-level-slider", "value"),
        State("m2-playing", "data"),
        prevent_initial_call=True,
    )
    # syncs the level slider with the sidebar, or steps it forward by one during auto-play
    def advance_level(sidebar_level, n_intervals, current_level, is_playing):
        from dash import callback_context
        trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
        if "sidebar-level-slider" in trigger:
            return sidebar_level
        if not is_playing:
            raise PreventUpdate
        return (current_level % 7) + 1  # cycle 1→2→…→7→1

    @app.callback(
        Output("m2-gt-graph", "figure"),
        Output("m2-recon-graph", "figure"),
        Output("m2-error-graph", "figure"),
        Input("m2-level-slider", "value"),
        Input("sidebar-sample-radio", "value"),
        Input("sidebar-algorithm-dropdown", "value"),
    )
    # redraws the ground truth, reconstruction, and error overlay whenever level, sample, or algorithm changes
    def update_images(level, sample, algorithm):
        gt_fig = _gt_figure(level, sample)
        recon_fig, error_fig = _recon_and_error_figures(level, sample, algorithm)
        return gt_fig, recon_fig, error_fig

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
    # redraws all seven metric curves whenever algorithm, sample, or level changes
    def update_curves(algorithm, sample, level):
        ssim_fig = _curve_figure("ssim", algorithm, sample, level)
        iou_fig = _curve_figure("iou_mean", algorithm, sample, level)
        dice_fig = _dice_curve_figure(algorithm, sample, level)
        hausdorff_fig = _curve_figure("hausdorff", algorithm, sample, level)
        poserr_fig = _curve_figure("position_error", algorithm, sample, level)
        resolution_fig = _curve_figure("resolution", algorithm, sample, level)
        runtime_fig = _curve_figure("runtime", algorithm, sample, level)
        return ssim_fig, iou_fig, dice_fig, hausdorff_fig, poserr_fig, resolution_fig, runtime_fig

    @app.callback(
        Output("m2-commentary", "children"),
        Input("m2-level-slider", "value"),
        Input("sidebar-algorithm-dropdown", "value"),
        Input("sidebar-sample-radio", "value"),
    )
    # refreshes the Level Analysis commentary panel with metric badges and the summary sentence
    def update_commentary(level, algorithm, sample):
        return _build_commentary(level, algorithm, sample)

    @app.callback(
        Output("m2-chips", "children"),
        Input("m2-level-slider", "value"),
        Input("sidebar-algorithm-dropdown", "value"),
        Input("sidebar-sample-radio", "value"),
    )
    # refreshes the chip row at the top with live stats for the selected combination
    def update_chips(level, algorithm, sample):
        meta = _LEVEL_META[level]
        chips = [
            _chip("algorithm", algorithm.upper(), accent="#cfe0ff"),
            _chip("level", f"L{level}", accent=WARN),
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
                accent=SUCCESS if agreement >= 90 else WARN,
            ))

            pos_err = metrics.get("position_error")
            if pos_err is not None:
                accent = SUCCESS if pos_err <= 10 else (WARN if pos_err <= 25 else DANGER)
                chips.append(_chip("position err", f"{pos_err:.1f} px", accent=accent))

            resolution = metrics.get("resolution")
            if resolution is not None:
                accent = SUCCESS if resolution <= 30 else (WARN if resolution <= 60 else DANGER)
                chips.append(_chip("resolution", f"{resolution:.0f} px", accent=accent))

            runtime = metrics.get("runtime")
            if runtime is not None and runtime >= _RUNTIME_REAL_THRESHOLD_S:
                if runtime >= 60:
                    m_, s_ = divmod(int(runtime), 60)
                    rt_label = f"{m_}m {s_:02d}s" if m_ < 60 else f"{m_//60}h {m_%60:02d}m"
                else:
                    rt_label = f"{runtime:.1f} s"
                chips.append(_chip("runtime", rt_label))
        except Exception:
            chips.append(_chip("pixel agreement", "n/a", accent=MUTED))

        return chips

    @app.callback(
        Output("m2-level-context", "children"),
        Input("m2-level-slider", "value"),
    )
    # redraws the data availability bar and description text below the chips
    def update_level_context(level):
        meta = _LEVEL_META[level]
        l1 = _LEVEL_META[1]
        meas_pct = meta["measurements"] / l1["measurements"] * 100
        elec_lost = l1["electrodes"] - meta["electrodes"]
        meas_lost = l1["measurements"] - meta["measurements"]

        if level == 1:
            description = "Full measurement protocol with maximum data. Best expected reconstruction quality."
            bar_color = SUCCESS
        elif level <= 3:
            description = f"{elec_lost} electrodes removed, {meas_lost:,} fewer measurements than Level 1. Mild data reduction and most algorithms still perform well."
            bar_color = "#4caf50"
        elif level <= 5:
            description = f"{elec_lost} electrodes removed, {meas_lost:,} fewer measurements. Moderate data loss and the inverse problem starts becoming under-determined. Quality begins to drop."
            bar_color = WARN
        else:
            description = f"{elec_lost} electrodes removed, {meas_lost:,} fewer measurements. Severe data loss with only {meas_pct:.0f}% of Level 1 data remaining. Most algorithms degrade significantly."
            bar_color = DANGER

        bar_width = meas_pct

        return html.Div([
            html.Div([
                html.Span(f"Level {level} data availability: {meas_pct:.0f}% of Level 1",
                          style={"color": TEXT, "fontSize": "12px", "fontWeight": 600}),
                html.Span(f"  {meta['electrodes']} electrodes, {meta['measurements']:,} measurements",
                          style={"color": MUTED, "fontSize": "11px"}),
            ], style={"marginBottom": "6px", "display": "flex", "alignItems": "baseline", "gap": "4px"}),
            # Progress bar
            html.Div([
                html.Div(style={
                    "width": f"{bar_width:.1f}%", "height": "6px",
                    "backgroundColor": bar_color, "borderRadius": "3px",
                    "transition": "width 0.4s ease, background-color 0.4s ease",
                }),
            ], style={
                "width": "100%", "height": "6px", "backgroundColor": CARD_ALT,
                "borderRadius": "3px", "marginBottom": "6px",
            }),
            html.Div(description, style={"color": MUTED, "fontSize": "11px",
                                         "fontStyle": "italic"}),
        ], style={
            "backgroundColor": CARD, "borderRadius": "8px",
            "padding": "10px 14px", "border": f"1px solid {BORDER}",
        })
