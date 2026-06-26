"""Module 5: Failure Autopsy.

Ranks the worst-performing ``(algorithm, level, sample)`` cases by SSIM and,
for the case selected via the sidebar, exposes four diagnostic panels:

    1. Spatial SSIM heatmap        — which pixels drove the score down
    2. 3×3 confusion matrix        — class-level misclassification pattern
    3. Boundary error polar plot   — angular distribution of prediction errors
    4. Measurement residual polar  — per-injection inclusion energy

A failure-type badge (A–E) is auto-assigned from the diagnostic signals.
Clicking a row in the ranked list snaps the shared sidebar to that case.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.io
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from ktc_vis.cache.hdf5_store import CacheMiss, load_result
from ktc_vis.dashboard.components.glossary import (
    glossary_term,
    info_pill,
    with_tooltip,
)
from ktc_vis.data.loader import KTCDataLoader
from ktc_vis.data.subsampler import subsample_electrodes
from ktc_vis.dashboard.theme import (
    ACCENT,
    ACCENT_SOFT,
    BORDER,
    CARD,
    CARD_ALT,
    CARD_STYLE,
    DANGER,
    MUTED,
    SECTION_LABEL_STYLE,
    SUCCESS,
    TEXT,
    TEXT_DIM,
    WARN,
)
from ktc_vis.metrics.class_metrics import compute_confusion_matrix
from ktc_vis.metrics.image_quality import compute_spatial_ssim_map
from ktc_vis.utils.figures import empty_figure

logger = logging.getLogger(__name__)

# ── Paths & registry ─────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CACHE_PATH = _PROJECT_ROOT / "data" / "cache" / "results.h5"
_RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "ktc2023"
_REF_MAT = _RAW_DIR / "measurements" / "ref.mat"
_SAMPLE_INDEX = {"a": 1, "b": 2, "c": 3, "d": 4}

_ALGORITHMS = ["abc1", "cuqi8", "pnpe2e"]
_SAMPLES = ["a", "b", "c", "d"]
_LEVELS = list(range(1, 8))

_LABEL = {"abc1": "ABC1", "cuqi8": "CUQI8", "pnpe2e": "PNPE2E"}
_ALG_COLOR = {"abc1": "#5b8def", "cuqi8": "#e85d75", "pnpe2e": "#f4c870"}
_CLASS_LABEL = ["water", "resistive", "conductive"]

# ── Failure taxonomy (per IMPLEMENTATION_GUIDE.md §3.3 & docs) ───────────────
_FAILURE_TYPES = {
    "A": {
        "name": "Ghost inclusion",
        "color": "#a07bff",
        "summary": "Algorithm predicts an inclusion where the ground truth is water.",
        "signal": "FP-heavy confusion (predicted non-water over true water).",
        "owner": "ABC1 CNN — electrode artifact mistaken for a real inclusion.",
    },
    "B": {
        "name": "Missing inclusion",
        "color": "#e85d75",
        "summary": "Algorithm fails to detect an inclusion that is present.",
        "signal": "FN-heavy confusion (predicted water over true inclusion).",
        "owner": "CUQI8 level-set converges to the wrong number of regions.",
    },
    "C": {
        "name": "Class flip",
        "color": "#f4c870",
        "summary": "Location is correct but conductivity sign is wrong.",
        "signal": "Resistive ↔ conductive swap concentrated on the diagonal off-axis.",
        "owner": "PNPE2E gets the inclusion right but inverts the polarity.",
    },
    "D": {
        "name": "Boundary erosion",
        "color": "#5b8def",
        "summary": "Inclusion is detected but eroded / blurred at the edges.",
        "signal": "Reasonable IoU but the predicted footprint is materially smaller than GT.",
        "owner": "ABC1 smoothness prior blurs the inclusion boundary.",
    },
    "E": {
        "name": "Mask suppression",
        "color": "#2ecc71",
        "summary": "End-to-end stage masks out an inclusion that the proxy recovered.",
        "signal": "PNPE2E + strong FN with a balanced confusion in the inclusion classes.",
        "owner": "PNPE2E E2E refinement collapses the inclusion mask.",
    },
}

# ── Component IDs ────────────────────────────────────────────────────────────
_RANKING_ID = "m5-ranking"
_DETAIL_CHIPS_ID = "m5-detail-chips"
_BADGE_ID = "m5-failure-badge"
_EXPLAIN_ID = "m5-failure-explanation"
_THUMBS_ID = "m5-thumbs"
_QUICK_METRICS_ID = "m5-quick-metrics"
_SIGNALS_ID = "m5-signal-breakdown"
_SPATIAL_SSIM_ID = "m5-spatial-ssim"
_CONFUSION_ID = "m5-confusion"
_BOUNDARY_ID = "m5-boundary"
_MEAS_ID = "m5-measurement-residual"
_SPATIAL_FOOTER_ID = "m5-spatial-footer"
_CONFUSION_FOOTER_ID = "m5-confusion-footer"
_BOUNDARY_FOOTER_ID = "m5-boundary-footer"
_MEAS_FOOTER_ID = "m5-meas-footer"
_FILTER_ID = "m5-alg-filter"
_COUNT_ID = "m5-count-slider"
_BANNER_ID = "m5-banner"
_DATA_STORE_ID = "m5-ranking-store"

# Pattern-matching row IDs: {"type": "m5-row", "key": "alg|level|sample"}.
_ROW_TYPE = "m5-row"


# ── Module-level singletons ──────────────────────────────────────────────────
_LOADER = KTCDataLoader()


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def layout() -> html.Div:
    """Top-level Failure Autopsy layout."""
    return html.Div(
        [
            _header(),
            html.Div(id=_BANNER_ID),
            dcc.Store(id=_DATA_STORE_ID),
            _top_row(),
            _reading_guide_card(),
            _diagnostic_grid(),
            _failure_legend(),
        ],
        style={
            "padding": "24px 28px",
            "minHeight": "100%",
            "display": "flex",
            "flexDirection": "column",
            "gap": "16px",
        },
    )


def register_callbacks(app) -> None:  # noqa: ANN001
    """Wire ranking, sidebar navigation, and diagnostic panels."""

    # ── Build (or rebuild) the ranking list when filters change ──────────────
    @app.callback(
        Output(_RANKING_ID, "children"),
        Output(_DATA_STORE_ID, "data"),
        Output(_BANNER_ID, "children"),
        Input(_FILTER_ID, "value"),
        Input(_COUNT_ID, "value"),
    )
    def _refresh_ranking(alg_filter: str, count: int):
        cases = _load_all_cases()
        if not cases:
            empty = html.Div(
                "No cached results found. Run scripts/run_benchmark.py first.",
                style={"color": MUTED, "padding": "24px", "textAlign": "center",
                       "fontStyle": "italic", "fontSize": "13px"},
            )
            return empty, [], _banner(
                "Cache is empty — diagnostic panels are unavailable.", "error",
            )

        if alg_filter and alg_filter != "all":
            cases = [c for c in cases if c["algorithm"] == alg_filter]

        cases.sort(key=lambda c: c["ssim"])
        cases = cases[: int(count)]
        rows = [_ranking_row(c, i + 1) for i, c in enumerate(cases)]
        return rows, cases, None

    # ── Row click → snap the shared sidebar to that case ─────────────────────
    @app.callback(
        Output("sidebar-algorithm-dropdown", "value", allow_duplicate=True),
        Output("sidebar-level-slider", "value", allow_duplicate=True),
        Output("sidebar-sample-radio", "value", allow_duplicate=True),
        Input({"type": _ROW_TYPE, "key": ALL}, "n_clicks"),
        State({"type": _ROW_TYPE, "key": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _navigate_from_row(n_clicks_list, id_list):
        if not n_clicks_list or not any(n_clicks_list):
            return no_update, no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or "key" not in triggered:
            return no_update, no_update, no_update
        alg, level, sample = triggered["key"].split("|")
        return alg, int(level), sample

    # ── Sidebar selection → refresh the 4 diagnostic panels ──────────────────
    @app.callback(
        Output(_DETAIL_CHIPS_ID, "children"),
        Output(_BADGE_ID, "children"),
        Output(_EXPLAIN_ID, "children"),
        Output(_THUMBS_ID, "children"),
        Output(_QUICK_METRICS_ID, "children"),
        Output(_SIGNALS_ID, "children"),
        Output(_SPATIAL_SSIM_ID, "figure"),
        Output(_CONFUSION_ID, "figure"),
        Output(_BOUNDARY_ID, "figure"),
        Output(_MEAS_ID, "figure"),
        Output(_SPATIAL_FOOTER_ID, "children"),
        Output(_CONFUSION_FOOTER_ID, "children"),
        Output(_BOUNDARY_FOOTER_ID, "children"),
        Output(_MEAS_FOOTER_ID, "children"),
        Input("sidebar-algorithm-dropdown", "value"),
        Input("sidebar-level-slider", "value"),
        Input("sidebar-sample-radio", "value"),
    )
    def _refresh_diagnostics(algorithm: str, level: int, sample: str):
        level = int(level)
        try:
            metrics, reconstruction = load_result(
                algorithm, level, sample, _CACHE_PATH,
            )
        except (CacheMiss, OSError) as exc:
            logger.warning("M5 cache miss: %s", exc)
            chips = _detail_chips(algorithm, level, sample, ssim=None)
            badge = _badge(None)
            explanation = _explanation(None, algorithm)
            empty = empty_figure("No cached result for this case")
            placeholder = _interpretation_placeholder("No data for this case.")
            return (chips, badge, explanation,
                    _thumbnails_placeholder("No cached result"),
                    _quick_metrics_placeholder("No metrics available"),
                    _signal_breakdown_placeholder("No classifier signals"),
                    empty, empty, empty, empty,
                    placeholder, placeholder, placeholder, placeholder)

        try:
            gt = _load_ground_truth(sample)
        except FileNotFoundError as exc:
            logger.warning("M5 ground truth missing: %s", exc)
            chips = _detail_chips(algorithm, level, sample, metrics.get("ssim"))
            badge = _badge(None)
            explanation = _explanation(None, algorithm)
            empty = empty_figure("Ground truth missing")
            placeholder = _interpretation_placeholder("Ground truth missing.")
            return (chips, badge, explanation,
                    _thumbnails_placeholder("Ground truth missing"),
                    _quick_metrics_view(metrics),
                    _signal_breakdown_placeholder("Classifier needs GT"),
                    empty, empty, empty, empty,
                    placeholder, placeholder, placeholder, placeholder)

        pred = np.asarray(reconstruction, dtype=np.uint8)
        cm = compute_confusion_matrix(pred, gt)
        ssim_map = compute_spatial_ssim_map(pred, gt)
        failure_code, signal_scores = _classify_failure(pred, gt, cm, algorithm)

        chips = _detail_chips(algorithm, level, sample, metrics.get("ssim"),
                              failure_code=failure_code)
        badge = _badge(failure_code)
        explanation = _explanation(failure_code, algorithm)
        thumbs = _thumbnails_view(gt, pred)
        quick = _quick_metrics_view(metrics)
        signals = _signal_breakdown_view(signal_scores, failure_code)

        spatial_fig = _spatial_ssim_figure(ssim_map)
        confusion_fig = _confusion_figure(cm)
        boundary_fig = _boundary_radial_figure(pred, gt, level)
        meas_fig = _measurement_residual_figure(level, sample, pred, gt)

        spatial_footer = _spatial_interpretation(ssim_map, metrics.get("ssim"))
        confusion_footer = _confusion_interpretation(cm)
        boundary_footer = _boundary_interpretation(pred, gt, level)
        try:
            measurement = _LOADER.load(level=level, sample=sample)
            per_inj = _per_injection_perturbation(measurement)
        except Exception:  # noqa: BLE001
            per_inj = None
        meas_footer = _measurement_interpretation(per_inj)

        return (chips, badge, explanation, thumbs, quick, signals,
                spatial_fig, confusion_fig, boundary_fig, meas_fig,
                spatial_footer, confusion_footer, boundary_footer, meas_footer)


# ═══════════════════════════════════════════════════════════════════════════════
# Layout building blocks
# ═══════════════════════════════════════════════════════════════════════════════

def _header() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "M5",
                        style={
                            "display": "inline-block",
                            "padding": "2px 10px",
                            "borderRadius": "999px",
                            "backgroundColor": DANGER,
                            "color": "#fff",
                            "fontSize": "11px",
                            "fontWeight": 600,
                            "letterSpacing": "0.5px",
                            "marginRight": "10px",
                        },
                    ),
                    html.Span(
                        "Failure Autopsy",
                        style={"color": TEXT, "fontSize": "20px", "fontWeight": 600},
                    ),
                ]
            ),
            html.P(
                "Ranks the worst-scoring cases across all algorithms and levels. "
                "Click a row to inspect spatial SSIM, the 3×3 confusion matrix, "
                "the angular distribution of boundary errors, and per-injection "
                "measurement perturbation energy. A failure-type badge (A–E) is "
                "assigned automatically from these signals.",
                style={"color": MUTED, "margin": "8px 0 0", "fontSize": "13px",
                       "lineHeight": "1.5"},
            ),
        ],
        style={**CARD_STYLE, "padding": "16px 20px",
               "borderLeft": f"3px solid {DANGER}"},
    )


def _top_row() -> html.Div:
    return html.Div(
        [
            _ranking_panel(),
            _selected_panel(),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "minmax(360px, 1fr) minmax(420px, 1.2fr)",
            "gap": "16px",
            "alignItems": "stretch",
        },
    )


def _ranking_panel() -> html.Div:
    header = html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "▼",
                        style={
                            "display": "inline-block",
                            "padding": "2px 8px",
                            "borderRadius": "6px",
                            "backgroundColor": DANGER,
                            "color": "#fff",
                            "fontSize": "10px",
                            "fontWeight": 700,
                            "marginRight": "10px",
                        },
                    ),
                    html.Span("Worst Cases (ascending SSIM)",
                              style={"color": TEXT, "fontWeight": 600,
                                     "fontSize": "13.5px"}),
                ]
            ),
            html.Span("click a row to inspect",
                      style={"color": MUTED, "fontSize": "11.5px"}),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "10px 14px",
            "borderBottom": f"1px solid {BORDER}",
        },
    )

    controls = html.Div(
        [
            html.Div(
                [
                    html.Div("Filter algorithm", style=SECTION_LABEL_STYLE),
                    dcc.RadioItems(
                        id=_FILTER_ID,
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "ABC1", "value": "abc1"},
                            {"label": "CUQI8", "value": "cuqi8"},
                            {"label": "PNPE2E", "value": "pnpe2e"},
                        ],
                        value="all",
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={
                            "color": TEXT_DIM, "marginRight": "10px",
                            "fontSize": "12px", "cursor": "pointer",
                        },
                    ),
                ],
                style={"flex": "1"},
            ),
            html.Div(
                [
                    html.Div("Show top", style=SECTION_LABEL_STYLE),
                    dcc.Slider(
                        id=_COUNT_ID, min=5, max=25, step=5, value=10,
                        marks={i: {"label": str(i),
                                   "style": {"color": MUTED, "fontSize": "10px"}}
                               for i in (5, 10, 15, 20, 25)},
                    ),
                ],
                style={"flex": "1.4", "minWidth": "160px"},
            ),
        ],
        style={
            "display": "flex",
            "gap": "16px",
            "padding": "12px 14px",
            "borderBottom": f"1px solid {BORDER}",
        },
    )

    body = html.Div(
        id=_RANKING_ID,
        children=[
            html.Div("Loading…", style={"color": MUTED, "padding": "24px",
                                        "textAlign": "center", "fontSize": "12px"}),
        ],
        style={"padding": "8px 8px 12px", "overflowY": "auto", "flex": "1"},
    )

    return html.Div(
        [header, controls, body],
        style={
            **CARD_STYLE,
            "display": "flex",
            "flexDirection": "column",
            "maxHeight": "560px",
        },
    )


def _selected_panel() -> html.Div:
    header = html.Div(
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
                    html.Span("Selected Case",
                              style={"color": TEXT, "fontWeight": 600,
                                     "fontSize": "13.5px"}),
                ]
            ),
            html.Span("set via sidebar or row click",
                      style={"color": MUTED, "fontSize": "11.5px"}),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "10px 14px",
            "borderBottom": f"1px solid {BORDER}",
        },
    )

    chips = html.Div(
        id=_DETAIL_CHIPS_ID,
        children=[],
        style={
            "display": "flex", "flexWrap": "wrap", "gap": "8px",
            "padding": "12px 14px 8px",
        },
    )

    body = html.Div(
        [
            html.Div(
                id=_BADGE_ID,
                children=_badge(None),
            ),
            html.Div(
                id=_EXPLAIN_ID,
                children=_explanation(None, None),
                style={"padding": "10px 14px 4px"},
            ),
            html.Div(
                id=_THUMBS_ID,
                children=_thumbnails_placeholder(),
                style={"padding": "4px 14px 6px"},
            ),
            html.Div(
                id=_QUICK_METRICS_ID,
                children=_quick_metrics_placeholder(),
                style={"padding": "4px 14px 6px"},
            ),
            html.Div(
                id=_SIGNALS_ID,
                children=_signal_breakdown_placeholder(),
                style={"padding": "4px 14px 14px"},
            ),
        ],
        style={"display": "flex", "flexDirection": "column", "gap": "4px",
               "overflowY": "auto"},
    )

    return html.Div(
        [header, chips, body],
        style={
            **CARD_STYLE,
            "display": "flex",
            "flexDirection": "column",
            "maxHeight": "560px",
        },
    )


def _diagnostic_grid() -> html.Div:
    panels = [
        _graph_panel(
            "Spatial SSIM", "where the pixels lose the score",
            _SPATIAL_SSIM_ID, badge="SSIM", badge_color=ACCENT_SOFT,
            footer_id=_SPATIAL_FOOTER_ID,
            help_text=(
                "Per-pixel SSIM map. Green = pixels that match the ground "
                "truth, red = pixels that hurt the overall score the most. "
                "Use it to localise WHERE the failure happened on the image."
            ),
        ),
        _graph_panel(
            "Confusion Matrix", "rows = ground truth · cols = prediction",
            _CONFUSION_ID, badge="CM", badge_color="#a07bff",
            footer_id=_CONFUSION_FOOTER_ID,
            help_text=(
                "Rows: true class. Columns: what the algorithm predicted. "
                "The diagonal is correct; off-diagonal cells tell you WHAT "
                "kind of mistake was made (ghost, missing, or class flip)."
            ),
        ),
        _graph_panel(
            "Boundary Error (polar)",
            "errors by angle · green = active electrode · gray × = removed",
            _BOUNDARY_ID, badge="∂", badge_color=DANGER,
            footer_id=_BOUNDARY_FOOTER_ID,
            help_text=(
                "Angular histogram of wrongly classified pixels around the "
                "tank centre. Tall bars in the gray × arc = blame the data; "
                "errors spread around the full ring = blame the algorithm."
            ),
        ),
        _graph_panel(
            "Measurement Perturbation (polar)",
            "blue = strong signal · red = weakest 25% · yellow dashed = threshold",
            _MEAS_ID, badge="V", badge_color=SUCCESS,
            footer_id=_MEAS_FOOTER_ID,
            help_text=(
                "For each current injection, how much the inclusion changed "
                "the boundary voltages versus an empty tank. Red bars carry "
                "almost no inclusion information — no algorithm can recover "
                "what was never measured."
            ),
        ),
    ]
    return html.Div(
        panels,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(2, 1fr)",
            "gap": "16px",
        },
    )


def _graph_panel(
    title: str,
    subtitle: str,
    graph_id: str,
    *,
    badge: str = "",
    badge_color: str = ACCENT,
    height: int = 360,
    footer_id: str | None = None,
    help_text: str | None = None,
) -> html.Div:
    title_block = [
        html.Span(
            badge,
            style={
                "display": "inline-block",
                "padding": "2px 8px",
                "borderRadius": "6px",
                "backgroundColor": badge_color,
                "color": "#fff" if badge_color != WARN else "#111",
                "fontSize": "10px",
                "fontWeight": 700,
                "marginRight": "10px",
            },
        ),
        html.Span(title, style={"color": TEXT, "fontWeight": 600,
                                "fontSize": "13.5px"}),
    ]
    if help_text:
        # Native browser tooltip via title attr — no extra callbacks needed.
        title_block.append(
            html.Span(
                "?",
                title=help_text,
                style={
                    "display": "inline-flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "width": "16px", "height": "16px",
                    "borderRadius": "999px",
                    "backgroundColor": BORDER,
                    "color": TEXT_DIM,
                    "fontSize": "10px",
                    "fontWeight": 700,
                    "marginLeft": "8px",
                    "cursor": "help",
                },
            )
        )

    children = [
        html.Div(
            [
                html.Div(title_block),
                html.Span(subtitle, style={"color": MUTED, "fontSize": "11.5px"}),
            ],
            style={
                "display": "flex", "alignItems": "center",
                "justifyContent": "space-between",
                "padding": "10px 14px",
                "borderBottom": f"1px solid {BORDER}",
            },
        ),
        html.Div(
            dcc.Graph(
                id=graph_id,
                figure=empty_figure("Loading…"),
                config={"displayModeBar": False, "responsive": True},
                style={"height": f"{height}px", "width": "100%"},
            ),
            style={"padding": "6px 6px 10px"},
        ),
    ]
    if footer_id:
        children.append(
            html.Div(
                id=footer_id,
                children=_interpretation_placeholder(),
                style={
                    "padding": "10px 14px 14px",
                    "borderTop": f"1px dashed {BORDER}",
                    "backgroundColor": CARD_ALT,
                },
            )
        )

    return html.Div(
        children,
        style={
            **CARD_STYLE,
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden",
        },
    )


def _failure_legend() -> html.Div:
    cells = []
    for code, meta in _FAILURE_TYPES.items():
        cells.append(html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            code,
                            style={
                                "display": "inline-block",
                                "padding": "2px 10px",
                                "borderRadius": "999px",
                                "backgroundColor": meta["color"],
                                "color": "#111" if meta["color"] == WARN else "#fff",
                                "fontWeight": 700, "fontSize": "12px",
                                "marginRight": "10px",
                            },
                        ),
                        html.Span(meta["name"], style={"color": TEXT,
                                                       "fontWeight": 600,
                                                       "fontSize": "13px"}),
                    ],
                    style={"marginBottom": "6px"},
                ),
                html.Div(meta["summary"], style={"color": TEXT_DIM,
                                                 "fontSize": "11.5px",
                                                 "lineHeight": "1.5"}),
                html.Div(meta["owner"], style={"color": MUTED,
                                               "fontSize": "11px",
                                               "marginTop": "6px",
                                               "fontStyle": "italic"}),
            ],
            style={
                **CARD_STYLE,
                "padding": "12px 14px",
                "borderTop": f"3px solid {meta['color']}",
            },
        ))
    return html.Div(
        [
            html.Div(
                [
                    html.Span("Failure Taxonomy", style={
                        "color": TEXT, "fontWeight": 600, "fontSize": "12px",
                        "letterSpacing": "0.8px", "textTransform": "uppercase",
                        "marginRight": "10px",
                    }),
                    html.Span("how the auto-assigned badge is read",
                              style={"color": MUTED, "fontSize": "12px"}),
                ],
                style={
                    "display": "flex", "alignItems": "baseline",
                    "borderBottom": f"1px dashed {BORDER}",
                    "paddingBottom": "6px", "marginBottom": "8px",
                },
            ),
            html.Div(
                cells,
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "12px",
                },
            ),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Reading guide & per-graph interpretation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _reading_guide_card() -> html.Div:
    """Static workflow strip explaining what each of the 4 graphs answers."""
    steps = [
        ("1", "Spatial SSIM", "WHERE on the image did the failure happen?",
         ACCENT_SOFT),
        ("2", "Confusion Matrix", "WHAT kind of mistake was made?",
         "#a07bff"),
        ("3", "Boundary (polar)", "Was the data sparse around that angle?",
         DANGER),
        ("4", "Measurement (polar)", "Did the injections even carry signal?",
         SUCCESS),
    ]
    cells = []
    for num, name, question, color in steps:
        cells.append(html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            num,
                            style={
                                "display": "inline-flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "width": "22px", "height": "22px",
                                "borderRadius": "999px",
                                "backgroundColor": color,
                                "color": "#111" if color == WARN else "#fff",
                                "fontSize": "11px", "fontWeight": 700,
                                "marginRight": "10px",
                            },
                        ),
                        html.Span(name, style={"color": TEXT,
                                               "fontWeight": 600,
                                               "fontSize": "12.5px"}),
                    ],
                    style={"display": "flex", "alignItems": "center",
                           "marginBottom": "6px"},
                ),
                html.Div(question, style={"color": TEXT_DIM,
                                          "fontSize": "11.5px",
                                          "lineHeight": "1.45"}),
            ],
            style={
                "padding": "10px 14px",
                "borderLeft": f"3px solid {color}",
                "backgroundColor": CARD_ALT,
                "borderRadius": "6px",
            },
        ))
    return html.Div(
        [
            html.Div(
                [
                    html.Span("How to Read This Module", style={
                        "color": TEXT, "fontWeight": 600, "fontSize": "12px",
                        "letterSpacing": "0.8px", "textTransform": "uppercase",
                        "marginRight": "10px",
                    }),
                    html.Span("read the 4 panels left-to-right, top-to-bottom",
                              style={"color": MUTED, "fontSize": "12px"}),
                ],
                style={
                    "display": "flex", "alignItems": "baseline",
                    "borderBottom": f"1px dashed {BORDER}",
                    "paddingBottom": "6px", "marginBottom": "10px",
                },
            ),
            html.Div(
                cells,
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "10px",
                },
            ),
        ],
        style={**CARD_STYLE, "padding": "14px 16px"},
    )


def _interpretation_placeholder(message: str = "Select a case to see the reading.") -> html.Div:
    return html.Div(
        message,
        style={"color": MUTED, "fontSize": "11.5px", "fontStyle": "italic",
               "lineHeight": "1.5"},
    )


def _interpretation_block(label: str, body: str, *, accent: str = ACCENT) -> html.Div:
    """Compact 'reading' footer with a coloured label and one-sentence body."""
    return html.Div(
        [
            html.Span(
                label,
                style={
                    "display": "inline-block",
                    "padding": "1px 7px",
                    "borderRadius": "4px",
                    "backgroundColor": accent,
                    "color": "#111" if accent == WARN else "#fff",
                    "fontSize": "9.5px", "fontWeight": 700,
                    "letterSpacing": "0.6px",
                    "textTransform": "uppercase",
                    "marginRight": "8px",
                    "verticalAlign": "middle",
                },
            ),
            html.Span(body, style={"color": TEXT_DIM, "fontSize": "11.5px",
                                   "lineHeight": "1.55"}),
        ],
    )


def _spatial_interpretation(ssim_map: np.ndarray, ssim_score: float | None) -> html.Div:
    """Plain-English reading of the spatial SSIM heatmap."""
    arr = np.asarray(ssim_map, dtype=float)
    if arr.size == 0:
        return _interpretation_placeholder("Spatial SSIM map is empty.")

    bad_mask = arr < 0
    bad_pct = 100.0 * bad_mask.sum() / arr.size

    if bad_pct < 0.5:
        verdict = (f"Almost the whole image agrees with ground truth "
                   f"({bad_pct:.1f}% red pixels). Errors are negligible.")
        accent = SUCCESS
    else:
        # Centroid of red region (image coords: row 0 at top).
        rows, cols = np.where(bad_mask)
        if rows.size > 0:
            cy = rows.mean() / arr.shape[0]   # 0..1, 0=top
            cx = cols.mean() / arr.shape[1]   # 0..1, 0=left
            vert = "top" if cy < 0.4 else ("bottom" if cy > 0.6 else "middle")
            horiz = "left" if cx < 0.4 else ("right" if cx > 0.6 else "centre")
            location = (f"{vert}-{horiz}" if vert != "middle" or horiz != "centre"
                        else "centre of the tank")
        else:
            location = "the tank"

        if bad_pct < 3:
            verdict = (f"{bad_pct:.1f}% of pixels disagree with GT, "
                       f"localised near the {location}. Small, focused error.")
            accent = SUCCESS
        elif bad_pct < 12:
            verdict = (f"{bad_pct:.1f}% of pixels disagree with GT, "
                       f"clustered near the {location}. Moderate, focused error.")
            accent = WARN
        else:
            verdict = (f"{bad_pct:.1f}% of pixels disagree with GT, "
                       f"largest red region near the {location}. "
                       f"Wide-area failure.")
            accent = DANGER

    if ssim_score is not None:
        verdict += f"  Overall SSIM = {float(ssim_score):.3f}."
    return _interpretation_block("Reading", verdict, accent=accent)


def _confusion_interpretation(cm: np.ndarray) -> html.Div:
    """Identify the dominant off-diagonal cell and name the failure mode."""
    arr = np.asarray(cm, dtype=float)
    if arr.shape != (3, 3):
        return _interpretation_placeholder("Confusion matrix has unexpected shape.")

    labels = ["water", "resistive", "conductive"]
    diag = np.diag(arr).copy()
    off = arr.copy()
    np.fill_diagonal(off, 0.0)

    if off.max() <= 0:
        return _interpretation_block(
            "Reading",
            "Diagonal-only — every pixel is in the correct class. "
            f"Per-class accuracy: water {diag[0]*100:.0f}%, "
            f"resistive {diag[1]*100:.0f}%, conductive {diag[2]*100:.0f}%.",
            accent=SUCCESS,
        )

    i, j = np.unravel_index(int(np.argmax(off)), off.shape)
    cell = float(off[i, j])

    # Tag the failure family of the dominant off-diagonal.
    if labels[i] == "water":
        family = "ghost — pixels invented where there is nothing"
        accent = DANGER
    elif labels[j] == "water":
        family = "missing — pixels erased where an inclusion exists"
        accent = WARN
    else:
        family = "class flip — resistive vs conductive swapped"
        accent = "#a07bff"

    body = (f"Biggest single error: {cell*100:.1f}% of {labels[i]} pixels were "
            f"predicted as {labels[j]} → {family}.")
    if cell >= 0.20:
        body += "  This is a dominant failure mode for this case."
    return _interpretation_block("Reading", body, accent=accent)


def _boundary_interpretation(
    pred: np.ndarray, gt: np.ndarray, level: int,
) -> html.Div:
    """Compare error-angle distribution against the removed-electrode arc."""
    pred_arr = np.asarray(pred)
    gt_arr = np.asarray(gt)
    err_mask = pred_arr != gt_arr
    total_err = int(err_mask.sum())
    if total_err == 0:
        return _interpretation_block(
            "Reading",
            "No misclassified pixels — there is no boundary error to localise.",
            accent=SUCCESS,
        )

    h, w = err_mask.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rows, cols = np.where(err_mask)
    # Image coords → math angle in degrees [0, 360), 0° = +x, CCW.
    angles = np.degrees(np.arctan2(-(rows - cy), cols - cx)) % 360.0

    # Removed-electrode arc: full ring has 32 electrodes; level keeps the
    # first N evenly-spaced ones, so the removed arc is the high-angle slice.
    n_total = 32
    n_active = max(1, n_total - 2 * (level - 1))
    # Active electrodes occupy angles 0..(n_active/n_total * 360).
    active_end_deg = (n_active / n_total) * 360.0
    in_removed = (angles >= active_end_deg) & (angles < 360.0)
    removed_pct = 100.0 * in_removed.sum() / total_err if total_err else 0.0
    expected_pct = max(0.0, 100.0 * (n_total - n_active) / n_total)

    # Peak bin (36 bins of 10°).
    hist, edges = np.histogram(angles, bins=36, range=(0, 360))
    peak_bin = int(np.argmax(hist))
    peak_deg = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0
    peak_share = 100.0 * hist[peak_bin] / total_err

    if level <= 1:
        body = (f"All {n_total} electrodes active. Error peaks around "
                f"{peak_deg:.0f}° ({peak_share:.0f}% of all error pixels). "
                f"No data-side blind arc — this is algorithm-side.")
        accent = "#a07bff"
    elif removed_pct >= 1.5 * expected_pct and removed_pct >= 25:
        body = (f"{removed_pct:.0f}% of errors fall in the removed-electrode "
                f"arc (≈{expected_pct:.0f}% expected by area). "
                f"Data-side limitation dominates this failure.")
        accent = DANGER
    elif removed_pct <= expected_pct * 0.6:
        body = (f"Only {removed_pct:.0f}% of errors land in the removed "
                f"arc ({expected_pct:.0f}% expected). Error peaks at "
                f"{peak_deg:.0f}° — algorithm-side, not data-sparsity.")
        accent = "#a07bff"
    else:
        body = (f"Errors spread fairly evenly around the ring; "
                f"{removed_pct:.0f}% in the removed arc vs "
                f"{expected_pct:.0f}% expected. Mixed cause.")
        accent = WARN
    return _interpretation_block("Reading", body, accent=accent)


def _measurement_interpretation(per_inj: np.ndarray | None) -> html.Div:
    """Describe how much of the measurement set is starved of inclusion signal."""
    if per_inj is None:
        return _interpretation_placeholder("Could not load raw measurements.")

    arr = np.asarray(per_inj, dtype=float)
    if arr.size == 0:
        return _interpretation_placeholder("No per-injection values available.")

    threshold = float(np.percentile(arr, 25))
    weak = arr < threshold
    weak_pct = 100.0 * weak.sum() / arr.size

    rng = float(arr.max() - arr.min())
    med = float(np.median(arr))
    contrast = (rng / med) if med > 1e-9 else 0.0

    # Angular position of weak injections (assume evenly spread around 360°).
    n = arr.size
    weak_idx = np.where(weak)[0]
    if weak_idx.size:
        weak_angles = (weak_idx / n) * 360.0
        ang_mean = float(weak_angles.mean())
        ang_spread = float(weak_angles.max() - weak_angles.min())
        location = (f"clustered near {ang_mean:.0f}°"
                    if ang_spread < 90 else "spread around the ring")
    else:
        location = "n/a"

    if contrast < 0.15:
        body = (f"Signal is nearly flat ({contrast*100:.0f}% peak-to-median "
                f"contrast). Most injections carry little inclusion info; "
                f"algorithms have very thin data to work with.")
        accent = DANGER
    elif weak_pct >= 30:
        body = (f"{weak_pct:.0f}% of injections are below the weak-signal "
                f"threshold ({location}). Reconstruction near those angles "
                f"is data-limited.")
        accent = WARN
    else:
        body = (f"Healthy signal — only {weak_pct:.0f}% of injections fall "
                f"below the P25 weak line ({location}). Algorithm had usable "
                f"data; any failure is algorithm-side.")
        accent = SUCCESS
    return _interpretation_block("Reading", body, accent=accent)


# ═══════════════════════════════════════════════════════════════════════════════
# Row rendering & chips
# ═══════════════════════════════════════════════════════════════════════════════

def _ranking_row(case: dict, rank: int) -> html.Button:
    alg = case["algorithm"]
    color = _ALG_COLOR[alg]
    ssim = float(case["ssim"])
    bar_pct = max(2.0, min(100.0, ssim * 100.0))
    badge = case.get("failure", "?")
    badge_color = _FAILURE_TYPES.get(badge, {}).get("color", MUTED)

    return html.Button(
        [
            html.Div(
                f"#{rank}",
                style={
                    "color": MUTED, "fontSize": "11px", "fontVariantNumeric": "tabular-nums",
                    "width": "32px", "fontWeight": 600,
                },
            ),
            html.Div(
                [
                    html.Span(_LABEL[alg], style={
                        "color": color, "fontWeight": 700, "fontSize": "12.5px",
                    }),
                    html.Span(
                        f"  L{case['level']} · {case['sample'].upper()}",
                        style={"color": TEXT_DIM, "fontSize": "11.5px",
                               "marginLeft": "6px"},
                    ),
                ],
                style={"flex": "1.6", "textAlign": "left"},
            ),
            html.Div(
                [
                    html.Div(
                        html.Div(style={
                            "height": "100%", "width": f"{bar_pct:.0f}%",
                            "backgroundColor": color, "borderRadius": "2px",
                            "opacity": "0.55",
                        }),
                        style={
                            "height": "4px", "backgroundColor": BORDER,
                            "borderRadius": "2px", "marginBottom": "4px",
                        },
                    ),
                    html.Span(
                        f"{ssim:.3f}",
                        style={
                            "color": TEXT, "fontWeight": 600, "fontSize": "12px",
                            "fontVariantNumeric": "tabular-nums",
                        },
                    ),
                ],
                style={"flex": "2", "minWidth": "80px"},
            ),
            html.Span(
                badge,
                style={
                    "display": "inline-block",
                    "minWidth": "22px",
                    "padding": "2px 8px",
                    "borderRadius": "999px",
                    "backgroundColor": badge_color,
                    "color": "#111" if badge_color == WARN else "#fff",
                    "fontWeight": 700,
                    "fontSize": "11px",
                    "textAlign": "center",
                },
            ),
        ],
        id={"type": _ROW_TYPE, "key": f"{alg}|{case['level']}|{case['sample']}"},
        n_clicks=0,
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "10px",
            "width": "100%",
            "padding": "8px 12px",
            "margin": "4px 4px",
            "backgroundColor": CARD_ALT,
            "border": f"1px solid {BORDER}",
            "borderRadius": "8px",
            "cursor": "pointer",
            "textAlign": "left",
            "color": TEXT,
        },
        n_clicks_timestamp=0,
    )


def _chip(label: str, value: str, accent: str | None = None) -> html.Div:
    return html.Div(
        [
            html.Span(
                glossary_term(label, label.upper()),
                style={
                    "color": MUTED, "fontSize": "10.5px", "letterSpacing": "0.6px",
                    "textTransform": "uppercase", "marginRight": "8px",
                },
            ),
            html.Span(
                value,
                style={
                    "color": accent or TEXT, "fontSize": "13px",
                    "fontWeight": 600, "fontVariantNumeric": "tabular-nums",
                },
            ),
        ],
        style={
            **CARD_STYLE,
            "padding": "6px 10px",
            "display": "inline-flex",
            "alignItems": "center",
        },
    )


def _detail_chips(
    algorithm: str,
    level: int,
    sample: str,
    ssim: float | None,
    *,
    failure_code: str | None = None,
) -> list:
    chips = [
        _chip("algorithm", _LABEL.get(algorithm, algorithm.upper()),
              accent=_ALG_COLOR.get(algorithm, TEXT)),
        _chip("level", f"L{level}", accent=WARN),
        _chip("sample", sample.upper()),
    ]
    if ssim is None:
        chips.append(_chip("ssim", "—", accent=DANGER))
    else:
        col = SUCCESS if ssim >= 0.7 else (WARN if ssim >= 0.4 else DANGER)
        chips.append(_chip("ssim", f"{ssim:.3f}", accent=col))
    if failure_code:
        meta = _FAILURE_TYPES[failure_code]
        chips.append(_chip("failure type",
                           f"{failure_code} · {meta['name']}",
                           accent=meta["color"]))
    return chips


def _badge(code: str | None) -> html.Div:
    if not code:
        meta = {"name": "—", "color": MUTED, "summary": "Select a case to classify."}
        code_display = "?"
    else:
        meta = _FAILURE_TYPES[code]
        code_display = code

    return html.Div(
        [
            with_tooltip(
                html.Span(
                    code_display,
                    style={
                        "display": "inline-flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "width": "42px", "height": "42px",
                        "borderRadius": "12px",
                        "backgroundColor": meta["color"],
                        "color": "#111" if meta["color"] == WARN else "#fff",
                        "fontWeight": 800, "fontSize": "20px",
                        "marginRight": "12px",
                    },
                ),
                code_display,
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Failure Type "),
                            glossary_term(code_display, code_display),
                            html.Span(f" · {meta['name']}"),
                            info_pill("failure type"),
                        ],
                        style={"color": TEXT, "fontWeight": 700, "fontSize": "14px"},
                    ),
                    html.Div(
                        meta["summary"],
                        style={"color": MUTED, "fontSize": "12px",
                               "marginTop": "2px"},
                    ),
                ],
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "padding": "12px 14px",
            "backgroundColor": CARD_ALT,
            "borderTop": f"1px solid {BORDER}",
            "borderBottom": f"1px solid {BORDER}",
        },
    )


def _explanation(code: str | None, algorithm: str | None) -> html.Div:
    if not code:
        text = ("Pick a case from the ranked list or change the sidebar selectors "
                "to assign a failure type and view the diagnostic panels.")
        return html.Div(text, style={"color": MUTED, "fontSize": "12.5px",
                                     "lineHeight": "1.6"})
    meta = _FAILURE_TYPES[code]
    alg_note = (
        f"Matches the typical failure of {_LABEL.get(algorithm, '—')}."
        if algorithm and meta["owner"].startswith(_LABEL.get(algorithm, "##"))
        else f"Auto-assigned from diagnostic signals — typical owner: "
             f"{meta['owner'].split(' — ')[0]}."
    )
    return html.Div(
        [
            html.Div(
                meta["signal"],
                style={"color": TEXT_DIM, "fontSize": "12.5px",
                       "lineHeight": "1.6", "marginBottom": "6px"},
            ),
            html.Div(alg_note, style={"color": MUTED, "fontSize": "11.5px",
                                      "fontStyle": "italic"}),
        ],
    )


# ── Selected-Case extras: thumbnails · quick metrics · signal breakdown ──────

def _section_caption(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "color": MUTED, "fontSize": "10px", "letterSpacing": "0.8px",
            "textTransform": "uppercase", "fontWeight": 600,
            "marginBottom": "6px",
        },
    )


def _thumbnails_placeholder(msg: str = "—") -> html.Div:
    return html.Div(
        [
            _section_caption("Ground Truth vs Reconstruction"),
            html.Div(msg, style={"color": MUTED, "fontSize": "11.5px",
                                 "fontStyle": "italic"}),
        ],
    )


def _thumbnails_view(gt: np.ndarray, pred: np.ndarray) -> html.Div:
    return html.Div(
        [
            _section_caption("Ground Truth vs Reconstruction"),
            html.Div(
                [
                    _thumb("GT", gt, SUCCESS),
                    _thumb("Pred", pred, "#5b8def"),
                ],
                style={"display": "grid",
                       "gridTemplateColumns": "1fr 1fr",
                       "gap": "8px"},
            ),
        ],
    )


def _thumb(label: str, seg: np.ndarray, accent: str) -> dcc.Graph:
    fig = go.Figure(
        go.Heatmap(
            z=seg,
            colorscale=[
                [0.0, "#14142a"],
                [0.33, "#14142a"],
                [0.34, "#5b8def"],
                [0.66, "#5b8def"],
                [0.67, "#e85d75"],
                [1.0, "#e85d75"],
            ],
            zmin=0, zmax=2, showscale=False,
            hovertemplate=f"{label}: %{{z}}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor=CARD_ALT, plot_bgcolor="#14142a",
        margin=dict(l=2, r=2, t=18, b=2),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, autorange="reversed"),
        title=dict(text=label, x=0.5, y=0.97,
                   font=dict(color=accent, size=11, family="monospace")),
        uirevision=f"m5-thumb-{label}",
    )
    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": False, "responsive": True},
        style={"height": "110px", "width": "100%"},
    )


def _quick_metrics_placeholder(msg: str = "—") -> html.Div:
    return html.Div(
        [
            _section_caption("Quick Metrics (from cache)"),
            html.Div(msg, style={"color": MUTED, "fontSize": "11.5px",
                                 "fontStyle": "italic"}),
        ],
    )


def _quick_metrics_view(metrics: dict) -> html.Div:
    """3×2 grid of the most actionable cached numbers."""
    items = [
        ("IoU water", metrics.get("iou_water"), "{:.2f}", None),
        ("IoU resistive", metrics.get("iou_resistive"), "{:.2f}", None),
        ("IoU conductive", metrics.get("iou_conductive"), "{:.2f}", None),
        ("Hausdorff", metrics.get("hausdorff"), "{:.1f} px", "lower better"),
        ("Position error", metrics.get("position_error"), "{:.1f} px", "lower better"),
        ("Runtime", metrics.get("runtime"), "{:.2f} s", "lower better"),
    ]

    def _fmt(value, tmpl):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "—"
        try:
            return tmpl.format(float(value))
        except (TypeError, ValueError):
            return "—"

    cells = []
    for label, value, tmpl, hint in items:
        cells.append(html.Div(
            [
                html.Div(
                    glossary_term(label, label),
                    style={"color": MUTED, "fontSize": "10.5px",
                           "letterSpacing": "0.3px"},
                ),
                html.Div(
                    _fmt(value, tmpl),
                    style={"color": TEXT, "fontSize": "14px", "fontWeight": 700,
                           "fontVariantNumeric": "tabular-nums",
                           "marginTop": "2px"},
                ),
                html.Div(hint or "", style={"color": MUTED, "fontSize": "9.5px",
                                            "fontStyle": "italic"}),
            ],
            style={
                "backgroundColor": CARD_ALT,
                "border": f"1px solid {BORDER}",
                "borderRadius": "8px",
                "padding": "8px 10px",
            },
        ))

    return html.Div(
        [
            _section_caption("Quick Metrics (from cache)"),
            html.Div(cells, style={"display": "grid",
                                   "gridTemplateColumns": "repeat(3, 1fr)",
                                   "gap": "6px"}),
        ],
    )


def _signal_breakdown_placeholder(msg: str = "—") -> html.Div:
    return html.Div(
        [
            _section_caption("Why this badge? Classifier signal scores"),
            html.Div(msg, style={"color": MUTED, "fontSize": "11.5px",
                                 "fontStyle": "italic"}),
        ],
    )


def _signal_breakdown_view(
    signal_scores: dict[str, float],
    winning_code: str,
) -> html.Div:
    """Mini horizontal bars for the per-code scores from ``_classify_failure``."""
    if not signal_scores:
        return _signal_breakdown_placeholder("No signals available")

    max_score = max((v for v in signal_scores.values() if v > 0), default=1.0)
    rows = []
    for code in ("A", "B", "C", "D", "E"):
        meta = _FAILURE_TYPES[code]
        raw = float(signal_scores.get(code, 0.0))
        pct = max(2.0, min(100.0, (raw / max_score) * 100.0)) if max_score > 0 else 2.0
        is_win = code == winning_code
        rows.append(html.Div(
            [
                html.Span(
                    glossary_term(code, code),
                    style={
                        "display": "inline-block",
                        "width": "22px",
                        "padding": "1px 0",
                        "borderRadius": "999px",
                        "backgroundColor": meta["color"] if is_win else BORDER,
                        "color": ("#111" if is_win and meta["color"] == WARN
                                  else ("#fff" if is_win else MUTED)),
                        "fontWeight": 700, "fontSize": "10px",
                        "textAlign": "center",
                        "marginRight": "8px",
                    },
                ),
                html.Span(
                    meta["name"],
                    style={"color": TEXT if is_win else TEXT_DIM,
                           "fontSize": "11.5px",
                           "fontWeight": 600 if is_win else 400,
                           "minWidth": "120px",
                           "display": "inline-block"},
                ),
                html.Div(
                    html.Div(style={
                        "height": "100%",
                        "width": f"{pct:.0f}%",
                        "backgroundColor": meta["color"],
                        "borderRadius": "3px",
                        "opacity": "0.85" if is_win else "0.35",
                    }),
                    style={
                        "height": "6px",
                        "backgroundColor": BORDER,
                        "borderRadius": "3px",
                        "flex": "1",
                        "margin": "0 8px",
                    },
                ),
                html.Span(
                    f"{raw:.2f}",
                    style={"color": TEXT if is_win else MUTED,
                           "fontSize": "10.5px",
                           "fontVariantNumeric": "tabular-nums",
                           "minWidth": "36px", "textAlign": "right"},
                ),
            ],
            style={"display": "flex", "alignItems": "center",
                   "marginBottom": "5px"},
        ))

    return html.Div(
        [
            _section_caption("Why this badge? Classifier signal scores"),
            html.Div(rows),
            html.Div(
                "Bars compare each failure type's evidence; the winning code "
                "determines the badge.",
                style={"color": MUTED, "fontSize": "10.5px",
                       "fontStyle": "italic", "marginTop": "6px"},
            ),
        ],
    )


def _banner(text: str, kind: str = "info") -> html.Div:
    palette = {
        "info": (ACCENT_SOFT, "rgba(91,141,239,0.08)"),
        "warn": (WARN, "rgba(244,200,112,0.10)"),
        "error": (DANGER, "rgba(232,93,117,0.10)"),
    }
    border, bg = palette.get(kind, palette["info"])
    return html.Div(
        text,
        style={
            "border": f"1px solid {border}",
            "backgroundColor": bg,
            "color": TEXT_DIM,
            "padding": "10px 14px",
            "borderRadius": "8px",
            "fontSize": "12.5px",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_all_cases(cache_path: Path = _CACHE_PATH) -> list[dict]:
    """Return one record per cached (alg, level, sample) with SSIM and badge."""
    records: list[dict] = []
    for alg in _ALGORITHMS:
        for lv in _LEVELS:
            for s in _SAMPLES:
                try:
                    metrics, recon = load_result(alg, lv, s, cache_path)
                except (CacheMiss, OSError):
                    continue
                ssim = float(metrics.get("ssim", float("nan")))
                if np.isnan(ssim):
                    continue
                try:
                    gt = _load_ground_truth(s)
                    cm = compute_confusion_matrix(np.asarray(recon, dtype=np.uint8), gt)
                    code, _ = _classify_failure(
                        np.asarray(recon, dtype=np.uint8), gt, cm, alg,
                    )
                except FileNotFoundError:
                    code = "?"
                records.append(
                    {
                        "algorithm": alg,
                        "level": lv,
                        "sample": s,
                        "ssim": ssim,
                        "failure": code,
                    }
                )
    return records


def _load_ground_truth(sample: str) -> np.ndarray:
    idx = _SAMPLE_INDEX[sample]
    path = _RAW_DIR / "ground_truth" / f"true{idx}.mat"
    if not path.exists():
        raise FileNotFoundError(f"ground truth missing: {path}")
    return scipy.io.loadmat(str(path))["truth"].astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Failure classifier
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_failure(
    pred: np.ndarray,
    gt: np.ndarray,
    cm: np.ndarray,
    algorithm: str,
) -> tuple[str, dict[str, float]]:
    """Assign a failure type (A–E) from confusion-matrix signals.

    Heuristics:
        A · Ghost            — FP-dominant (pred non-water vs GT water)
        B · Missing          — FN-dominant (pred water vs GT inclusion)
        C · Class flip       — resistive ↔ conductive swap
        D · Boundary erosion — inclusion detected but the footprint shrinks
        E · Mask suppression — PNPE2E with FN-heavy + balanced inclusion confusion
    """
    fp = float(cm[0, 1] + cm[0, 2])
    fn = float(cm[1, 0] + cm[2, 0])
    swap = float(cm[1, 2] + cm[2, 1])

    pred_area = int((pred > 0).sum())
    gt_area = int((gt > 0).sum())
    inter = int(((pred > 0) & (gt > 0)).sum())
    union = int(((pred > 0) | (gt > 0)).sum())
    iou = inter / union if union else 0.0
    size_ratio = pred_area / gt_area if gt_area else 0.0

    erosion = 0.0
    if iou >= 0.20 and 0.0 < size_ratio < 0.85:
        erosion = (1.0 - size_ratio) * iou

    inclusion_balance = min(cm[1, 2], cm[2, 1])

    scores = {
        "A": fp,
        "B": fn,
        "C": swap,
        "D": erosion,
        "E": fn * (1.5 if algorithm == "pnpe2e" else 0.0)
        + inclusion_balance * (0.5 if algorithm == "pnpe2e" else 0.0),
    }

    # Strong unique signal → respect it
    if swap >= 0.25 and swap >= max(fp, fn) * 0.9:
        return "C", scores
    if fp > 0.0 and fp >= fn * 1.4 and fp >= 0.05:
        return "A", scores
    # Erosion is FN-driven but preserves a reasonable IoU — check before plain B.
    if erosion >= 0.10 and iou >= 0.20:
        return "D", scores
    if fn > 0.0 and fn >= fp * 1.4 and fn >= 0.05:
        if algorithm == "pnpe2e" and inclusion_balance >= 0.05:
            return "E", scores
        return "B", scores
    if erosion >= 0.05:
        return "D", scores

    # Generic tiebreak by algorithm's documented archetype
    fallback = {"abc1": "D", "cuqi8": "B", "pnpe2e": "C"}
    return fallback.get(algorithm, "A"), scores


# ═══════════════════════════════════════════════════════════════════════════════
# Figure builders
# ═══════════════════════════════════════════════════════════════════════════════

def _spatial_ssim_figure(ssim_map: np.ndarray) -> go.Figure:
    """Heatmap of local SSIM. Red = bad pixels driving the score down."""
    fig = go.Figure(
        go.Heatmap(
            z=ssim_map,
            colorscale="RdYlGn",
            zmin=-1.0, zmax=1.0,
            zsmooth=False,
            colorbar=dict(
                thickness=10, len=0.85,
                tickfont=dict(color="#ccc", size=10),
                title=dict(text="SSIM", side="right", font=dict(color="#ccc", size=11)),
            ),
            hovertemplate="x:%{x}<br>y:%{y}<br>ssim:%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor="#14142a",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, autorange="reversed"),
        uirevision="m5-ssim",
    )
    return fig


def _confusion_figure(cm: np.ndarray) -> go.Figure:
    """3×3 confusion matrix annotated with percentages."""
    text = [[f"{cm[i, j] * 100:.1f}%" for j in range(3)] for i in range(3)]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=_CLASS_LABEL,
            y=_CLASS_LABEL,
            colorscale=[
                [0.0, "#14142a"],
                [0.5, "#5b8def"],
                [1.0, "#e85d75"],
            ],
            zmin=0.0, zmax=1.0,
            text=text,
            texttemplate="%{text}",
            textfont=dict(color="#fff", size=13, family="monospace"),
            colorbar=dict(
                thickness=10, len=0.85,
                tickfont=dict(color="#ccc", size=10),
                title=dict(text="row %", side="right",
                           font=dict(color="#ccc", size=11)),
            ),
            hovertemplate="GT %{y} → pred %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        margin=dict(l=70, r=10, t=20, b=40),
        xaxis=dict(title=dict(text="Prediction", font=dict(color=MUTED, size=11)),
                   tickfont=dict(color="#ccc", size=11), side="bottom"),
        yaxis=dict(title=dict(text="Ground Truth", font=dict(color=MUTED, size=11)),
                   tickfont=dict(color="#ccc", size=11), autorange="reversed"),
        uirevision="m5-confusion",
    )
    return fig


def _boundary_radial_figure(
    pred: np.ndarray,
    gt: np.ndarray,
    level: int,
    n_bins: int = 36,
) -> go.Figure:
    """Polar histogram of error pixels by angle around the image centroid."""
    err = (pred != gt) & ((pred > 0) | (gt > 0))
    ys, xs = np.where(err)
    h, w = err.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    if len(xs) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No segmentation errors", showarrow=False,
            xref="paper", yref="paper", x=0.5, y=0.5,
            font=dict(color=MUTED, size=12),
        )
        fig.update_layout(
            paper_bgcolor=CARD, plot_bgcolor=CARD,
            margin=dict(l=10, r=10, t=10, b=10),
            uirevision="m5-boundary",
        )
        return fig

    angles_deg = (np.degrees(np.arctan2(-(ys - cy), xs - cx)) + 360) % 360
    bin_edges = np.linspace(0, 360, n_bins + 1)
    counts, _ = np.histogram(angles_deg, bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Electrode geometry: 32 positions equally spaced around the rim.
    # The current level keeps `active` (drawn green); the rest are removed
    # (drawn dim gray so the user can see the measurement gaps).
    active = set(subsample_electrodes(level))
    all_angles = [(360.0 * idx / 32.0) % 360 for idx in range(32)]
    active_angles  = [a for i, a in enumerate(all_angles) if i in active]
    removed_angles = [a for i, a in enumerate(all_angles) if i not in active]

    rmax = float(counts.max()) if counts.max() > 0 else 1.0
    r_rim = rmax * 1.18                       # ring sits clearly outside the bars
    r_axis_max = rmax * 1.30                   # leave headroom so dots aren't clipped

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=counts,
        theta=centers,
        width=[360 / n_bins] * n_bins,
        marker=dict(
            color=counts,
            colorscale=[[0.0, "#1e1e2f"], [0.5, "#f4c870"], [1.0, "#e85d75"]],
            line=dict(color="#1e1e2f", width=0.5),
        ),
        hovertemplate="θ=%{theta:.0f}°<br>errors=%{r}<extra></extra>",
        name="Error count",
    ))

    # Removed electrodes (faint gray X marks)
    if removed_angles:
        fig.add_trace(go.Scatterpolar(
            r=[r_rim] * len(removed_angles),
            theta=removed_angles,
            mode="markers",
            marker=dict(size=10, color="rgba(154,160,180,0.55)",
                        symbol="x-thin",
                        line=dict(color="#9aa0b4", width=1.4)),
            name="Removed electrode",
            hovertemplate="removed @ %{theta:.0f}°<extra></extra>",
        ))

    # Active electrodes (bright green dots)
    fig.add_trace(go.Scatterpolar(
        r=[r_rim] * len(active_angles),
        theta=active_angles,
        mode="markers",
        marker=dict(size=11, color=SUCCESS, symbol="circle",
                    line=dict(color="#0a0a18", width=1.0)),
        name=f"Active electrode ({len(active_angles)})",
        hovertemplate="electrode @ %{theta:.0f}°<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor=CARD,
        polar=dict(
            bgcolor="#14142a",
            radialaxis=dict(tickfont=dict(color=MUTED, size=9),
                            gridcolor="#2a2a4a", linecolor="#2a2a4a",
                            angle=90, tickangle=90,
                            range=[0, r_axis_max]),
            angularaxis=dict(tickfont=dict(color=TEXT, size=10),
                             gridcolor="#2a2a4a", linecolor="#2a2a4a",
                             direction="counterclockwise", rotation=0),
        ),
        showlegend=True,
        legend=dict(font=dict(color=TEXT, size=11),
                    bgcolor="rgba(0,0,0,0.45)",
                    bordercolor=BORDER, borderwidth=1,
                    orientation="h", x=0.5, xanchor="center", y=-0.08),
        margin=dict(l=20, r=20, t=10, b=40),
        uirevision="m5-boundary",
    )
    return fig


def _measurement_residual_figure(
    level: int,
    sample: str,
    pred: np.ndarray,
    gt: np.ndarray,
) -> go.Figure:
    """Per-injection inclusion-vs-reference voltage energy.

    For each injection, plots ||V_meas - V_ref|| (RMS) — the larger the bar,
    the more the inclusion perturbs that injection.  Bars below the dashed
    threshold are uninformative measurements (data-side limitation).
    """
    try:
        measurement = _LOADER.load(level=level, sample=sample)
        per_inj = _per_injection_perturbation(measurement)
    except Exception as exc:  # pragma: no cover - surfaced to UI
        logger.warning("M5 measurement residual unavailable: %s", exc)
        fig = go.Figure()
        fig.add_annotation(
            text="Measurement data unavailable", showarrow=False,
            xref="paper", yref="paper", x=0.5, y=0.5,
            font=dict(color=MUTED, size=12),
        )
        fig.update_layout(paper_bgcolor=CARD, plot_bgcolor=CARD,
                          margin=dict(l=10, r=10, t=10, b=10),
                          uirevision="m5-meas")
        return fig

    n = len(per_inj)
    angles = np.linspace(0.0, 360.0, n, endpoint=False)

    # Weak-signal threshold = 25th percentile of per-injection perturbation.
    # This is robust across levels: at L1 ref subtraction is exact; at L2-L7
    # the cached ref.mat is sliced to match the subsampled shape (approximate),
    # so a percentile cutoff stays meaningful even when raw magnitudes shift.
    if n:
        threshold = float(np.percentile(per_inj, 25))
    else:
        threshold = 0.0

    colors = ["#e85d75" if v <= threshold else "#5b8def" for v in per_inj]

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=per_inj,
        theta=angles,
        width=[360 / max(n, 1)] * n,
        marker=dict(color=colors, line=dict(color="#1e1e2f", width=0.4)),
        hovertemplate=("injection %{theta:.0f}°<br>"
                       "‖ΔV‖=%{r:.4f}<extra></extra>"),
        name="‖V − V_ref‖",
    ))
    # Threshold ring — drawn AFTER the bars so it sits on top of them.
    if threshold > 0 and n:
        ring_theta = np.linspace(0, 360, 180)
        fig.add_trace(go.Scatterpolar(
            r=[threshold] * len(ring_theta),
            theta=ring_theta,
            mode="lines",
            line=dict(color=WARN, width=2.0, dash="dash"),
            hoverinfo="skip",
            name=f"weak-signal threshold (P25 = {threshold:.3f})",
        ))

    fig.update_layout(
        paper_bgcolor=CARD,
        polar=dict(
            bgcolor="#14142a",
            radialaxis=dict(tickfont=dict(color=MUTED, size=9),
                            gridcolor="#2a2a4a", linecolor="#2a2a4a",
                            angle=90, tickangle=90),
            angularaxis=dict(tickfont=dict(color=TEXT, size=10),
                             gridcolor="#2a2a4a", linecolor="#2a2a4a",
                             direction="counterclockwise", rotation=0),
        ),
        showlegend=True,
        legend=dict(font=dict(color=TEXT, size=11),
                    bgcolor="rgba(0,0,0,0.45)",
                    bordercolor=BORDER, borderwidth=1,
                    orientation="h", x=0.5, xanchor="center", y=-0.08),
        margin=dict(l=20, r=20, t=10, b=40),
        uirevision="m5-meas",
    )
    return fig


def _per_injection_perturbation(measurement) -> np.ndarray:  # noqa: ANN001
    """Return ||V_inj - V_ref|| per injection, falling back to ||V_inj|| RMS."""
    v_meas = np.asarray(measurement.voltage_matrix, dtype=np.float64)
    n_inj, n_meas = v_meas.shape

    v_ref = _load_reference_voltage(n_inj, n_meas)
    if v_ref is not None and v_ref.shape == v_meas.shape:
        delta = v_meas - v_ref
    else:
        delta = v_meas - np.mean(v_meas, axis=0, keepdims=True)

    return np.sqrt(np.mean(delta ** 2, axis=1))


_REF_VOLTAGE_CACHE: dict[tuple[int, int], np.ndarray | None] = {}


def _load_reference_voltage(n_inj: int, n_meas: int) -> np.ndarray | None:
    """Load and reshape the empty-tank voltage to ``(n_inj, n_meas)`` if possible."""
    key = (n_inj, n_meas)
    if key in _REF_VOLTAGE_CACHE:
        return _REF_VOLTAGE_CACHE[key]
    if not _REF_MAT.exists():
        _REF_VOLTAGE_CACHE[key] = None
        return None
    try:
        ref = scipy.io.loadmat(str(_REF_MAT))
        for k in ("Uelref", "Uel", "uref"):
            if k in ref:
                arr = np.asarray(ref[k], dtype=np.float64).flatten()
                if arr.size >= n_inj * n_meas:
                    arr = arr[: n_inj * n_meas].reshape(n_inj, n_meas)
                    _REF_VOLTAGE_CACHE[key] = arr
                    return arr
        _REF_VOLTAGE_CACHE[key] = None
        return None
    except Exception:  # pragma: no cover
        _REF_VOLTAGE_CACHE[key] = None
        return None
