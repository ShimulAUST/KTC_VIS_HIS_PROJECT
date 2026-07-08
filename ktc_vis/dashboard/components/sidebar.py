# Shared sidebar controls.

from __future__ import annotations

from dash import dcc, html

from ktc_vis.dashboard.theme import (
    ACCENT,
    BORDER,
    CARD_STYLE,
    MUTED,
    SECTION_LABEL_STYLE,
    SURFACE,
    TEXT,
    TEXT_DIM,
)

_ALGORITHMS = [
    {"label": "ABC1", "value": "abc1"},
    {"label": "CUQI8", "value": "cuqi8"},
    {"label": "PNPE2E", "value": "pnpe2e"},
]
_SAMPLES = [
    {"label": "A", "value": "a"},
    {"label": "B", "value": "b"},
    {"label": "C", "value": "c"},
    {"label": "D", "value": "d"},
]


def build_sidebar() -> html.Div:
    # Return the shared sidebar with algorithm, level, and sample selectors.
    return html.Div(
        id="sidebar",
        children=[
            _brand(),
            _section(
                "Algorithm",
                "Reconstruction backend",
                dcc.Dropdown(
                    id="sidebar-algorithm-dropdown",
                    options=_ALGORITHMS,
                    value="abc1",
                    clearable=False,
                    className="ktc-dropdown",
                    style={"color": "#111"},
                ),
            ),
            _section(
                "Difficulty Level",
                "Electrodes & subsampling (KTC L1–L7)",
                _level_slider(),
            ),
            _section(
                "Sample",
                "Phantom geometry",
                dcc.RadioItems(
                    id="sidebar-sample-radio",
                    options=_SAMPLES,
                    value="a",
                    inline=True,
                    className="ktc-radio",
                    inputStyle={"marginRight": "4px"},
                    labelStyle={
                        "color": TEXT,
                        "marginRight": "12px",
                        "fontSize": "13px",
                        "fontWeight": 500,
                        "cursor": "pointer",
                    },
                ),
            ),
            _footer_help(),
        ],
        style={
            "width": "260px",
            "minWidth": "260px",
            "backgroundColor": SURFACE,
            "borderRight": f"1px solid {BORDER}",
            "padding": "20px 18px",
            "height": "100vh",
            "overflowY": "auto",
            "boxSizing": "border-box",
            "display": "flex",
            "flexDirection": "column",
            "gap": "14px",
        },
    )


# ── Building blocks ───────────────────────────────────────────


def _brand() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "K",
                        style={
                            "display": "inline-flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "width": "32px",
                            "height": "32px",
                            "borderRadius": "8px",
                            "backgroundColor": ACCENT,
                            "color": "#fff",
                            "fontWeight": 700,
                            "fontSize": "16px",
                            "marginRight": "10px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                "KTC-Vis",
                                style={"color": TEXT, "fontWeight": 700,
                                       "fontSize": "16px", "lineHeight": "1.1"},
                            ),
                            html.Div(
                                "EIT Benchmark",
                                style={"color": MUTED, "fontSize": "11px"},
                            ),
                        ]
                    ),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
        ],
        style={
            "padding": "4px 4px 12px",
            "borderBottom": f"1px solid {BORDER}",
        },
    )


def _section(title: str, hint: str, control) -> html.Div:  # noqa: ANN001
    return html.Div(
        [
            html.Div(title, style=SECTION_LABEL_STYLE),
            html.Div(
                hint,
                style={"color": MUTED, "fontSize": "11px", "margin": "-4px 0 10px"},
            ),
            control,
        ],
        style={
            **CARD_STYLE,
            "padding": "14px 14px 12px",
        },
    )


def _level_slider() -> html.Div:
    return html.Div(
        [
            dcc.Slider(
                id="sidebar-level-slider",
                min=1,
                max=7,
                step=1,
                value=1,
                marks={
                    i: {"label": f"L{i}",
                        "style": {"color": MUTED, "fontSize": "10px"}}
                    for i in range(1, 8)
                },
                tooltip={"placement": "bottom", "always_visible": False},
                included=True,
            ),
            html.Div(
                "Higher levels drop electrodes and voltage measurements.",
                style={"color": MUTED, "fontSize": "10.5px", "marginTop": "10px",
                       "lineHeight": "1.4"},
            ),
        ]
    )


def _footer_help() -> html.Div:
    return html.Div(
        [
            html.Div("Tip", style=SECTION_LABEL_STYLE),
            html.Div(
                "Selections apply to every module tab. Switch tabs to compare "
                "explorers, animators, grids, and metrics for the same setup.",
                style={"color": TEXT_DIM, "fontSize": "11.5px", "lineHeight": "1.5"},
            ),
        ],
        style={
            **CARD_STYLE,
            "padding": "12px 14px",
            "marginTop": "auto",
        },
    )
