"""Dashboard top-level layout. Owner: Asmita Bhuva."""

from dash import dcc, html

from ktc_vis.dashboard.components.sidebar import build_sidebar
from ktc_vis.dashboard.modules import (
    m1_reconstruction_explorer,
    m2_difficulty_animator,
    m3_comparison_grid,
    m4_fingerprint_radar,
    m5_failure_autopsy,
    m6_measurement_viewer,
)
from ktc_vis.dashboard.theme import ACCENT, BG, BORDER, MUTED, SURFACE, TEXT

_TABS = [
    ("M1 · Explorer", m1_reconstruction_explorer),
    ("M2 · Animator", m2_difficulty_animator),
    ("M3 · Comparison", m3_comparison_grid),
    ("M4 · Radar", m4_fingerprint_radar),
    ("M5 · Autopsy", m5_failure_autopsy),
    ("M6 · Measurements", m6_measurement_viewer),
]


def create_layout() -> html.Div:
    """Return the full app layout: sidebar + tabbed module area."""
    tab_style = {
        "backgroundColor": SURFACE,
        "color": MUTED,
        "padding": "10px 16px",
        "border": "none",
        "borderRight": f"1px solid {BORDER}",
        "fontSize": "12.5px",
        "fontWeight": 500,
    }
    tab_selected_style = {
        "backgroundColor": BG,
        "color": TEXT,
        "padding": "10px 16px",
        "borderTop": f"2px solid {ACCENT}",
        "borderRight": f"1px solid {BORDER}",
        "fontSize": "12.5px",
        "fontWeight": 600,
    }

    tabs = dcc.Tabs(
        id="main-tabs",
        value="tab-0",
        children=[
            dcc.Tab(
                label=label,
                value=f"tab-{i}",
                children=module.layout(),
                style=tab_style,
                selected_style=tab_selected_style,
            )
            for i, (label, module) in enumerate(_TABS)
        ],
        style={"flex": "1"},
        colors={"border": BORDER, "primary": ACCENT, "background": SURFACE},
    )

    return html.Div(
        [
            build_sidebar(),
            html.Div(
                tabs,
                style={
                    "flex": "1",
                    "overflow": "auto",
                    "backgroundColor": BG,
                },
            ),
        ],
        style={
            "display": "flex",
            "height": "100vh",
            "backgroundColor": BG,
            "fontFamily": "Inter, -apple-system, Segoe UI, Roboto, sans-serif",
        },
    )


def register_all_callbacks(app) -> None:  # noqa: ANN001
    """Register callbacks from all modules."""
    for _, module in _TABS:
        module.register_callbacks(app)
