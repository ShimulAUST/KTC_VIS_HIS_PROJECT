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

_TABS = [
    ("M1 · Explorer",    m1_reconstruction_explorer),
    ("M2 · Animator",    m2_difficulty_animator),
    ("M3 · Comparison",  m3_comparison_grid),
    ("M4 · Radar",       m4_fingerprint_radar),
    ("M5 · Autopsy",     m5_failure_autopsy),
    ("M6 · Measurements",m6_measurement_viewer),
]


def create_layout() -> html.Div:
    """Return the full app layout: sidebar + tabbed module area."""
    tabs = dcc.Tabs(
        id="main-tabs",
        value="tab-0",
        children=[
            dcc.Tab(
                label=label,
                value=f"tab-{i}",
                children=module.layout(),
                style={"backgroundColor": "#1a1a2e", "color": "#aaa", "padding": "6px 12px"},
                selected_style={"backgroundColor": "#2a2a3f", "color": "#fff",
                                "borderTop": "2px solid #7b61ff", "padding": "6px 12px"},
            )
            for i, (label, module) in enumerate(_TABS)
        ],
        style={"flex": "1"},
        colors={"border": "#333", "primary": "#7b61ff", "background": "#1a1a2e"},
    )

    return html.Div(
        [build_sidebar(), html.Div(tabs, style={"flex": "1", "overflow": "auto"})],
        style={"display": "flex", "height": "100vh",
               "backgroundColor": "#12121e", "fontFamily": "Inter, sans-serif"},
    )


def register_all_callbacks(app) -> None:  # noqa: ANN001
    """Register callbacks from all modules."""
    for _, module in _TABS:
        module.register_callbacks(app)
