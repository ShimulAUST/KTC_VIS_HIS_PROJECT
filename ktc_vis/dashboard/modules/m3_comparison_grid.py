"""Module 3: Side-by-Side Comparison Grid. Owner: Asmita Bhuva."""

from dash import html


def layout() -> html.Div:
    return html.Div([
        html.H3("Algorithm Comparison Grid", style={"color": "#eee"}),
        html.P(
            "3-column grid (ABC1 | CUQI8 | PNPE2E) with pairwise difference images "
            "and SSIM score table.",
            style={"color": "#aaa"},
        ),
        _coming_soon(),
    ], style={"padding": "20px"})


def register_callbacks(app) -> None:  # noqa: ANN001
    """Register M3 callbacks. TODO: implement in Asmita's sprint."""


def _coming_soon() -> html.Div:
    return html.Div(
        "Pending implementation — requires benchmark cache.",
        style={
            "backgroundColor": "#2a2a3f", "color": "#666", "borderRadius": "8px",
            "padding": "40px", "textAlign": "center", "marginTop": "20px",
        },
    )
