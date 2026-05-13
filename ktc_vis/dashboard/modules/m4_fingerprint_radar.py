"""Module 4: Fingerprint Radar. Owner: Asmita Bhuva."""

from dash import html


def layout() -> html.Div:
    return html.Div([
        html.H3("Algorithm Fingerprint Radar", style={"color": "#eee"}),
        html.P(
            "9-axis radar chart comparing all algorithms across metrics. "
            "Voltage residual axis is inverted (lower = better = shown higher).",
            style={"color": "#aaa"},
        ),
        _coming_soon(),
    ], style={"padding": "20px"})


def register_callbacks(app) -> None:  # noqa: ANN001
    """Register M4 callbacks. TODO: implement in Asmita's sprint."""


def _coming_soon() -> html.Div:
    return html.Div(
        "Pending implementation — requires benchmark cache.",
        style={
            "backgroundColor": "#2a2a3f", "color": "#666", "borderRadius": "8px",
            "padding": "40px", "textAlign": "center", "marginTop": "20px",
        },
    )
