"""Module 6: Measurement Domain Viewer. Owner: Asmita Bhuva."""

from dash import html


def layout() -> html.Div:
    return html.Div([
        html.H3("Measurement Domain Viewer", style={"color": "#eee"}),
        html.P(
            "3 sub-panels: Current (polar bar), Voltage (polar + difference), "
            "Resistance (scatter + 2D heatmap). Updates on level/sample selection.",
            style={"color": "#aaa"},
        ),
        _coming_soon(),
    ], style={"padding": "20px"})


def register_callbacks(app) -> None:  # noqa: ANN001
    """Register M6 callbacks. TODO: implement in Asmita's sprint."""


def _coming_soon() -> html.Div:
    return html.Div(
        "Pending implementation.",
        style={
            "backgroundColor": "#2a2a3f", "color": "#666", "borderRadius": "8px",
            "padding": "40px", "textAlign": "center", "marginTop": "20px",
        },
    )
