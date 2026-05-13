"""Module 5: Failure Autopsy. Owner: Asmita Bhuva."""

from dash import html


def layout() -> html.Div:
    return html.Div([
        html.H3("Failure Autopsy", style={"color": "#eee"}),
        html.P(
            "Ranked worst-performing cases by SSIM. Click a row to see spatial SSIM "
            "heatmap, confusion matrix, boundary plot, and failure type badge (A–E).",
            style={"color": "#aaa"},
        ),
        _coming_soon(),
    ], style={"padding": "20px"})


def register_callbacks(app) -> None:  # noqa: ANN001
    """Register M5 callbacks. TODO: implement in Asmita's sprint."""


def _coming_soon() -> html.Div:
    return html.Div(
        "Pending implementation — requires benchmark cache.",
        style={
            "backgroundColor": "#2a2a3f", "color": "#666", "borderRadius": "8px",
            "padding": "40px", "textAlign": "center", "marginTop": "20px",
        },
    )
