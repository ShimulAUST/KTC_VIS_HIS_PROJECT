"""Module 2: Difficulty Animator. Owner: Asmita Bhuva."""

from dash import html


def layout() -> html.Div:
    return html.Div([
        html.H3("Difficulty Animator", style={"color": "#eee"}),
        html.P(
            "Animates reconstruction quality across levels 1–7 with degradation curves "
            "(SSIM, IoU, Hausdorff). Requires populated HDF5 cache.",
            style={"color": "#aaa"},
        ),
        _coming_soon(),
    ], style={"padding": "20px"})


def register_callbacks(app) -> None:  # noqa: ANN001
    """Register M2 callbacks. TODO: implement in Asmita's sprint."""


def _coming_soon() -> html.Div:
    return html.Div(
        "Pending implementation — run benchmark first to populate cache.",
        style={
            "backgroundColor": "#2a2a3f", "color": "#666", "borderRadius": "8px",
            "padding": "40px", "textAlign": "center", "marginTop": "20px",
        },
    )
