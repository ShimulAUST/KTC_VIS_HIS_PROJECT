"""Module 1: Reconstruction Explorer. Owner: Asmita Bhuva."""

from dash import dcc, html


def layout() -> html.Div:
    """4-panel layout: Ground Truth | Reconstruction | Segmentation | Error Overlay."""
    return html.Div([
        html.H3("Reconstruction Explorer", style={"color": "#eee"}),
        html.P(
            "Displays Ground Truth, Reconstruction, Segmentation, and Error Overlay "
            "for the selected algorithm / level / sample.",
            style={"color": "#aaa"},
        ),
        html.Div(
            [_placeholder_panel(t) for t in
             ["Ground Truth", "Reconstruction", "Segmentation", "Error Overlay"]],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
    ], style={"padding": "20px"})


def register_callbacks(app) -> None:  # noqa: ANN001
    """Register M1 callbacks. TODO: implement in Asmita's sprint."""


def _placeholder_panel(title: str) -> html.Div:
    return html.Div([
        html.H5(title, style={"color": "#ccc", "textAlign": "center", "margin": "8px 0"}),
        dcc.Graph(
            figure={
                "data": [],
                "layout": {
                    "paper_bgcolor": "#2a2a3f",
                    "plot_bgcolor": "#2a2a3f",
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False},
                    "annotations": [{"text": "Pending implementation",
                                     "showarrow": False,
                                     "font": {"color": "#666", "size": 13}}],
                    "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
                    "height": 256,
                },
            },
            config={"displayModeBar": False},
        ),
    ], style={"flex": "1", "minWidth": "240px", "backgroundColor": "#2a2a3f",
              "borderRadius": "8px", "padding": "8px"})
