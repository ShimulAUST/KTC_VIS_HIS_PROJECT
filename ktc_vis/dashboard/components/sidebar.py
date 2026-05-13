"""Shared sidebar controls. Owner: Asmita Bhuva."""

from dash import dcc, html


def build_sidebar() -> html.Div:
    """Return the shared sidebar with algorithm, level, and sample selectors."""
    return html.Div(
        id="sidebar",
        children=[
            html.H2("KTC-Vis", style={"color": "#fff", "marginBottom": "4px"}),
            html.P(
                "EIT Algorithm Benchmarking",
                style={"color": "#aaa", "fontSize": "12px", "marginTop": 0},
            ),
            html.Hr(style={"borderColor": "#444"}),

            html.Label("Algorithm", style={"color": "#ccc", "fontWeight": "bold"}),
            dcc.Dropdown(
                id="sidebar-algorithm-dropdown",
                options=[
                    {"label": "ABC1", "value": "abc1"},
                    {"label": "CUQI8", "value": "cuqi8"},
                    {"label": "PNPE2E", "value": "pnpe2e"},
                ],
                value="abc1",
                clearable=False,
                style={"marginBottom": "16px"},
            ),

            html.Label("Difficulty Level", style={"color": "#ccc", "fontWeight": "bold"}),
            dcc.Slider(
                id="sidebar-level-slider",
                min=1, max=7, step=1, value=1,
                marks={i: {"label": str(i), "style": {"color": "#ccc"}} for i in range(1, 8)},
                tooltip={"placement": "bottom"},
            ),
            html.Div(style={"marginBottom": "16px"}),

            html.Label("Sample", style={"color": "#ccc", "fontWeight": "bold"}),
            dcc.RadioItems(
                id="sidebar-sample-radio",
                options=[
                    {"label": " A", "value": "a"},
                    {"label": " B", "value": "b"},
                    {"label": " C", "value": "c"},
                ],
                value="a",
                inline=True,
                style={"color": "#ccc", "marginBottom": "16px"},
            ),

            html.Hr(style={"borderColor": "#444"}),
            html.P(
                "Select an algorithm, difficulty level, and sample above. "
                "All modules update together.",
                style={"color": "#888", "fontSize": "11px"},
            ),
        ],
        style={
            "width": "220px",
            "minWidth": "220px",
            "backgroundColor": "#1e1e2f",
            "padding": "20px",
            "height": "100vh",
            "overflowY": "auto",
            "boxSizing": "border-box",
        },
    )
