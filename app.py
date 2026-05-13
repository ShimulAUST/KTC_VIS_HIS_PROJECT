"""KTC-Vis entry point.

Run with:
    python app.py

Then open http://localhost:8050
"""

import yaml
import dash

from ktc_vis.dashboard.layout import create_layout, register_all_callbacks

# ── Load config ───────────────────────────────────────────────
with open("configs/experiment.yaml") as f:
    config = yaml.safe_load(f)

dashboard_cfg = config.get("dashboard", {})

# ── Initialise Dash app ───────────────────────────────────────
app = dash.Dash(
    __name__,
    title=dashboard_cfg.get("title", "KTC-Vis"),
    suppress_callback_exceptions=True,
)
server = app.server  # exposed for gunicorn / Docker

app.layout = create_layout()
register_all_callbacks(app)

# ── Startup banner ────────────────────────────────────────────
if __name__ == "__main__":
    host = dashboard_cfg.get("host", "0.0.0.0")
    port = int(dashboard_cfg.get("port", 8050))
    debug = bool(dashboard_cfg.get("debug", False))

    print("\n" + "=" * 54)
    print("  KTC-Vis Dashboard")
    print(f"  http://localhost:{port}")
    print(f"  Algorithms : {', '.join(config['benchmark']['algorithms'])}")
    print(f"  Levels     : {config['benchmark']['levels']}")
    print(f"  Cache      : {config['data']['cache_path']}")
    print("=" * 54 + "\n")

    app.run(host=host, port=port, debug=debug)
