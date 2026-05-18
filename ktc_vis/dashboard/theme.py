"""Shared design tokens for the dashboard UI.

Centralises colors, spacing, and reusable style dictionaries so every
module renders with a consistent look.
"""

from __future__ import annotations

# ── Palette ───────────────────────────────────────────────────
BG = "#0f0f1a"
SURFACE = "#161628"
CARD = "#1e1e2f"
CARD_ALT = "#252538"
BORDER = "#2e2e44"
BORDER_STRONG = "#3a3a55"

TEXT = "#e8e8f0"
TEXT_DIM = "#c9cad6"
MUTED = "#9aa0b4"

ACCENT = "#7b61ff"
ACCENT_SOFT = "#5b8def"
SUCCESS = "#2ecc71"
WARN = "#f4c870"
DANGER = "#e85d75"

# ── Reusable styles ───────────────────────────────────────────
CARD_STYLE = {
    "backgroundColor": CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "12px",
    "boxShadow": "0 1px 2px rgba(0,0,0,0.25)",
}

SECTION_LABEL_STYLE = {
    "color": MUTED,
    "fontSize": "10.5px",
    "letterSpacing": "0.8px",
    "textTransform": "uppercase",
    "fontWeight": 600,
    "margin": "0 0 8px 0",
}
