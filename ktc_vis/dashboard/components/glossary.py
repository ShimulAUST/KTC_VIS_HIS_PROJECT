"""Glossary tooltips for the KTC-Vis dashboard.

Provides a single source of truth (`GLOSSARY`) for every abbreviation, metric,
and failure code used across the modules, plus three tiny helpers that wrap
any Dash element in a native browser tooltip:

* :func:`with_tooltip` — wrap an existing component, no visual change.
* :func:`info_pill`    — render a small inline "i" badge for adding to titles.
* :func:`glossary_term` — render the term itself as text with a dotted
  underline that hints at hover-for-definition.

Tooltips use the standard ``title`` HTML attribute, so they render natively in
every browser without any callbacks, JS, or extra CSS.
"""

from __future__ import annotations

from dash import html

from ..theme import BORDER, MUTED, TEXT_DIM


# ── Glossary ─────────────────────────────────────────────────────────────────
# Keys are case-insensitive lookup tokens; values are 1–2 sentence definitions
# written for a non-EIT reader.  Keep entries short — they render in a native
# browser tooltip and longer text gets cut on small screens.

GLOSSARY: dict[str, str] = {
    # ── Image-quality metrics ────────────────────────────────────────────
    "ssim": (
        "Structural Similarity Index (0…1). Compares the reconstructed image "
        "to the ground truth structurally. 1.0 = perfect, < 0.5 = badly "
        "distorted."
    ),
    "iou": (
        "Intersection over Union (0…1). Overlap between the predicted region "
        "and the true region for one class. 1.0 = perfect match."
    ),
    "iou water": "IoU for the water (background) class.",
    "iou resistive": "IoU for resistive inclusions (red class).",
    "iou conductive": "IoU for conductive inclusions (blue class).",
    "iou mean": "Average of the three per-class IoU values.",
    "hausdorff": (
        "Worst-case boundary distance (pixels) between predicted and true "
        "inclusion contours. Lower is better."
    ),
    "position error": (
        "Distance (pixels) between the centroid of the predicted inclusion "
        "and the true centroid. Lower is better."
    ),
    "resolution": (
        "Effective spatial resolution (pixels) of the reconstruction. "
        "Lower means sharper detail can be recovered."
    ),
    "runtime": "Wall-clock seconds for one reconstruction on this machine.",
    # ── EIT / KTC2023 vocabulary ─────────────────────────────────────────
    "electrode": (
        "Metal contact on the tank rim that injects current or measures "
        "voltage. KTC2023 uses up to 32 electrodes."
    ),
    "injection": (
        "One current-injection pattern (a pair of electrodes drives current). "
        "Each injection produces a set of boundary voltages."
    ),
    "level": (
        "KTC2023 difficulty level (1…7). Level 1 keeps all 32 electrodes; "
        "each higher level removes 2 more, leaving 20 at level 7."
    ),
    "sample": (
        "One of four test phantoms (1–4) with different inclusion shapes "
        "and conductivities."
    ),
    "algorithm": (
        "Reconstruction method under evaluation: abc1 (deep learning), "
        "cuqi8 (Bayesian inverse), or pnpe2e (plug-and-play end-to-end)."
    ),
    "abc1": "abc1 — deep-learning reconstruction (UNet variant).",
    "cuqi8": "cuqi8 — Bayesian inverse solver with MAP estimation.",
    "pnpe2e": "pnpe2e — plug-and-play end-to-end deep prior.",
    "ground truth": (
        "The known true inclusion map (256×256), used as the reference "
        "for every metric."
    ),
    "reference": (
        "Empty-tank voltages recorded before the inclusions were added. "
        "Used to compute the perturbation each injection carries."
    ),
    "perturbation": (
        "Per-injection change in boundary voltage caused by the inclusions, "
        "relative to the empty-tank reference."
    ),
    "p25": (
        "25th percentile — the value below which the weakest quarter of the "
        "measurements falls. Used as the weak-signal threshold."
    ),
    "confusion matrix": (
        "Rows = true class, columns = predicted class. The diagonal is "
        "correct; off-diagonal cells reveal what was confused with what."
    ),
    "spatial ssim": (
        "Per-pixel SSIM map: green pixels match the ground truth, red "
        "pixels hurt the score the most."
    ),
    # ── Failure codes (M5) ──────────────────────────────────────────────
    "a": (
        "Failure A — Ghost. The algorithm invented inclusions where the "
        "tank actually contains only water."
    ),
    "b": (
        "Failure B — Missing. The algorithm failed to recover an inclusion "
        "that is present in the ground truth."
    ),
    "c": (
        "Failure C — Class flip. Resistive vs. conductive labels were "
        "swapped on a recovered inclusion."
    ),
    "d": (
        "Failure D — Boundary erosion. The shape is roughly right but its "
        "edge is shrunken or eaten away."
    ),
    "e": (
        "Failure E — Mask suppression. The inclusion mask was suppressed "
        "to nearly nothing despite usable signal."
    ),
    "failure type": (
        "Auto-assigned diagnostic code (A–E) that names the dominant way "
        "this case failed."
    ),
    # ── Small UI words ──────────────────────────────────────────────────
    "lower better": "Smaller numbers are better for this metric.",
    "higher better": "Larger numbers are better for this metric.",
}


def lookup(term: str) -> str | None:
    """Return the glossary definition for ``term`` (case/space-insensitive)."""
    if term is None:
        return None
    key = str(term).strip().lower()
    if key in GLOSSARY:
        return GLOSSARY[key]
    # Try a few common normalisations.
    for variant in (key.replace("_", " "), key.replace("-", " "), key.rstrip(":")):
        if variant in GLOSSARY:
            return GLOSSARY[variant]
    return None


# ── Wrappers ─────────────────────────────────────────────────────────────────

def with_tooltip(child, term: str):
    """Wrap ``child`` in a span carrying a native browser tooltip for ``term``.

    If ``term`` is not in the glossary, ``child`` is returned unchanged so
    callers can use this freely without checking.
    """
    text = lookup(term)
    if not text:
        return child
    return html.Span(
        child,
        title=f"{term.upper()} — {text}",
        style={"cursor": "help"},
    )


def info_pill(term: str, *, size: int = 14) -> html.Span:
    """Small "i" badge to place next to a heading or label.

    Returns an empty Span if the term isn't in the glossary, so the caller
    never has to branch.
    """
    text = lookup(term)
    if not text:
        return html.Span()
    return html.Span(
        "i",
        title=f"{term.upper()} — {text}",
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "justifyContent": "center",
            "width": f"{size}px",
            "height": f"{size}px",
            "borderRadius": "999px",
            "backgroundColor": BORDER,
            "color": TEXT_DIM,
            "fontSize": f"{max(8, size - 5)}px",
            "fontWeight": 700,
            "fontStyle": "italic",
            "fontFamily": "Georgia, serif",
            "marginLeft": "6px",
            "cursor": "help",
            "verticalAlign": "middle",
            "userSelect": "none",
        },
    )


def glossary_term(
    term: str,
    label: str | None = None,
    *,
    color: str | None = None,
) -> html.Span:
    """Render ``label`` (or ``term`` if no label) as hover-for-definition text.

    Adds a dotted underline so users see there is something to hover. Falls
    back to plain text when the term is not in the glossary.
    """
    display = label if label is not None else term
    text = lookup(term)
    if not text:
        return html.Span(display, style={"color": color} if color else None)
    style = {
        "borderBottom": f"1px dotted {MUTED}",
        "cursor": "help",
    }
    if color:
        style["color"] = color
    return html.Span(
        display,
        title=f"{term.upper()} — {text}",
        style=style,
    )
