"""Tests for the shared glossary-tooltip component."""

from __future__ import annotations

from dash import html

from ktc_vis.dashboard.components.glossary import (
    GLOSSARY,
    glossary_term,
    info_pill,
    lookup,
    with_tooltip,
)


# ── Vocabulary coverage ──────────────────────────────────────────────────────

REQUIRED_TERMS = (
    # image-quality
    "ssim", "iou", "hausdorff", "position error", "runtime",
    # KTC vocabulary
    "electrode", "injection", "level", "sample", "algorithm",
    # failure codes used by M5
    "a", "b", "c", "d", "e",
)


def test_glossary_covers_required_terms():
    missing = [t for t in REQUIRED_TERMS if t not in GLOSSARY]
    assert not missing, f"glossary missing core terms: {missing}"


def test_lookup_is_case_and_space_insensitive():
    assert lookup("SSIM") == GLOSSARY["ssim"]
    assert lookup(" IoU ") == GLOSSARY["iou"]
    assert lookup("Iou_Water") == GLOSSARY["iou water"]
    assert lookup("Position-Error") == GLOSSARY["position error"]


def test_lookup_returns_none_for_unknown_term():
    assert lookup("definitely-not-a-term") is None
    assert lookup(None) is None


# ── with_tooltip ─────────────────────────────────────────────────────────────

def test_with_tooltip_wraps_known_term():
    wrapped = with_tooltip(html.Span("0.78"), "ssim")
    assert isinstance(wrapped, html.Span)
    assert wrapped.title.startswith("SSIM — ")
    assert "Structural Similarity" in wrapped.title


def test_with_tooltip_passes_through_unknown_term():
    inner = html.Span("0.78")
    out = with_tooltip(inner, "no-such-term")
    assert out is inner  # unchanged, same object


# ── info_pill ────────────────────────────────────────────────────────────────

def test_info_pill_renders_for_known_term():
    pill = info_pill("hausdorff")
    assert isinstance(pill, html.Span)
    assert pill.children == "i"
    assert pill.title.startswith("HAUSDORFF — ")


def test_info_pill_empty_for_unknown_term():
    pill = info_pill("definitely-not-a-term")
    assert isinstance(pill, html.Span)
    # Empty span — nothing rendered, no title attribute either.
    assert pill.children is None
    assert getattr(pill, "title", None) is None


# ── glossary_term ────────────────────────────────────────────────────────────

def test_glossary_term_underlined_when_known():
    span = glossary_term("ssim", "SSIM")
    assert span.children == "SSIM"
    assert "dotted" in span.style["borderBottom"]
    assert span.title.startswith("SSIM — ")


def test_glossary_term_plain_when_unknown():
    span = glossary_term("no-such-term", "label")
    assert span.children == "label"
    assert getattr(span, "title", None) is None
    # No dotted underline either.
    assert span.style is None or "borderBottom" not in (span.style or {})


def test_glossary_term_failure_codes_have_definitions():
    for code in ("A", "B", "C", "D", "E"):
        span = glossary_term(code)
        assert span.title is not None
        assert code in span.title
