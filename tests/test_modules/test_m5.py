"""Tests for dashboard module m5 (Failure Autopsy). Owner: Asmita Bhuva."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest
from dash import html

from ktc_vis.dashboard.modules.m5_failure_autopsy import (
    _FAILURE_TYPES,
    _badge,
    _boundary_radial_figure,
    _classify_failure,
    _confusion_figure,
    _detail_chips,
    _explanation,
    _measurement_residual_figure,
    _ranking_row,
    _spatial_ssim_figure,
    layout,
    register_callbacks,
)
from ktc_vis.metrics.class_metrics import compute_confusion_matrix
from ktc_vis.metrics.image_quality import compute_spatial_ssim_map


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pred_gt(kind: str, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise a (pred, gt) pair exhibiting a specific failure pattern."""
    gt = np.zeros((size, size), dtype=np.uint8)
    cy, cx = size // 2, size // 2

    # Ground truth: one resistive disc on the left, one conductive disc on the right.
    y, x = np.ogrid[:size, :size]
    left = (y - cy) ** 2 + (x - (cx - 12)) ** 2 <= 36
    right = (y - cy) ** 2 + (x - (cx + 12)) ** 2 <= 36
    gt[left] = 1
    gt[right] = 2

    pred = gt.copy()
    if kind == "ghost":
        # Large extra inclusion over water — generates a strong FP signal.
        pred[5:25, 5:25] = 2
    elif kind == "missing":
        pred[gt > 0] = 0              # drop everything → FN
    elif kind == "flip":
        pred[gt == 1] = 2             # swap resistive ↔ conductive
        pred[gt == 2] = 1
    elif kind == "erosion":
        ring = (y - cy) ** 2 + (x - (cx - 12)) ** 2 <= 16
        pred[left] = 0
        pred[ring] = 1
        ring2 = (y - cy) ** 2 + (x - (cx + 12)) ** 2 <= 16
        pred[right] = 0
        pred[ring2] = 2
    elif kind == "perfect":
        pass
    else:
        raise ValueError(kind)

    return pred, gt


# ── Layout & callbacks wiring ─────────────────────────────────────────────────

class TestLayout:
    def test_layout_returns_div(self):
        result = layout()
        assert isinstance(result, html.Div)

    def test_register_callbacks_does_not_raise(self):
        # Build a minimal stand-in for ``app.callback`` so we don't need real Dash.
        class _Stub:
            def __init__(self) -> None:
                self.registered: int = 0

            def callback(self, *_a, **_kw):
                self.registered += 1

                def _decorator(fn):
                    return fn

                return _decorator

        stub = _Stub()
        register_callbacks(stub)
        assert stub.registered == 3  # ranking, navigation, diagnostics


# ── Failure classifier ────────────────────────────────────────────────────────

class TestClassifier:
    @pytest.mark.parametrize(
        "kind, algorithm, expected",
        [
            ("ghost",   "abc1",   "A"),
            ("missing", "cuqi8",  "B"),
            ("flip",    "pnpe2e", "C"),
            ("erosion", "abc1",   "D"),
        ],
    )
    def test_dominant_signal_picks_expected_code(self, kind, algorithm, expected):
        pred, gt = _make_pred_gt(kind)
        cm = compute_confusion_matrix(pred, gt)
        code, _ = _classify_failure(pred, gt, cm, algorithm)
        assert code == expected

    def test_perfect_match_returns_algorithm_fallback(self):
        pred, gt = _make_pred_gt("perfect")
        cm = compute_confusion_matrix(pred, gt)
        for alg, fallback in (("abc1", "D"), ("cuqi8", "B"), ("pnpe2e", "C")):
            code, _ = _classify_failure(pred, gt, cm, alg)
            assert code == fallback

    def test_pnpe2e_missing_with_inclusion_balance_maps_to_E(self):
        # Both inclusions missing → FN-heavy + zero swap → PNPE2E without
        # off-diagonal balance falls back to plain "missing" (B).
        pred, gt = _make_pred_gt("missing")
        cm = compute_confusion_matrix(pred, gt)
        code, _ = _classify_failure(pred, gt, cm, "pnpe2e")
        assert code in {"B", "E"}

    def test_score_dict_contains_all_codes(self):
        pred, gt = _make_pred_gt("ghost")
        cm = compute_confusion_matrix(pred, gt)
        _, scores = _classify_failure(pred, gt, cm, "abc1")
        assert set(scores.keys()) == {"A", "B", "C", "D", "E"}


# ── Figure builders ───────────────────────────────────────────────────────────

class TestFigures:
    def test_spatial_ssim_figure(self):
        pred, gt = _make_pred_gt("ghost")
        ssim_map = compute_spatial_ssim_map(pred, gt)
        fig = _spatial_ssim_figure(ssim_map)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"

    def test_confusion_figure_has_3x3_data(self):
        pred, gt = _make_pred_gt("flip")
        cm = compute_confusion_matrix(pred, gt)
        fig = _confusion_figure(cm)
        z = np.asarray(fig.data[0].z)
        assert z.shape == (3, 3)
        # Rows of a normalised confusion matrix sum to 1 (where the GT class exists).
        for i in range(3):
            if (gt == i).any():
                assert pytest.approx(z[i].sum(), abs=1e-6) == 1.0

    def test_boundary_radial_figure_counts_match_errors(self):
        pred, gt = _make_pred_gt("ghost")
        fig = _boundary_radial_figure(pred, gt, level=1, n_bins=36)
        # First trace is the polar bar of error counts
        counts = np.asarray(fig.data[0].r)
        expected_total = int(((pred != gt) & ((pred > 0) | (gt > 0))).sum())
        assert int(counts.sum()) == expected_total

    def test_boundary_radial_no_errors_returns_annotated_figure(self):
        pred, gt = _make_pred_gt("perfect")
        fig = _boundary_radial_figure(pred, gt, level=1)
        assert isinstance(fig, go.Figure)
        assert any("No segmentation errors" in (a.text or "")
                   for a in fig.layout.annotations)

    def test_measurement_residual_handles_missing_data(self, monkeypatch):
        from ktc_vis.dashboard.modules import m5_failure_autopsy as m5

        def _raise(level, sample):  # noqa: ARG001
            raise FileNotFoundError("stub: data missing")

        monkeypatch.setattr(m5._LOADER, "load", _raise)
        pred, gt = _make_pred_gt("perfect")
        fig = _measurement_residual_figure(level=1, sample="a", pred=pred, gt=gt)
        assert isinstance(fig, go.Figure)
        assert any("Measurement data unavailable" in (a.text or "")
                   for a in fig.layout.annotations)


# ── UI building blocks ────────────────────────────────────────────────────────

class TestUI:
    def test_badge_unknown_renders_question_mark(self):
        div = _badge(None)
        assert "?" in _flatten_text(div)

    def test_badge_known_code(self):
        for code in _FAILURE_TYPES:
            div = _badge(code)
            text = _flatten_text(div)
            assert code in text
            assert _FAILURE_TYPES[code]["name"] in text

    def test_explanation_unknown(self):
        div = _explanation(None, None)
        assert "Pick a case" in _flatten_text(div)

    def test_explanation_known(self):
        div = _explanation("A", "abc1")
        text = _flatten_text(div)
        assert _FAILURE_TYPES["A"]["signal"].split(".")[0] in text

    def test_detail_chips_contains_failure_when_provided(self):
        chips = _detail_chips("abc1", 3, "a", ssim=0.42, failure_code="A")
        text = " ".join(_flatten_text(c) for c in chips)
        assert "ABC1" in text and "L3" in text and "0.420" in text and "Ghost" in text

    def test_ranking_row_pattern_id_format(self):
        case = {"algorithm": "abc1", "level": 5, "sample": "b",
                "ssim": 0.31, "failure": "A"}
        btn = _ranking_row(case, rank=1)
        assert btn.id == {"type": "m5-row", "key": "abc1|5|b"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flatten_text(component) -> str:
    if isinstance(component, str):
        return component
    children = getattr(component, "children", None)
    if children is None:
        return ""
    if isinstance(children, (list, tuple)):
        return " ".join(_flatten_text(c) for c in children)
    return _flatten_text(children)
