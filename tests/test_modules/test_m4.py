"""Tests for dashboard module m4. Owner: Asmita Bhuva."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest
from dash import html

from ktc_vis.cache.hdf5_store import CacheMiss
from ktc_vis.dashboard.modules.m4_fingerprint_radar import (
    _ALGORITHMS,
    _AXES,
    _INVERT,
    _build_heatmap,
    _build_radar,
    _compute_axes,
    _fmt_raw,
    _hex_rgba,
    _load_cache_data,
    _score_color,
    layout,
    register_callbacks,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _fake_raw_data() -> dict:
    """Synthetic per-algorithm cache data for all 7 levels and 3 samples."""
    data: dict = {alg: {} for alg in _ALGORITHMS}
    runtimes = {"abc1": 0.5, "cuqi8": 5.0, "pnpe2e": 2.0}
    ssim_base = {"abc1": 0.80, "cuqi8": 0.70, "pnpe2e": 0.75}

    for alg in _ALGORITHMS:
        for level in range(1, 8):
            for sample in ["a", "b", "c"]:
                rng = np.random.default_rng(level * 10 + ord(sample))
                noise = rng.normal(0, 0.02)
                data[alg][(level, sample)] = {
                    "ssim":          float(np.clip(ssim_base[alg] + noise, 0, 1)),
                    "iou_conductive": float(np.clip(0.60 + noise, 0, 1)),
                    "iou_resistive":  float(np.clip(0.55 + noise, 0, 1)),
                    "iou_water":      0.90,
                    "runtime":        runtimes[alg],
                }
    return data


# ── _compute_axes ─────────────────────────────────────────────────────────────

class TestComputeAxes:
    def test_returns_all_axes_for_all_algorithms(self):
        norm, raw = _compute_axes(_fake_raw_data(), level=4, sample="a")
        for alg in _ALGORITHMS:
            assert set(norm[alg].keys()) == set(_AXES)
            assert set(raw[alg].keys()) == set(_AXES)

    def test_normalised_in_0_1(self):
        norm, _ = _compute_axes(_fake_raw_data(), level=4, sample="a")
        for alg in _ALGORITHMS:
            for axis, val in norm[alg].items():
                assert 0.0 <= val <= 1.0, f"{alg}/{axis} = {val}"

    def test_speed_inversion_fastest_wins(self):
        """ABC1 has the shortest runtime and should score highest on Speed."""
        norm, _ = _compute_axes(_fake_raw_data(), level=4, sample="a")
        assert norm["abc1"]["Speed"] > norm["cuqi8"]["Speed"]

    def test_invert_axes_exist_in_axis_list(self):
        for ax in _INVERT:
            assert ax in _AXES

    def test_empty_cache_all_zeros(self):
        empty = {alg: {} for alg in _ALGORITHMS}
        norm, raw = _compute_axes(empty, level=1, sample="a")
        for alg in _ALGORITHMS:
            for axis in _AXES:
                assert norm[alg][axis] == 0.0
                assert math.isnan(raw[alg][axis])

    def test_missing_measurement_axes_nan_in_raw(self):
        _, raw = _compute_axes(_fake_raw_data(), level=1, sample="a")
        for alg in _ALGORITHMS:
            assert math.isnan(raw[alg]["Voltage Residual"])
            assert math.isnan(raw[alg]["Curr. Sensitivity"])

    def test_best_algorithm_scores_1_per_axis(self):
        norm, _ = _compute_axes(_fake_raw_data(), level=4, sample="a")
        for axis in _AXES:
            scores = [norm[alg][axis] for alg in _ALGORITHMS]
            if any(s > 0 for s in scores):
                assert max(scores) == pytest.approx(1.0, abs=1e-6)

    def test_sample_change_updates_ssim_axes(self):
        """Flipping SSIM rankings between samples must change the normalised scores."""
        data = _fake_raw_data()
        # Sample a: abc1 has higher SSIM than cuqi8 (default: 0.80 vs 0.70)
        # Sample b: invert the ranking
        for level in range(1, 8):
            data["abc1"][(level, "b")]["ssim"] = 0.50
            data["cuqi8"][(level, "b")]["ssim"] = 0.95

        norm_a, _ = _compute_axes(data, level=4, sample="a")
        norm_b, _ = _compute_axes(data, level=4, sample="b")

        # On sample a: abc1 beats cuqi8 on Total SSIM
        assert norm_a["abc1"]["Total SSIM"] > norm_a["cuqi8"]["Total SSIM"]
        # On sample b: cuqi8 beats abc1 (rankings flipped)
        assert norm_b["cuqi8"]["Total SSIM"] > norm_b["abc1"]["Total SSIM"]

    def test_robustness_consistent_algo_scores_higher(self):
        """ABC1 with near-zero sample-to-sample std should beat variable CUQI8."""
        raw_data: dict = {alg: {} for alg in _ALGORITHMS}
        for sample in ["a", "b", "c"]:
            raw_data["abc1"][(4, sample)] = {
                "ssim": 0.80, "iou_conductive": 0.5, "iou_resistive": 0.5,
                "runtime": 1.0,
            }
            raw_data["cuqi8"][(4, sample)] = {
                "ssim": {"a": 0.80, "b": 0.40, "c": 0.90}[sample],
                "iou_conductive": 0.5, "iou_resistive": 0.5, "runtime": 5.0,
            }
            raw_data["pnpe2e"][(4, sample)] = {
                "ssim": 0.75, "iou_conductive": 0.5, "iou_resistive": 0.5,
                "runtime": 2.0,
            }
        norm, _ = _compute_axes(raw_data, level=4, sample="a")
        assert norm["abc1"]["Robustness"] > norm["cuqi8"]["Robustness"]


# ── _build_radar ──────────────────────────────────────────────────────────────

class TestBuildRadar:
    def _norm(self):
        norm, _ = _compute_axes(_fake_raw_data(), level=4, sample="a")
        return norm

    def test_one_trace_per_algorithm(self):
        fig = _build_radar(self._norm(), highlighted=None, level=4, sample="a")
        assert len(fig.data) == len(_ALGORITHMS)

    def test_trace_names_match_labels(self):
        fig = _build_radar(self._norm(), highlighted=None, level=4, sample="a")
        assert {t.name for t in fig.data} == {"ABC1", "CUQI8", "PNPE2E"}

    def test_highlighted_trace_thicker(self):
        fig = _build_radar(self._norm(), highlighted="abc1", level=4, sample="a")
        abc1 = next(t for t in fig.data if t.name == "ABC1")
        others = [t for t in fig.data if t.name != "ABC1"]
        assert abc1.line.width > others[0].line.width

    def test_polygon_closed(self):
        fig = _build_radar(self._norm(), highlighted=None, level=4, sample="a")
        for trace in fig.data:
            assert len(trace.r) == len(_AXES) + 1
            assert trace.r[0] == trace.r[-1]

    def test_title_contains_level_and_sample(self):
        fig = _build_radar(self._norm(), highlighted=None, level=3, sample="b")
        assert "3" in fig.layout.title.text
        assert "B" in fig.layout.title.text


# ── _build_heatmap ────────────────────────────────────────────────────────────

class TestBuildHeatmap:
    def test_returns_div(self):
        norm, raw = _compute_axes(_fake_raw_data(), level=4, sample="a")
        assert isinstance(_build_heatmap(norm, raw), html.Div)


# ── layout / register_callbacks ───────────────────────────────────────────────

class TestLayout:
    def test_returns_div(self):
        assert isinstance(layout(), html.Div)

    def test_radar_id_present(self):
        assert "m4-radar" in str(layout())

    def test_profiles_id_present(self):
        assert "m4-profiles" in str(layout())


class TestRegisterCallbacks:
    def test_registers_without_error(self):
        registered = []

        class _MockApp:
            def callback(self, *args, **kwargs):
                def dec(fn):
                    registered.append(fn.__name__)
                    return fn
                return dec

        register_callbacks(_MockApp())
        assert len(registered) >= 1


# ── Utilities ─────────────────────────────────────────────────────────────────

class TestUtilities:
    def test_hex_rgba(self):
        assert _hex_rgba("#ff8040", 0.5) == "rgba(255,128,64,0.500)"

    def test_score_color_high(self):
        assert _score_color(1.0) == "#2ecc71"

    def test_score_color_mid(self):
        assert _score_color(0.5) == "#f4c870"

    def test_score_color_low(self):
        assert _score_color(0.0) == "#e85d75"

    def test_fmt_raw_nan(self):
        assert _fmt_raw("Total SSIM", float("nan")) == "n/a"

    def test_fmt_raw_speed(self):
        assert _fmt_raw("Speed", 3.14) == "3.14 s"

    def test_fmt_raw_robustness(self):
        assert "std" in _fmt_raw("Robustness", 0.05)


# ── _load_cache_data ──────────────────────────────────────────────────────────

class TestLoadCacheData:
    def test_cache_miss_returns_empty(self):
        with patch(
            "ktc_vis.dashboard.modules.m4_fingerprint_radar.load_result",
            side_effect=CacheMiss("no cache"),
        ):
            result = _load_cache_data(cache_path="/nonexistent/results.h5")
        for alg in _ALGORITHMS:
            assert result[alg] == {}
