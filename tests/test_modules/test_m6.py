"""Tests for dashboard module m6 (Measurement Domain Viewer). Owner: Asmita Bhuva."""

import numpy as np
import plotly.graph_objects as go
import pytest
from dash import html

from ktc_vis.dashboard.modules import m6_measurement_viewer as m6

_DATA_AVAILABLE = (m6._LOADER.measurements_dir / "data1.mat").exists()
requires_data = pytest.mark.skipif(
    not _DATA_AVAILABLE, reason="KTC2023 raw data not staged"
)


def test_layout_returns_div():
    assert isinstance(m6.layout(), html.Div)


def test_register_callbacks():
    import dash

    app = dash.Dash(__name__)
    m6.register_callbacks(app)
    callback_ids = " ".join(app.callback_map.keys())
    for component_id in (
        "m6-current-polar", "m6-voltage-polar", "m6-voltage-diff",
        "m6-resistance-scatter", "m6-resistance-heatmap",
        "m6-injection-slider", "m6-interval",
    ):
        assert component_id in callback_ids


@requires_data
@pytest.mark.parametrize("level", [1, 7])
def test_figures_are_valid_for_edge_levels(level):
    measurement = m6._load_measurement(level, "a")
    reference = m6._load_reference(level)
    last_inj = measurement.current_matrix.shape[0] - 1

    for inj_idx in (0, last_inj):
        for fig in (
            m6.current_polar_figure(measurement, level, inj_idx),
            m6.voltage_polar_figure(measurement, reference, level, inj_idx),
            m6.voltage_diff_figure(measurement, reference, inj_idx),
            m6.resistance_scatter_figure(measurement, inj_idx),
            m6.resistance_heatmap_figure(measurement, inj_idx),
        ):
            assert isinstance(fig, go.Figure)
            assert len(fig.data) >= 1


@requires_data
@pytest.mark.parametrize("level", [1, 2, 7])
def test_reference_aligned_with_measurement(level):
    """Reference must subsample to exactly the measurement's shape."""
    measurement = m6._load_measurement(level, "a")
    reference = m6._load_reference(level)
    assert reference is not None
    assert reference.voltage_matrix.shape == measurement.voltage_matrix.shape
    assert reference.current_matrix.shape == measurement.current_matrix.shape
    # Identical injection protocol: difference V − V_ref is meaningful.
    np.testing.assert_allclose(
        reference.current_matrix, measurement.current_matrix
    )


@requires_data
def test_voltage_diff_without_reference_returns_placeholder():
    measurement = m6._load_measurement(1, "a")
    fig = m6.voltage_diff_figure(measurement, None, 0)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0  # placeholder annotation only


@requires_data
@pytest.mark.parametrize("level", [1, 7])
def test_pair_angles_match_voltage_columns(level):
    measurement = m6._load_measurement(level, "a")
    angles = m6._pair_angles(measurement, level)
    assert len(angles) == measurement.voltage_matrix.shape[1]
    assert all(0.0 <= a < 360.0 for a in angles)
