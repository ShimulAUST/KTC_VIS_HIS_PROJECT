"""Module 2: Difficulty Animator.

Shows how the input measurement and the reconstruction quality degrade as
the difficulty level rises from 1 → 7.

Until the HDF5 metric cache is populated by ``scripts/run_benchmark.py``,
SSIM/IoU/Hausdorff curves are computed on-the-fly from the precomputed
reference reconstructions (Level 1) and the subsampled measurements.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from ktc_vis.adapters import ReferenceOutputAdapter
from ktc_vis.data.loader import KTCDataLoader
from ktc_vis.data.subsampler import LEVEL_SPECS

logger = logging.getLogger(__name__)

_LOADER = KTCDataLoader()

_CURVE_ID = "m2-degradation-curve"
_MEAS_ID = "m2-measurement-curve"
_RECON_ID = "m2-recon-preview"
_STATUS_ID = "m2-status"

_PAPER_BG = "#2a2a3f"
_PLOT_BG = "#1a1a2e"


def layout() -> html.Div:
    return html.Div(
        [
            html.H3("Difficulty Animator", style={"color": "#eee"}),
            html.P(
                "Reconstruction quality and input measurement size across "
                "difficulty levels 1–7. Curves update with the selected "
                "algorithm and sample.",
                style={"color": "#aaa"},
            ),
            html.Div(id=_STATUS_ID, style={"color": "#888", "fontSize": "12px",
                                           "marginBottom": "8px"}),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(id=_CURVE_ID, config={"displayModeBar": False}),
                        style={"flex": "1", "minWidth": "320px",
                               "backgroundColor": _PAPER_BG,
                               "borderRadius": "8px", "padding": "4px"},
                    ),
                    html.Div(
                        dcc.Graph(id=_MEAS_ID, config={"displayModeBar": False}),
                        style={"flex": "1", "minWidth": "320px",
                               "backgroundColor": _PAPER_BG,
                               "borderRadius": "8px", "padding": "4px"},
                    ),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                       "marginBottom": "12px"},
            ),
        ],
        style={"padding": "20px"},
    )


@lru_cache(maxsize=16)
def _recon_for(algorithm: str, sample: str) -> np.ndarray:
    """Cached level-1 reference reconstruction for (algorithm, sample)."""
    adapter = ReferenceOutputAdapter(algorithm)
    meas = _LOADER.load(level=1, sample=sample)
    return adapter.reconstruct(meas).astype(np.uint8)


def _pixel_agreement(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred == gt).mean())


def _per_class_iou(pred: np.ndarray, gt: np.ndarray) -> dict[int, float]:
    out = {}
    for cls in (0, 1, 2):
        p, g = pred == cls, gt == cls
        union = np.logical_or(p, g).sum()
        out[cls] = float(np.logical_and(p, g).sum() / union) if union else 1.0
    return out


def register_callbacks(app) -> None:  # noqa: ANN001

    @app.callback(
        Output(_CURVE_ID, "figure"),
        Output(_MEAS_ID, "figure"),
        Output(_STATUS_ID, "children"),
        Input("sidebar-algorithm-dropdown", "value"),
        Input("sidebar-sample-radio", "value"),
    )
    def _update_curves(algorithm: str, sample: str):
        try:
            gt = _LOADER.load(level=1, sample=sample).ground_truth.astype(np.uint8)
            recon = _recon_for(algorithm, sample)
        except Exception as exc:
            logger.exception("M2 update failed")
            msg = f"Error: {exc}"
            return _empty(msg), _empty(msg), msg

        # Quality curve — recon is fixed at level 1 (precomputed); the curve
        # is therefore a flat reference. Once the HDF5 cache is populated this
        # will show real per-level degradation.
        levels = list(range(1, 8))
        agreement = [_pixel_agreement(recon, gt) for _ in levels]
        iou_inc = []
        for _ in levels:
            iou = _per_class_iou(recon, gt)
            iou_inc.append((iou[1] + iou[2]) / 2)

        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(
            x=levels, y=agreement, mode="lines+markers",
            name="Pixel agreement", line=dict(color="#5b8def", width=2),
        ))
        fig_q.add_trace(go.Scatter(
            x=levels, y=iou_inc, mode="lines+markers",
            name="IoU (inclusions)", line=dict(color="#e85d75", width=2),
        ))
        fig_q.update_layout(
            title=dict(text=f"Quality vs Difficulty · {algorithm}/{sample}",
                       font=dict(color="#ddd", size=13), x=0.5),
            paper_bgcolor=_PAPER_BG, plot_bgcolor=_PLOT_BG,
            xaxis=dict(title="Difficulty Level", color="#aaa",
                       tickmode="linear", dtick=1, gridcolor="#333"),
            yaxis=dict(title="Score (0–1)", color="#aaa",
                       range=[0, 1.05], gridcolor="#333"),
            legend=dict(font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
            height=320, margin=dict(l=50, r=20, t=40, b=40),
        )

        # Measurement size curve — actually changes per level
        n_meas = []
        n_inj = []
        for lvl in levels:
            m = _LOADER.load(level=lvl, sample=sample)
            n_meas.append(m.voltage_matrix.size)
            n_inj.append(m.current_matrix.shape[0])

        expected = [LEVEL_SPECS[l]["n_measurements"] for l in levels]

        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(
            x=levels, y=n_meas, mode="lines+markers",
            name="Subsampled (this loader)",
            line=dict(color="#7b61ff", width=2),
        ))
        fig_m.add_trace(go.Scatter(
            x=levels, y=expected, mode="lines+markers",
            name="KTC2023 Table 1",
            line=dict(color="#aaa", width=2, dash="dot"),
        ))
        fig_m.update_layout(
            title=dict(text="Input Measurements per Level",
                       font=dict(color="#ddd", size=13), x=0.5),
            paper_bgcolor=_PAPER_BG, plot_bgcolor=_PLOT_BG,
            xaxis=dict(title="Difficulty Level", color="#aaa",
                       tickmode="linear", dtick=1, gridcolor="#333"),
            yaxis=dict(title="# voltage measurements", color="#aaa",
                       gridcolor="#333"),
            legend=dict(font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
            height=320, margin=dict(l=60, r=20, t=40, b=40),
        )

        status = (
            f"algorithm={algorithm}  sample={sample}  •  "
            "Quality curve is flat until Docker adapters provide live "
            "reconstructions for levels 2–7."
        )
        return fig_q, fig_m, status


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_PAPER_BG, plot_bgcolor=_PLOT_BG,
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(text=msg, showarrow=False,
                          font=dict(color="#888"), xref="paper", yref="paper",
                          x=0.5, y=0.5)],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig
