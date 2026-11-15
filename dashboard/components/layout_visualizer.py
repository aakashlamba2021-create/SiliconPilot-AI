"""
2D Interactive Layout Geometry Visualizer — SiliconPilot AI
Renders authentic transistor placement geometry, gate finger arrays, guard rings,
and floorplan bounding boxes derived from layout parameters (W, L, Fingers, M, x, y, delta_x, delta_y).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def render_transistor_layout(
    candidate: pd.Series | Dict[str, Any],
    title: str = "Transistor Placement & Floorplan Geometry",
    canvas_size: int = 520,
) -> go.Figure:
    """
    Renders an authentic 2D layout geometry visualization of a candidate design.
    Features:
      - Overall floorplan bounding box derived from layout area
      - Substrate well enclosure & guard ring
      - Active diffusion area (W x L per finger)
      - Polysilicon gate fingers (interleaved S-G-D-G-S)
      - Placement perturbation displacement vector (delta_x, delta_y)
    """
    w = float(candidate.get("transistor_width_W", 25.0))
    l = float(candidate.get("transistor_length_L", 0.5))
    fingers = int(candidate.get("finger_count", 4))
    m = int(candidate.get("multiplicity", 2))
    orient = int(candidate.get("orientation", 0))
    x_pos = float(candidate.get("x_position", 50.0))
    y_pos = float(candidate.get("y_position", 50.0))
    dx = float(candidate.get("delta_x", 0.0))
    dy = float(candidate.get("delta_y", 0.0))
    total_area = float(candidate.get("area", 1500.0))

    # Calculate bounding box dimensions matching area (assuming ~1.2 aspect ratio)
    bbox_h = math.sqrt(total_area / 1.2)
    bbox_w = bbox_h * 1.2

    fig = go.Figure()

    # 1. Floorplan Bounding Box (Outer Boundary)
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=bbox_w,
        y1=bbox_h,
        line=dict(color="#38BDF8", width=2, dash="dot"),
        fillcolor="rgba(14, 165, 233, 0.04)",
        name="Layout Bounding Box",
    )

    # 2. Substrate Well Enclosure / Guard Ring
    margin = min(bbox_w, bbox_h) * 0.08
    fig.add_shape(
        type="rect",
        x0=margin,
        y0=margin,
        x1=bbox_w - margin,
        y1=bbox_h - margin,
        line=dict(color="#F59E0B", width=1.5),
        fillcolor="rgba(245, 158, 11, 0.03)",
        name="Substrate Guard Ring",
    )

    # 3. Transistor Diffusion Active Area
    # Scale coordinates into canvas space
    center_x = (x_pos / 100.0) * (bbox_w - 2 * margin) + margin
    center_y = (y_pos / 100.0) * (bbox_h - 2 * margin) + margin

    diff_width = max(6.0, (w / 50.0) * (bbox_w * 0.45))
    diff_length = max(4.0, (l / 2.0) * (bbox_h * 0.25))

    # Apply rotation
    if orient in [90, 270]:
        diff_width, diff_length = diff_length, diff_width

    diff_x0 = center_x - diff_width / 2
    diff_y0 = center_y - diff_length / 2
    diff_x1 = center_x + diff_width / 2
    diff_y1 = center_y + diff_length / 2

    # Active diffusion region (Silicon OD)
    fig.add_shape(
        type="rect",
        x0=diff_x0,
        y0=diff_y0,
        x1=diff_x1,
        y1=diff_y1,
        line=dict(color="#22C55E", width=2),
        fillcolor="rgba(34, 197, 94, 0.25)",
        name="Active Diffusion (OD)",
    )

    # 4. Polysilicon Gate Fingers (PO)
    finger_step = diff_width / (fingers + 1)
    for i in range(1, fingers + 1):
        fx = diff_x0 + i * finger_step
        fig.add_shape(
            type="rect",
            x0=fx - finger_step * 0.15,
            y0=diff_y0 - diff_length * 0.10,
            x1=fx + finger_step * 0.15,
            y1=diff_y1 + diff_length * 0.10,
            line=dict(color="#EF4444", width=1),
            fillcolor="rgba(239, 68, 68, 0.65)",
        )

    # 5. Perturbation Offset Vector Arrow
    if abs(dx) > 0.05 or abs(dy) > 0.05:
        vector_scale = 3.0
        fig.add_annotation(
            x=center_x + dx * vector_scale,
            y=center_y + dy * vector_scale,
            ax=center_x,
            ay=center_y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2.5,
            arrowcolor="#EC4899",
        )

    # Center Marker
    fig.add_trace(
        go.Scatter(
            x=[center_x],
            y=[center_y],
            mode="markers+text",
            marker=dict(color="#F43F5E", size=8, symbol="cross"),
            text=[f"M({w}µ/{l}µ, Nf={fingers})"],
            textposition="top center",
            textfont=dict(color="#E2E8F0", size=10),
            name="Transistor Centroid",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#38BDF8", size=14)),
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        xaxis=dict(
            title="Layout X Dimension (µm)",
            range=[-2, bbox_w + 4],
            showgrid=True,
            gridcolor="#1F2937",
            zeroline=False,
        ),
        yaxis=dict(
            title="Layout Y Dimension (µm)",
            range=[-2, bbox_h + 4],
            showgrid=True,
            gridcolor="#1F2937",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        width=canvas_size,
        height=canvas_size,
        showlegend=False,
    )
    return fig


def render_layout_comparison(
    baseline: pd.Series | Dict[str, Any],
    recommended: pd.Series | Dict[str, Any],
) -> go.Figure:
    """
    Renders an overlaid or dual comparison of Baseline vs top-ranked exploratory parameter preview.
    """
    fig = go.Figure()

    # Baseline Bounding Box
    b_area = float(baseline.get("area", 2000.0))
    b_h = math.sqrt(b_area / 1.2)
    b_w = b_h * 1.2
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=b_w,
        y1=b_h,
        line=dict(color="#94A3B8", width=2, dash="dash"),
        fillcolor="rgba(148, 163, 184, 0.05)",
        name="Baseline Footprint",
    )

    # Top-ranked exploratory candidate bounding box
    r_area = float(recommended.get("area", 1500.0))
    r_h = math.sqrt(r_area / 1.2)
    r_w = r_h * 1.2
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=r_w,
        y1=r_h,
        line=dict(color="#22C55E", width=2),
        fillcolor="rgba(34, 197, 94, 0.08)",
        name="Exploratory Footprint",
    )

    # Scatter points for centroids
    b_x = (float(baseline.get("x_position", 50)) / 100.0) * b_w
    b_y = (float(baseline.get("y_position", 50)) / 100.0) * b_h
    r_x = (float(recommended.get("x_position", 50)) / 100.0) * r_w
    r_y = (float(recommended.get("y_position", 50)) / 100.0) * r_h

    fig.add_trace(
        go.Scatter(
            x=[b_x],
            y=[b_y],
            mode="markers+text",
            marker=dict(color="#94A3B8", size=10, symbol="circle-open"),
            text=["Baseline"],
            textposition="bottom right",
            textfont=dict(color="#94A3B8", size=11),
            name="Baseline Centroid",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[r_x],
            y=[r_y],
            mode="markers+text",
            marker=dict(color="#22C55E", size=12, symbol="star"),
            text=["Exploratory estimate"],
            textposition="top right",
            textfont=dict(color="#22C55E", size=11, weight="bold"),
            name="Exploratory Candidate Centroid",
        )
    )

    # Shift arrow
    fig.add_annotation(
        x=r_x,
        y=r_y,
        ax=b_x,
        ay=b_y,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor="#38BDF8",
    )

    max_dim = max(b_w, r_w, b_h, r_h) * 1.15
    fig.update_layout(
        title=dict(
            text=f"Floorplan Area Shift: {b_area:.1f} µm² → {r_area:.1f} µm² ({((r_area-b_area)/b_area)*100:+.1f}%)",
            font=dict(color="#38BDF8", size=14),
        ),
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        xaxis=dict(title="X Dimension (µm)", range=[-5, max_dim], showgrid=True, gridcolor="#1F2937"),
        yaxis=dict(title="Y Dimension (µm)", range=[-5, max_dim], showgrid=True, gridcolor="#1F2937", scaleanchor="x", scaleratio=1),
        margin=dict(l=40, r=40, t=50, b=40),
        height=480,
    )
    return fig


def render_pareto_tradeoff_chart(
    df: pd.DataFrame,
    x_col: str = "area",
    y_col: str = "dc_gain",
    title: str = "Multi-Objective Pareto Trade-Off Landscape",
) -> go.Figure:
    """
    Renders an interactive Pareto frontier scatter chart highlighting Rank-1 non-dominated candidates.
    """
    plot_df = df.copy()
    plot_df["Pareto Status"] = np.where(plot_df.get("is_pareto_optimal", False), "Pareto Optimal (Rank 1)", "Dominated Candidate")

    hover_cols = [c for c in ["candidate_id", "siliconpilot_score", "dc_gain", "gain_bandwidth_product", "phase_margin", "area", "drc_clean_prob"] if c in plot_df.columns]

    fig = go.Figure()

    # Non-Pareto dominated points
    dominated = plot_df[~plot_df["is_pareto_optimal"]]
    if not dominated.empty:
        fig.add_trace(
            go.Scatter(
                x=dominated[x_col],
                y=dominated[y_col],
                mode="markers",
                marker=dict(color="#64748B", size=8, opacity=0.6),
                text=dominated["candidate_id"],
                customdata=dominated[hover_cols],
                hovertemplate="<b>%{customdata[0]}</b><br>"
                + f"{x_col}: "
                + "%{x:.2f}<br>"
                + f"{y_col}: "
                + "%{y:.2f}<br>Score: %{customdata[1]}<extra></extra>",
                name="Dominated Candidates",
            )
        )

    # Pareto Optimal points
    pareto = plot_df[plot_df["is_pareto_optimal"]].sort_values(by=x_col)
    if not pareto.empty:
        fig.add_trace(
            go.Scatter(
                x=pareto[x_col],
                y=pareto[y_col],
                mode="lines+markers",
                line=dict(color="#06B6D4", width=2, dash="dash"),
                marker=dict(color="#22C55E", size=12, symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
                text=pareto["candidate_id"],
                customdata=pareto[hover_cols],
                hovertemplate="<b>%{customdata[0]} (Pareto Optimal)</b><br>"
                + f"{x_col}: "
                + "%{x:.2f}<br>"
                + f"{y_col}: "
                + "%{y:.2f}<br>Score: %{customdata[1]}<extra></extra>",
                name="Pareto Optimal Frontier",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#38BDF8", size=15)),
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        xaxis=dict(title=f"{x_col.replace('_', ' ').title()}", showgrid=True, gridcolor="#1F2937"),
        yaxis=dict(title=f"{y_col.replace('_', ' ').title()}", showgrid=True, gridcolor="#1F2937"),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    return fig

