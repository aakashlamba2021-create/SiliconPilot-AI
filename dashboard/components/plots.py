"""
Interactive Plotly Visualizations — SiliconPilot AI Dashboard
"""
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Feature Correlation Matrix") -> go.Figure:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 20:
        num_df = num_df.iloc[:, :20]
    if num_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No numeric columns are available for correlation analysis.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        )
        fig.update_layout(title=title, template="plotly_dark", height=300)
        return fig
    corr = num_df.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=title,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def plot_pareto_frontier(
    df: pd.DataFrame,
    x_col: str = "area",
    y_col: str = "dc_gain",
    title: str = "Pareto Frontier: Area vs DC Gain",
) -> go.Figure:
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="is_clean_candidate" if "is_clean_candidate" in df.columns else None,
        hover_data=[c for c in ["siliconpilot_score", "violation_status", "phase_margin"] if c in df.columns],
        title=title,
        color_discrete_map={True: "#22C55E", False: "#EF4444"},
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def plot_metric_bars(metrics_dict: dict, title: str = "Model Performance by Target (R²)") -> go.Figure:
    targets = list(metrics_dict.keys())
    r2_vals = [metrics_dict[t].get("R2", 0) for t in targets]

    fig = go.Figure(
        data=[
            go.Bar(
                x=targets,
                y=r2_vals,
                marker_color=["#38BDF8" if v > 0 else "#EF4444" for v in r2_vals],
                text=[f"{v:.3f}" for v in r2_vals],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=title,
        yaxis=dict(title="R² Score"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig

