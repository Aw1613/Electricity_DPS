"""Plotly chart utilities for Delhi Electricity Demand Prediction System."""

from typing import Optional
import pandas as pd
import plotly.graph_objects as go


def plot_demand_overview(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    demand_col: str = "demand_mw",
    forecast_col: Optional[str] = "forecast_mw",
    grid_capacity_mw: Optional[float] = None,
    warning_threshold_mw: Optional[float] = None,
) -> go.Figure:
    """Create an interactive time-series chart of actual demand and optional forecast."""
    fig = go.Figure()

    if demand_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[timestamp_col],
                y=df[demand_col],
                mode="lines",
                name="Actual Demand (MW)",
                line=dict(color="#1f77b4", width=2),
            )
        )

    if forecast_col and forecast_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[timestamp_col],
                y=df[forecast_col],
                mode="lines",
                name="Forecast Demand (MW)",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            )
        )

    if grid_capacity_mw:
        fig.add_hline(
            y=grid_capacity_mw,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Max Capacity ({grid_capacity_mw:,.0f} MW)",
            annotation_position="top right",
        )

    if warning_threshold_mw:
        fig.add_hline(
            y=warning_threshold_mw,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"Warning Threshold ({warning_threshold_mw:,.0f} MW)",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title="Delhi Electricity Demand & Forecast (MW)",
        xaxis_title="Time",
        yaxis_title="Demand (MW)",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
