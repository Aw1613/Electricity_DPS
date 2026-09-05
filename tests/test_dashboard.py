"""Unit tests for dashboard charts, components, and payload rendering (Prompt 9)."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.charts import (
    plot_actual_vs_predicted,
    plot_7d_forecast_trend,
    plot_temperature_vs_demand,
    plot_capacity_gauge,
    plot_area_breakdown_bars,
    plot_area_breakdown_pie,
    plot_renewable_net_demand_chart,
    plot_hourly_alert_timeline,
    plot_instant_day_profile,
    plot_instant_week_context,
    plot_instant_feeder_bars,
)
from src.services.demand_service import get_complete_dashboard_payload


def test_plotly_figures_generation():
    """Verify all dashboard chart utility functions generate valid Plotly figures."""
    timestamps = pd.date_range("2024-06-01 00:00:00", periods=24, freq="h")
    demands = [5000.0 + i * 100 for i in range(24)]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "demand_mw": demands,
        "predicted_demand_mw": demands,
        "predicted_lower_mw": [d * 0.97 for d in demands],
        "predicted_upper_mw": [d * 1.03 for d in demands],
        "gross_demand_mw": demands,
        "solar_generation_mw": [0.0] * 6 + [300.0] * 12 + [0.0] * 6,
        "net_demand_mw": demands,
        "temperature": [35.0] * 24,
        "alert_status": ["NORMAL"] * 20 + ["WARNING"] * 4,
    })

    # 1. Actual vs Predicted
    fig1 = plot_actual_vs_predicted(historical_df=df, forecast_df=df, capacity_mw=9000.0)
    assert isinstance(fig1, go.Figure)

    # 2. 7-Day Trend
    fig2 = plot_7d_forecast_trend(forecast_7d_df=df, capacity_mw=9000.0)
    assert isinstance(fig2, go.Figure)

    # 3. Temperature vs Demand
    fig3 = plot_temperature_vs_demand(df=df)
    assert isinstance(fig3, go.Figure)

    # 4. Capacity Gauge
    fig4 = plot_capacity_gauge(current_or_peak_mw=7800.0, capacity_mw=9000.0)
    assert isinstance(fig4, go.Figure)

    # 5. Area Bars and Donut
    area_df = pd.DataFrame({
        "area": ["South Delhi", "North Delhi", "West Delhi"],
        "peak_demand_mw": [2500.0, 2000.0, 1800.0],
        "share_pct": [40.0, 32.0, 28.0],
    })
    fig5 = plot_area_breakdown_bars(area_df)
    fig6 = plot_area_breakdown_pie(area_df)
    assert isinstance(fig5, go.Figure)
    assert isinstance(fig6, go.Figure)

    # 6. Renewable Net Demand
    fig7 = plot_renewable_net_demand_chart(df)
    assert isinstance(fig7, go.Figure)

    # 7. Hourly Alert Timeline
    fig8 = plot_hourly_alert_timeline(df)
    assert isinstance(fig8, go.Figure)

    # 8. Instant Day Profile
    day_test_df = pd.DataFrame({
        "timestamp": timestamps,
        "hour": list(range(24)),
        "actual_demand_mw": demands,
        "predicted_demand_mw": demands,
        "temperature_c": [34.0] * 24,
    })
    fig9 = plot_instant_day_profile(day_test_df, selected_hour=15, capacity_mw=9000.0)
    assert isinstance(fig9, go.Figure)

    # 9. Instant Week Context
    fig10 = plot_instant_week_context(df, selected_date_str="2024-06-01", capacity_mw=9000.0)
    assert isinstance(fig10, go.Figure)

    # 10. Instant Feeder Bars
    feeder_test_df = pd.DataFrame({
        "area": ["South Delhi", "North Delhi"],
        "feeder": ["BRPL-South", "TPDDL-North"],
        "discom": ["BSES Rajdhani", "Tata Power-DDL"],
        "instant_demand_mw": [2400.0, 1920.0],
        "share_pct": [30.0, 24.0],
    })
    fig11 = plot_instant_feeder_bars(feeder_test_df)
    assert isinstance(fig11, go.Figure)


def test_dashboard_master_payload():
    """Verify get_complete_dashboard_payload supplies complete state for app.py."""
    payload = get_complete_dashboard_payload(
        capacity_mw=9000.0,
        warning_threshold=0.85,
        critical_threshold=0.95,
        solar_capacity_mw=450.0,
    )

    required_keys = ["snapshot", "forecast_24h", "forecast_7d", "model_info", "area_analysis", "renewable_analysis", "history_48h"]
    for k in required_keys:
        assert k in payload, f"Key '{k}' missing from dashboard payload."

    # Validate forecast 24h structure
    fc_24h = payload["forecast_24h"]
    assert len(fc_24h["forecast_df"]) == 24
    assert fc_24h["peak_demand_mw"] > 0
    assert "top_peaks" in fc_24h


def test_dashboard_components(monkeypatch):
    """Verify UI components handle parameters including delta_color without errors."""
    from dashboard.components import render_kpi_card, render_alert_banner, render_top_peaks_table

    metric_calls = []
    monkeypatch.setattr("streamlit.metric", lambda *args, **kwargs: metric_calls.append(kwargs))

    render_kpi_card(
        label="24h Forecast Peak",
        value="6,500 MW",
        delta="+450 MW surge",
        delta_color="inverse",
        help_text="Test tooltip",
    )
    assert len(metric_calls) == 1
    assert metric_calls[0]["delta_color"] == "inverse"
    assert metric_calls[0]["label"] == "24h Forecast Peak"

