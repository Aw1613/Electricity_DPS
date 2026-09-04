"""Unit tests for Phase 5 forecasting engine: 24h, 7d recursive, and peak analysis."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecast.predict_24h import predict_next_24h, prepare_recent_history, prepare_weather_forecast
from src.forecast.predict_7d import predict_next_7d, aggregate_daily_forecast
from src.forecast.analyze import detect_peaks, add_uncertainty_bounds, get_top_peaks


def test_prepare_recent_history():
    """Verify history preparation pads or sorts correctly."""
    # Mock short history
    timestamps = pd.date_range("2024-06-01 00:00:00", periods=200, freq="h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "demand_mw": np.random.uniform(4000, 7000, 200),
    })
    history = prepare_recent_history(df, min_hours=168)
    assert len(history) >= 168
    assert "timestamp" in history.columns
    assert "demand_mw" in history.columns


def test_prepare_weather_forecast():
    """Verify future weather forecast generation."""
    start_ts = pd.Timestamp("2024-06-05 12:00:00")
    weather_24h = prepare_weather_forecast(start_timestamp=start_ts, horizon_hours=24)
    assert len(weather_24h) == 24
    assert "temperature" in weather_24h.columns
    assert "humidity" in weather_24h.columns
    assert weather_24h["temperature"].min() > 10.0
    assert weather_24h["temperature"].max() < 60.0


def test_predict_next_24h():
    """Verify 24-hour prediction engine output format and bounds."""
    forecast = predict_next_24h()
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == 24
    assert "timestamp" in forecast.columns
    assert "predicted_demand_mw" in forecast.columns
    assert "temperature" in forecast.columns

    # Verify no NaN values
    assert forecast["predicted_demand_mw"].isna().sum() == 0

    # Operational range sanity check for Delhi demand (1,500 to 12,000 MW)
    assert (forecast["predicted_demand_mw"] >= 1500.0).all()
    assert (forecast["predicted_demand_mw"] <= 12000.0).all()


def test_predict_next_7d_and_aggregation():
    """Verify 7-day multi-step recursive forecasting and daily aggregation."""
    forecast_7d = predict_next_7d()
    assert isinstance(forecast_7d, pd.DataFrame)
    assert len(forecast_7d) == 168
    assert forecast_7d["predicted_demand_mw"].isna().sum() == 0

    # Check that sequential hours advance chronologically
    ts_series = pd.to_datetime(forecast_7d["timestamp"])
    diffs = ts_series.diff()[1:]
    assert (diffs == pd.Timedelta(hours=1)).all()

    # Verify daily aggregation
    daily_summary = aggregate_daily_forecast(forecast_7d)
    assert len(daily_summary) == 7
    assert "peak_demand_mw" in daily_summary.columns
    assert "mean_demand_mw" in daily_summary.columns
    assert (daily_summary["peak_demand_mw"] >= daily_summary["mean_demand_mw"]).all()


def test_detect_peaks_and_uncertainty():
    """Verify peak detection, capacity utilization, alert level, and uncertainty intervals."""
    # Synthetic forecast test DataFrame
    timestamps = pd.date_range("2024-06-15 00:00:00", periods=24, freq="h")
    demands = [5000.0 + i * 150.0 for i in range(24)]  # Peak will be 5000 + 23*150 = 8450 MW
    forecast_df = pd.DataFrame({
        "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_demand_mw": demands,
        "temperature": [35.0] * 24,
        "hour": timestamps.hour,
    })

    # Capacity = 9000 MW, 8450 / 9000 = 93.88% -> WARNING status
    peak_info = detect_peaks(forecast_df, capacity_mw=9000.0, warning_threshold=0.85, critical_threshold=0.95)
    assert peak_info["peak_demand_mw"] == 8450.0
    assert peak_info["peak_timestamp"] == str(timestamps[-1])
    assert peak_info["capacity_utilization_pct"] == 93.89
    assert peak_info["status"] == "WARNING"

    # Top 5 peaks
    assert len(peak_info["top_peaks"]) == 5
    assert peak_info["top_peaks"][0]["predicted_demand_mw"] == 8450.0

    # Uncertainty bounds
    assert "uncertainty" in peak_info
    assert peak_info["uncertainty"]["peak_lower_bound_mw"] < 8450.0
    assert peak_info["uncertainty"]["peak_upper_bound_mw"] > 8450.0

    # Test add_uncertainty_bounds
    df_with_bounds = add_uncertainty_bounds(forecast_df, uncertainty_mape=3.0)
    assert "predicted_lower_mw" in df_with_bounds.columns
    assert "predicted_upper_mw" in df_with_bounds.columns
    assert (df_with_bounds["predicted_lower_mw"] <= df_with_bounds["predicted_demand_mw"]).all()
    assert (df_with_bounds["predicted_upper_mw"] >= df_with_bounds["predicted_demand_mw"]).all()
