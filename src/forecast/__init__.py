"""Forecast generation and peak analysis package for Delhi Electricity Demand Prediction System."""

from src.forecast.predict_24h import predict_next_24h, generate_24h_forecast
from src.forecast.predict_7d import predict_next_7d, generate_7d_forecast, aggregate_daily_forecast
from src.forecast.analyze import detect_peaks, add_uncertainty_bounds, get_top_peaks

__all__ = [
    "predict_next_24h",
    "generate_24h_forecast",
    "predict_next_7d",
    "generate_7d_forecast",
    "aggregate_daily_forecast",
    "detect_peaks",
    "add_uncertainty_bounds",
    "get_top_peaks",
]
