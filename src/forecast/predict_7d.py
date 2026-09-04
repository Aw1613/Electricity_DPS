"""Next 7-day multi-step recursive electricity demand forecasting engine for Delhi (Task 5.2).

Performs 168-hour sequential autoregressive forecasting where each step's prediction
is recursively fed back as lagged observations (lag_1h, lag_24h, lag_168h, and rolling stats)
to forecast the full 7-day horizon.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.data.data_loader import load_historical_demand, load_weather
from src.features.build_features import get_feature_columns, get_season_code
from src.forecast.predict_24h import prepare_recent_history, prepare_weather_forecast
from src.models.predict import load_demand_model, predict_demand

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def predict_next_7d(
    historical_df: Optional[pd.DataFrame] = None,
    weather_forecast_df: Optional[pd.DataFrame] = None,
    model_path: Optional[Union[str, Path]] = None,
    horizon_hours: int = 168,
) -> pd.DataFrame:
    """Generate multi-step recursive demand forecast for the next 7 days (168 hours).

    Args:
        historical_df: Historical DataFrame with ['timestamp', 'demand_mw'] (at least 168h).
        weather_forecast_df: Optional 7-day future weather DataFrame.
        model_path: Optional custom path to demand_model.joblib.
        horizon_hours: Forecasting horizon in hours (defaults to 168 for 7 days).

    Returns:
        DataFrame containing 168 hourly predictions with timestamp, predicted_demand_mw,
        and weather/calendar features.
    """
    # 1. Prepare historical buffer (minimum 168 hours to populate lag_168h)
    history = prepare_recent_history(historical_df, min_hours=168)
    last_timestamp = history["timestamp"].iloc[-1]
    forecast_start = last_timestamp + timedelta(hours=1)

    # 2. Prepare 7-day hourly weather forecast
    weather_7d = prepare_weather_forecast(
        start_timestamp=forecast_start,
        horizon_hours=horizon_hours,
        weather_df=weather_forecast_df,
    )

    # 3. Load model and schema
    model, feature_names, metadata = load_demand_model(model_path=model_path)
    expected_features = feature_names if feature_names else get_feature_columns()

    # Maintain dynamic demand series starting with historical observations
    demand_series: List[float] = list(history["demand_mw"].values)
    forecast_records: List[Dict[str, Any]] = []

    # 4. Multi-step recursive rollout across all 168 hours
    for step in range(horizon_hours):
        w_row = weather_7d.iloc[step]
        ts = pd.to_datetime(w_row["timestamp"])

        hour = ts.hour
        day_of_week = ts.dayofweek
        day_of_month = ts.day
        month = ts.month
        is_weekend = int(day_of_week >= 5)
        season = get_season_code(month)

        temp = float(w_row["temperature"])
        hum = float(w_row["humidity"])
        app_temp = float(w_row["apparent_temperature"])
        precip = float(w_row["precipitation"])
        wind = float(w_row["wind_speed"])

        # Dynamically compute recursive lag features:
        # For step 0: lag_1h is real historical observation t_0.
        # For step 1: lag_1h is prediction t_1.
        # For step 24: lag_24h is prediction t_0+1, etc.
        features: Dict[str, Any] = {
            "hour": hour,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "month": month,
            "is_weekend": is_weekend,
            "season": season,
            "temperature": temp,
            "humidity": hum,
            "apparent_temperature": app_temp,
            "precipitation": precip,
            "wind_speed": wind,
            # Lags
            "lag_1h": float(demand_series[-1]),
            "lag_2h": float(demand_series[-2]),
            "lag_3h": float(demand_series[-3]),
            "lag_24h": float(demand_series[-24]),
            "lag_48h": float(demand_series[-48]),
            "lag_168h": float(demand_series[-168]),
            # Rolling stats on shifted buffer
            "rolling_mean_3h": float(np.mean(demand_series[-3:])),
            "rolling_mean_6h": float(np.mean(demand_series[-6:])),
            "rolling_mean_24h": float(np.mean(demand_series[-24:])),
            "rolling_max_24h": float(np.max(demand_series[-24:])),
            # Non-linear weather interactions
            "temperature_squared": float(temp ** 2),
            "temperature_x_hour": float(temp * hour),
            "temperature_x_weekend": float(temp * is_weekend),
        }

        # Predict next hour demand
        pred_mw = float(predict_demand(features, model_path=model_path)[0])
        # Safeguard within realistic Delhi operational envelope
        pred_mw = float(np.clip(pred_mw, 1500.0, 12000.0))

        # Crucial recursive step: append prediction to series so subsequent hours use it
        demand_series.append(pred_mw)

        forecast_records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_demand_mw": round(pred_mw, 2),
            "temperature": round(temp, 2),
            "humidity": round(hum, 1),
            "apparent_temperature": round(app_temp, 2),
            "precipitation": round(precip, 2),
            "wind_speed": round(wind, 2),
            "hour": hour,
            "day_of_week": day_of_week,
            "day_name": ts.strftime("%A"),
            "date": ts.strftime("%Y-%m-%d"),
            "is_weekend": is_weekend,
            "step": step + 1,
        })

    return pd.DataFrame(forecast_records)


def aggregate_daily_forecast(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 168-hour forecast into actionable daily summaries for grid planning."""
    df = forecast_df.copy()
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
    if "day_name" not in df.columns:
        df["day_name"] = pd.to_datetime(df["timestamp"]).dt.strftime("%A")

    daily_rows = []
    for date_val, group in df.groupby("date", sort=False):
        peak_idx = group["predicted_demand_mw"].idxmax()
        peak_row = group.loc[peak_idx]

        daily_rows.append({
            "date": date_val,
            "day_name": group["day_name"].iloc[0],
            "peak_demand_mw": round(float(group["predicted_demand_mw"].max()), 2),
            "peak_hour": int(peak_row["hour"]),
            "peak_timestamp": str(peak_row["timestamp"]),
            "min_demand_mw": round(float(group["predicted_demand_mw"].min()), 2),
            "mean_demand_mw": round(float(group["predicted_demand_mw"].mean()), 2),
            "max_temperature_c": round(float(group["temperature"].max()), 1),
            "avg_temperature_c": round(float(group["temperature"].mean()), 1),
        })

    return pd.DataFrame(daily_rows)


# Alias for clean modular imports
generate_7d_forecast = predict_next_7d


if __name__ == "__main__":
    print("Executing 7-Day Recursive Forecast for Delhi Electricity Grid...")
    df_7d = predict_next_7d()
    print(f"Generated {len(df_7d)} hourly recursive predictions.")
    print("\n--- Daily Aggregated Forecast Summary ---")
    daily_summary = aggregate_daily_forecast(df_7d)
    print(daily_summary.to_string(index=False))
