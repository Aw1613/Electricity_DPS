"""Next 24-hour electricity demand forecasting engine for Delhi (Task 5.1).

Generates hourly predictions for the next 24 hours by aligning recent historical
demand with upcoming weather forecasts and evaluating the trained ML model.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.data.data_loader import load_historical_demand, load_weather
from src.features.build_features import get_feature_columns, get_season_code
from src.models.predict import load_demand_model, predict_demand

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def prepare_recent_history(
    historical_df: Optional[pd.DataFrame] = None,
    min_hours: int = 168,
) -> pd.DataFrame:
    """Ensure at least min_hours of continuous historical demand are available."""
    if historical_df is None:
        historical_df = load_historical_demand()

    df = historical_df.copy()
    if "timestamp" not in df.columns or "demand_mw" not in df.columns:
        raise ValueError("historical_df must contain 'timestamp' and 'demand_mw' columns.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < min_hours:
        # Pad by repeating earliest records if history is somehow shorter than 168h
        padding_needed = min_hours - len(df)
        first_row = df.iloc[0]
        padding = [
            {
                "timestamp": first_row["timestamp"] - timedelta(hours=i + 1),
                "demand_mw": float(first_row["demand_mw"]),
            }
            for i in range(padding_needed)
        ]
        pad_df = pd.DataFrame(padding).sort_values("timestamp")
        df = pd.concat([pad_df, df], ignore_index=True)

    return df


def prepare_weather_forecast(
    start_timestamp: pd.Timestamp,
    horizon_hours: int = 24,
    weather_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Align or generate weather forecast attributes for the upcoming horizon."""
    end_timestamp = start_timestamp + timedelta(hours=horizon_hours - 1)

    if weather_df is None:
        try:
            weather_df = load_weather(
                start_date=start_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                end_date=(end_timestamp + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            )
        except Exception:
            weather_df = None

    # Filter weather covering the requested window
    records = []
    if weather_df is not None and not weather_df.empty:
        w_df = weather_df.copy()
        w_df["ts"] = pd.to_datetime(w_df["timestamp"])
        window_weather = w_df[(w_df["ts"] >= start_timestamp) & (w_df["ts"] <= end_timestamp)]
        if len(window_weather) >= horizon_hours:
            for _, row in window_weather.iloc[:horizon_hours].iterrows():
                temp = float(row.get("temperature", row.get("temperature_2m", 32.0)))
                hum = float(row.get("humidity", row.get("relative_humidity_2m", 45.0)))
                app_temp = float(row.get("apparent_temperature", temp))
                precip = float(row.get("precipitation", 0.0))
                wind = float(row.get("wind_speed", row.get("wind_speed_10m", 12.0)))
                records.append({
                    "timestamp": pd.to_datetime(row["ts"]),
                    "temperature": temp,
                    "humidity": hum,
                    "apparent_temperature": app_temp,
                    "precipitation": precip,
                    "wind_speed": wind,
                })

    # If weather data was incomplete or offline, generate realistic profile
    if len(records) < horizon_hours:
        records.clear()
        for i in range(horizon_hours):
            curr_ts = start_timestamp + timedelta(hours=i)
            hour = curr_ts.hour
            # Diurnal temperature cycle for Delhi (30°C to 42°C in summer peak)
            temp = 34.0 + 7.0 * np.sin((hour - 9) * np.pi / 12)
            hum = max(20.0, min(80.0, 48.0 - 15.0 * np.sin((hour - 9) * np.pi / 12)))
            app_temp = temp + (hum / 100.0) * max(0.0, temp - 24.0) * 0.4
            records.append({
                "timestamp": curr_ts,
                "temperature": round(float(temp), 2),
                "humidity": round(float(hum), 1),
                "apparent_temperature": round(float(app_temp), 2),
                "precipitation": 0.0,
                "wind_speed": 10.5,
            })

    return pd.DataFrame(records)


def predict_next_24h(
    historical_df: Optional[pd.DataFrame] = None,
    weather_forecast_df: Optional[pd.DataFrame] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate hourly electricity demand forecast for the next 24 hours.

    Args:
        historical_df: Historical DataFrame with ['timestamp', 'demand_mw'] (at least 168h).
        weather_forecast_df: Optional future 24h weather DataFrame.
        model_path: Optional custom path to demand_model.joblib.

    Returns:
        DataFrame with 24 rows containing timestamp, predicted_demand_mw, weather, and time components.
    """
    history = prepare_recent_history(historical_df, min_hours=168)
    last_timestamp = history["timestamp"].iloc[-1]
    forecast_start = last_timestamp + timedelta(hours=1)

    weather_24h = prepare_weather_forecast(
        start_timestamp=forecast_start,
        horizon_hours=24,
        weather_df=weather_forecast_df,
    )

    model, feature_names, metadata = load_demand_model(model_path=model_path)
    expected_features = feature_names if feature_names else get_feature_columns()

    demand_series = list(history["demand_mw"].values)
    forecast_results: List[Dict[str, Any]] = []

    for i in range(24):
        w_row = weather_24h.iloc[i]
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

        # Construct single feature row matching model schema
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
            # Lags strictly referencing past observed or recursively predicted values
            "lag_1h": float(demand_series[-1]),
            "lag_2h": float(demand_series[-2]),
            "lag_3h": float(demand_series[-3]),
            "lag_24h": float(demand_series[-24]),
            "lag_48h": float(demand_series[-48]),
            "lag_168h": float(demand_series[-168]),
            # Rolling features on shifted demand series
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
        pred_val = float(predict_demand(features, model_path=model_path)[0])
        # Plausibility clamping (Delhi grid bounds ~1,500 MW to 12,000 MW)
        pred_val = float(np.clip(pred_val, 1500.0, 12000.0))

        # Append to historical buffer for multi-step recursive lookahead
        demand_series.append(pred_val)

        forecast_results.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_demand_mw": round(pred_val, 2),
            "temperature": round(temp, 2),
            "humidity": round(hum, 1),
            "apparent_temperature": round(app_temp, 2),
            "precipitation": round(precip, 2),
            "wind_speed": round(wind, 2),
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
        })

    return pd.DataFrame(forecast_results)


# Alias for clean modular imports
generate_24h_forecast = predict_next_24h


if __name__ == "__main__":
    print("Generating Next 24-Hour Forecast for Delhi Electricity Grid...")
    forecast_df = predict_next_24h()
    print(f"Generated {len(forecast_df)} hourly forecasts:")
    print(forecast_df[["timestamp", "predicted_demand_mw", "temperature", "hour"]].head())
    print(f"\n24h Forecast Peak: {forecast_df['predicted_demand_mw'].max()} MW at {forecast_df.loc[forecast_df['predicted_demand_mw'].idxmax(), 'timestamp']}")
