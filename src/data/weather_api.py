"""Open-Meteo weather API client with local caching and offline fallback for Delhi."""

import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import requests

# Delhi geographic coordinates
DELHI_LATITUDE = 28.6139
DELHI_LONGITUDE = 77.2090
DELHI_TIMEZONE = "Asia/Kolkata"

# Storage paths
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
MOCK_DIR = DATA_DIR / "mock"

CACHE_FILE = RAW_DIR / "weather_cache.csv"
MOCK_FILE = MOCK_DIR / "weather_mock.csv"

# Required weather variables
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
]


def generate_mock_weather(
    start_date: str = "2023-01-01 00:00:00",
    end_date: str = "2023-12-31 23:00:00",
    output_path: Optional[Path] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic Delhi hourly weather data for offline fallback."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")

    records = []
    for ts in timestamps:
        hour = ts.hour
        day_of_year = ts.dayofyear

        # Base seasonal temperature curve (Delhi: hot in May-June, cold in Dec-Jan)
        # Peak around day 165 (mid-June), min around day 10 (early Jan)
        seasonal_temp = 25.0 + 13.0 * np.sin(2 * np.pi * (day_of_year - 95) / 365.25)

        # Diurnal temperature swing (~10-12°C swing, min at 6am, max at 15pm)
        daily_temp_cycle = 6.0 * np.sin((hour - 9) * np.pi / 12)
        temp_noise = rng.normal(0, 0.8)
        temperature = round(float(seasonal_temp + daily_temp_cycle + temp_noise), 2)

        # Humidity (inversely related to temperature, but very high during monsoon days 190-250)
        is_monsoon = 190 <= day_of_year <= 250
        base_humidity = 75.0 if is_monsoon else 45.0
        daily_humidity_cycle = -15.0 * np.sin((hour - 9) * np.pi / 12)
        humidity = round(float(np.clip(base_humidity + daily_humidity_cycle + rng.normal(0, 5.0), 15.0, 98.0)), 1)

        # Apparent temperature (heat index: increases drastically with humidity in summer)
        if temperature > 27.0:
            heat_index_bonus = (humidity / 100.0) * (temperature - 20.0) * 0.45
            apparent_temp = round(temperature + heat_index_bonus, 2)
        elif temperature < 15.0:
            wind_chill_penalty = 1.5 + (10.0 - temperature) * 0.1
            apparent_temp = round(temperature - wind_chill_penalty, 2)
        else:
            apparent_temp = temperature

        # Precipitation (mostly during monsoon season)
        if is_monsoon and rng.random() < 0.22:
            precipitation = round(float(rng.exponential(2.5)), 2)
        elif rng.random() < 0.02:
            precipitation = round(float(rng.uniform(0.1, 1.5)), 2)
        else:
            precipitation = 0.0

        # Wind speed (km/h, typically 5 to 25 km/h in Delhi)
        wind_speed = round(float(np.clip(rng.gamma(shape=3.0, scale=3.5), 2.0, 35.0)), 2)

        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature_2m": temperature,
            "relative_humidity_2m": humidity,
            "apparent_temperature": apparent_temp,
            "precipitation": precipitation,
            "wind_speed_10m": wind_speed,
        })

    df = pd.DataFrame(records)

    target_path = output_path or MOCK_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return df


def fetch_weather_from_api(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeout: int = 8,
) -> pd.DataFrame:
    """Fetch hourly weather data for Delhi directly from Open-Meteo API.

    Uses forecast endpoint for current/recent dates, and archive endpoint for historical ranges.
    """
    hourly_vars = ",".join(WEATHER_VARIABLES)

    if start_date and end_date:
        # Extract YYYY-MM-DD from provided strings
        start_day = str(start_date)[:10]
        end_day = str(end_date)[:10]
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={DELHI_LATITUDE}&longitude={DELHI_LONGITUDE}&"
            f"start_date={start_day}&end_date={end_day}&"
            f"hourly={hourly_vars}&timezone={DELHI_TIMEZONE}"
        )
    else:
        # Fetch current forecast + past 30 days
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={DELHI_LATITUDE}&longitude={DELHI_LONGITUDE}&"
            f"hourly={hourly_vars}&past_days=30&forecast_days=7&timezone={DELHI_TIMEZONE}"
        )

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    hourly = data.get("hourly", {})
    if "time" not in hourly:
        raise ValueError("Invalid Open-Meteo response structure: 'hourly.time' missing.")

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]).strftime("%Y-%m-%d %H:%M:%S"),
        "temperature_2m": hourly.get("temperature_2m"),
        "relative_humidity_2m": hourly.get("relative_humidity_2m"),
        "apparent_temperature": hourly.get("apparent_temperature"),
        "precipitation": hourly.get("precipitation"),
        "wind_speed_10m": hourly.get("wind_speed_10m"),
    })

    return df


def fetch_weather_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
    timeout: int = 6,
) -> pd.DataFrame:
    """Fetch Delhi weather data with caching and fallback.

    1. Checks local cache first (unless force_refresh is True).
    2. Tries fetching fresh data from Open-Meteo API.
    3. On network or API failure, falls back to cache or generates realistic mock data.
    """
    cache_target = cache_path or CACHE_FILE

    # If cache exists and not forcing refresh, check cache coverage
    if not force_refresh and cache_target.exists():
        try:
            cached_df = pd.read_csv(cache_target)
            if all(col in cached_df.columns for col in ["timestamp"] + WEATHER_VARIABLES):
                # If date filtering requested, filter cached
                cached_df["ts"] = pd.to_datetime(cached_df["timestamp"])
                if start_date:
                    cached_df = cached_df[cached_df["ts"] >= pd.to_datetime(start_date)]
                if end_date:
                    cached_df = cached_df[cached_df["ts"] <= pd.to_datetime(end_date)]
                cached_df = cached_df.drop(columns=["ts"])

                if len(cached_df) > 0:
                    return cached_df
        except Exception:
            pass  # Fall through to API or fallback

    # Attempt live API fetch
    try:
        df = fetch_weather_from_api(start_date=start_date, end_date=end_date, timeout=timeout)
        # Save successful fetch to cache
        cache_target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_target, index=False)
        return df
    except Exception as e:
        print(f"Notice: Open-Meteo API unavailable ({e}). Falling back to local cache or mock weather.")

    # Fallback to existing cache file if present
    if cache_target.exists():
        try:
            return pd.read_csv(cache_target)
        except Exception:
            pass

    # Fallback to mock file or generate mock weather
    if MOCK_FILE.exists():
        try:
            return pd.read_csv(MOCK_FILE)
        except Exception:
            pass

    # Final fallback: generate realistic synthetic weather
    s_date = start_date or "2023-01-01 00:00:00"
    e_date = end_date or "2023-12-31 23:00:00"
    return generate_mock_weather(start_date=s_date, end_date=e_date, output_path=cache_target)


if __name__ == "__main__":
    weather_df = fetch_weather_data()
    print("Weather data retrieved successfully:")
    print(weather_df.head())
