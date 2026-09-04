"""Unified data loading utilities for Delhi Electricity Demand Prediction System."""

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.data.generate_synthetic import (
    generate_synthetic_demand,
    generate_synthetic_renewables,
)
from src.data.weather_api import fetch_weather_data

# Directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MOCK_DIR = DATA_DIR / "mock"


def load_historical_demand(
    filepath: Optional[Union[str, Path]] = None,
    aggregate_total: bool = True,
    demo_mode: bool = False,
) -> pd.DataFrame:
    """Load historical electricity demand data into a standardized DataFrame.

    Checks custom filepath -> raw data folder -> mock dataset (generating if missing).

    Args:
        filepath: Optional path to specific demand CSV.
        aggregate_total: If True and area/feeder columns exist, aggregates total demand_mw by timestamp.
        demo_mode: If True, bypasses raw files and uses local synthetic mock data.

    Returns:
        DataFrame containing sorted ['timestamp', 'demand_mw'] (or with area/feeder if aggregate_total=False).
    """
    if demo_mode:
        target_path = MOCK_DIR / "synthetic_demand.csv"
        if not target_path.exists():
            generate_synthetic_demand(output_path=target_path)
    elif filepath is not None:
        target_path = Path(filepath)
    elif (RAW_DIR / "load.csv").exists():
        target_path = RAW_DIR / "load.csv"
    elif (RAW_DIR / "historical_demand.csv").exists():
        target_path = RAW_DIR / "historical_demand.csv"
    elif (RAW_DIR / "powerdemand_5min_2021_to_2024_with weather.csv").exists():
        target_path = RAW_DIR / "powerdemand_5min_2021_to_2024_with weather.csv"
    elif (MOCK_DIR / "synthetic_demand.csv").exists():
        target_path = MOCK_DIR / "synthetic_demand.csv"
    else:
        # Generate synthetic fallback dataset
        target_path = MOCK_DIR / "synthetic_demand.csv"
        generate_synthetic_demand(output_path=target_path)

    df = pd.read_csv(target_path)

    # Standardize column names (lower case, strip whitespace)
    df.columns = [c.strip().lower() for c in df.columns]

    # Handle various common timestamp naming conventions
    time_col = None
    for col in ["timestamp", "datetime", "date_time", "time", "date"]:
        if col in df.columns:
            time_col = col
            break
    if time_col and time_col != "timestamp":
        df = df.rename(columns={time_col: "timestamp"})

    # Handle demand column variations
    demand_col = None
    for col in ["demand_mw", "total_demand_mw", "load_mw", "demand", "load", "power demand", "power_demand", "power"]:
        if col in df.columns:
            demand_col = col
            break
    if demand_col and demand_col != "demand_mw":
        df = df.rename(columns={demand_col: "demand_mw"})

    # Ensure timestamp parsing
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")

    # Aggregate by timestamp if multiple areas are present and total demand is requested
    if aggregate_total and "area" in df.columns:
        df = (
            df.groupby("timestamp", as_index=False)["demand_mw"]
            .sum()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Automatically resample sub-hourly data (e.g. 5-min intervals) to hourly mean
    if len(df) > 1:
        time_diffs = df["timestamp"].diff().dropna()
        if not time_diffs.empty and time_diffs.median() < pd.Timedelta(hours=1):
            numeric_cols = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
            df_resampled = df.set_index("timestamp")[numeric_cols].resample("1h").mean()
            df = df_resampled.interpolate(method="linear").ffill().bfill().reset_index()

    return df


def load_weather(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filepath: Optional[Union[str, Path]] = None,
    demo_mode: bool = False,
) -> pd.DataFrame:
    """Load weather data for Delhi into a standardized DataFrame.

    Checks custom filepath -> raw data folder -> Open-Meteo API / cache / mock generator.

    Returns:
        DataFrame with ['timestamp', 'temperature_2m', 'relative_humidity_2m',
                        'apparent_temperature', 'precipitation', 'wind_speed_10m']
    """
    if demo_mode:
        df = fetch_weather_data(start_date=start_date, end_date=end_date, demo_mode=True)
    elif filepath is not None:
        df = pd.read_csv(Path(filepath))
    elif (RAW_DIR / "weather.csv").exists():
        df = pd.read_csv(RAW_DIR / "weather.csv")
    else:
        df = fetch_weather_data(start_date=start_date, end_date=end_date, demo_mode=False)

    df.columns = [c.strip().lower() for c in df.columns]

    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    numeric_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "wind_speed_10m",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("timestamp").reset_index(drop=True)


def load_renewable_data(
    filepath: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load renewable (solar / wind) generation data into a standardized DataFrame.

    Returns:
        DataFrame with ['timestamp', 'solar_generation_mw', 'renewable_generation_mw']
    """
    if filepath is not None:
        target_path = Path(filepath)
    elif (RAW_DIR / "renewable.csv").exists():
        target_path = RAW_DIR / "renewable.csv"
    elif (MOCK_DIR / "synthetic_renewable.csv").exists():
        target_path = MOCK_DIR / "synthetic_renewable.csv"
    else:
        target_path = MOCK_DIR / "synthetic_renewable.csv"
        generate_synthetic_renewables(output_path=target_path)

    df = pd.read_csv(target_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in ["solar_generation_mw", "renewable_generation_mw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("timestamp").reset_index(drop=True)


def load_area_data(
    filepath: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load granular area and feeder level electricity demand data.

    Returns:
        DataFrame with ['timestamp', 'area', 'feeder', 'demand_mw']
    """
    if filepath is not None:
        target_path = Path(filepath)
    elif (RAW_DIR / "area_demand.csv").exists():
        target_path = RAW_DIR / "area_demand.csv"
    elif (MOCK_DIR / "synthetic_demand.csv").exists():
        target_path = MOCK_DIR / "synthetic_demand.csv"
    else:
        target_path = MOCK_DIR / "synthetic_demand.csv"
        generate_synthetic_demand(output_path=target_path)

    df = pd.read_csv(target_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "demand_mw" in df.columns:
        df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")

    expected_cols = ["timestamp", "area", "feeder", "demand_mw"]
    cols_present = [c for c in expected_cols if c in df.columns]

    return df[cols_present].sort_values(["timestamp", "area"]).reset_index(drop=True)
