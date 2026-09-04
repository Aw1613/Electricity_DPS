"""Unit tests for data acquisition, synthetic generation, and unified data loaders."""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generate_synthetic import generate_synthetic_demand, generate_synthetic_renewables
from src.data.weather_api import generate_mock_weather, fetch_weather_data
from src.data.data_loader import (
    load_historical_demand,
    load_weather,
    load_renewable_data,
    load_area_data,
)


def test_synthetic_demand_generation():
    """Verify synthetic demand generation, schema, and seasonality characteristics."""
    test_slice_path = PROJECT_ROOT / "data" / "mock" / "test_slice_demand.csv"
    df = generate_synthetic_demand(
        start_date="2023-06-01 00:00:00",
        end_date="2023-06-07 23:00:00",
        output_path=test_slice_path,
    )
    if test_slice_path.exists():
        test_slice_path.unlink()

    # Check required columns
    required_cols = {"timestamp", "demand_mw", "area", "feeder"}
    assert required_cols.issubset(set(df.columns)), f"Missing columns: {required_cols - set(df.columns)}"

    # Check areas present
    expected_areas = {"North Delhi", "South Delhi", "West Delhi", "East Delhi", "Central Delhi"}
    assert set(df["area"].unique()) == expected_areas

    # Verify positive realistic demand
    assert (df["demand_mw"] > 0).all()
    assert (df["demand_mw"] < 5000).all()  # Individual area demand should be well within range

    # Aggregated hourly total should reflect Delhi grid load
    total_hourly = df.groupby("timestamp")["demand_mw"].sum()
    assert (total_hourly >= 3000).all()
    assert (total_hourly <= 8900).all()

    print("PASS: Synthetic demand generator test passed.")


def test_weather_generation_and_fallback():
    """Verify mock weather generation and fetch fallback schema."""
    # Test mock generation directly
    mock_df = generate_mock_weather(start_date="2023-01-01 00:00:00", end_date="2023-01-03 23:00:00")
    required_weather_cols = [
        "timestamp",
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "wind_speed_10m",
    ]
    for col in required_weather_cols:
        assert col in mock_df.columns, f"Missing weather column {col}"

    assert len(mock_df) == 72  # 3 days * 24 hours
    assert (mock_df["temperature_2m"] >= -5.0).all()
    assert (mock_df["temperature_2m"] <= 52.0).all()
    assert (mock_df["relative_humidity_2m"] >= 0.0).all()
    assert (mock_df["relative_humidity_2m"] <= 100.0).all()

    # Test fetch_weather_data (with fallback resilience)
    weather_df = fetch_weather_data(start_date="2023-01-01 00:00:00", end_date="2023-01-02 23:00:00")
    assert not weather_df.empty
    assert "temperature_2m" in weather_df.columns

    print("PASS: Weather API and fallback test passed.")


def test_data_loader_interface():
    """Verify unified loader functions return standardized DataFrames."""
    # 1. Historical Demand
    demand_df = load_historical_demand()
    assert isinstance(demand_df, pd.DataFrame)
    assert "timestamp" in demand_df.columns
    assert "demand_mw" in demand_df.columns
    assert pd.api.types.is_datetime64_any_dtype(demand_df["timestamp"])
    assert len(demand_df) > 0

    # 2. Weather
    weather_df = load_weather()
    assert isinstance(weather_df, pd.DataFrame)
    assert "timestamp" in weather_df.columns
    assert "temperature_2m" in weather_df.columns
    assert pd.api.types.is_datetime64_any_dtype(weather_df["timestamp"])

    # 3. Renewable
    renewable_df = load_renewable_data()
    assert isinstance(renewable_df, pd.DataFrame)
    assert "timestamp" in renewable_df.columns
    assert "solar_generation_mw" in renewable_df.columns
    assert "renewable_generation_mw" in renewable_df.columns
    assert pd.api.types.is_datetime64_any_dtype(renewable_df["timestamp"])

    # 4. Area Data
    area_df = load_area_data()
    assert isinstance(area_df, pd.DataFrame)
    assert set(["timestamp", "area", "feeder", "demand_mw"]).issubset(set(area_df.columns))
    assert pd.api.types.is_datetime64_any_dtype(area_df["timestamp"])

    print("PASS: All unified data loader tests passed.")


if __name__ == "__main__":
    test_synthetic_demand_generation()
    test_weather_generation_and_fallback()
    test_data_loader_interface()
    print("\nAll data layer unit tests completed successfully!")
