"""Unit tests for feature engineering and data leakage prevention."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (
    build_features,
    get_feature_columns,
    add_calendar_features,
    add_demand_lag_features,
    add_rolling_features,
    add_weather_interaction_features,
    get_season_code,
)


def create_sample_preprocessed_df(n_hours: int = 300) -> pd.DataFrame:
    """Generate a continuous test DataFrame."""
    timestamps = pd.date_range(start="2023-01-01 00:00:00", periods=n_hours, freq="1h")
    np.random.seed(42)
    demand = 4000.0 + 1000.0 * np.sin(np.linspace(0, 20 * np.pi, n_hours)) + np.random.normal(0, 50, n_hours)
    temp = 20.0 + 15.0 * np.sin(np.linspace(0, 10 * np.pi, n_hours))

    return pd.DataFrame({
        "timestamp": timestamps,
        "demand_mw": demand,
        "temperature": temp,
        "humidity": np.random.uniform(30, 80, n_hours),
        "apparent_temperature": temp + 2.0,
        "precipitation": np.zeros(n_hours),
        "wind_speed": np.random.uniform(5, 20, n_hours),
    })


def test_calendar_features():
    """Verify calendar feature extraction and season mapping."""
    df = create_sample_preprocessed_df(24)
    out = add_calendar_features(df)

    expected_cols = ["hour", "day_of_week", "day_of_month", "month", "is_weekend", "season"]
    for col in expected_cols:
        assert col in out.columns, f"Missing calendar column: {col}"

    # Hour 0 to 23
    assert list(out["hour"]) == list(range(24))
    # Season code checks
    assert get_season_code(1) == 1  # Winter
    assert get_season_code(5) == 2  # Summer
    assert get_season_code(8) == 3  # Monsoon
    assert get_season_code(10) == 4  # Autumn
    print("PASS: Calendar features test passed.")


def test_demand_lag_leakage_prevention():
    """Verify demand lags strictly reference prior time steps without leakage."""
    df = create_sample_preprocessed_df(200)
    out = add_demand_lag_features(df)

    for i in range(168, 200):
        # Current row demand must NOT equal any lag
        assert out.loc[i, "lag_1h"] == df.loc[i - 1, "demand_mw"]
        assert out.loc[i, "lag_2h"] == df.loc[i - 2, "demand_mw"]
        assert out.loc[i, "lag_3h"] == df.loc[i - 3, "demand_mw"]
        assert out.loc[i, "lag_24h"] == df.loc[i - 24, "demand_mw"]
        assert out.loc[i, "lag_48h"] == df.loc[i - 48, "demand_mw"]
        assert out.loc[i, "lag_168h"] == df.loc[i - 168, "demand_mw"]

    print("PASS: Demand lag features and leakage prevention test passed.")


def test_rolling_statistics_leakage_prevention():
    """Verify rolling statistics NEVER include the current target observation."""
    df = create_sample_preprocessed_df(50)
    out = add_rolling_features(df)

    # Check row 10: rolling_mean_3h must be the mean of rows 7, 8, 9 (NOT row 10!)
    expected_3h_mean = df.loc[7:9, "demand_mw"].mean()
    actual_3h_mean = out.loc[10, "rolling_mean_3h"]
    assert np.isclose(expected_3h_mean, actual_3h_mean), f"Expected {expected_3h_mean}, got {actual_3h_mean}"

    # Check row 25: rolling_mean_24h must be mean of rows 1 to 24 (NOT row 25!)
    expected_24h_mean = df.loc[1:24, "demand_mw"].mean()
    actual_24h_mean = out.loc[25, "rolling_mean_24h"]
    assert np.isclose(expected_24h_mean, actual_24h_mean), f"Expected {expected_24h_mean}, got {actual_24h_mean}"

    # Check row 25: rolling_max_24h must be max of rows 1 to 24 (NOT row 25!)
    expected_24h_max = df.loc[1:24, "demand_mw"].max()
    actual_24h_max = out.loc[25, "rolling_max_24h"]
    assert np.isclose(expected_24h_max, actual_24h_max), f"Expected {expected_24h_max}, got {actual_24h_max}"

    # Explicit corruption test: perturbing target demand at row 10 must NOT change rolling stats at row 10
    df_perturbed = df.copy()
    df_perturbed.loc[10, "demand_mw"] = 999999.0  # extreme spike in target
    out_perturbed = add_rolling_features(df_perturbed)
    assert np.isclose(out.loc[10, "rolling_mean_3h"], out_perturbed.loc[10, "rolling_mean_3h"]), (
        "TARGET LEAKAGE DETECTED: rolling_mean_3h changed when current target was perturbed!"
    )

    print("PASS: Rolling statistics leakage prevention verified.")


def test_weather_interactions():
    """Verify calculation of weather interaction features."""
    df = create_sample_preprocessed_df(24)
    out = add_weather_interaction_features(df)

    for col in ["temperature_squared", "temperature_x_hour", "temperature_x_weekend"]:
        assert col in out.columns

    for i in range(len(df)):
        temp = out.loc[i, "temperature"]
        hour = out.loc[i, "hour"]
        weekend = out.loc[i, "is_weekend"]
        assert np.isclose(out.loc[i, "temperature_squared"], temp ** 2)
        assert np.isclose(out.loc[i, "temperature_x_hour"], temp * hour)
        assert np.isclose(out.loc[i, "temperature_x_weekend"], temp * weekend)

    print("PASS: Weather interaction features test passed.")


def test_build_features_clean_nans():
    """Verify full feature building pipeline and NaN handling."""
    df = create_sample_preprocessed_df(300)
    out = build_features(df, drop_na=True)

    # Because max lag is 168h, exactly 168 rows should be dropped
    assert len(out) == 300 - 168
    # No remaining NaNs
    assert not out.isna().any().any(), "Found unexpected NaNs in feature matrix."

    # Verify all expected model feature columns exist
    feature_cols = get_feature_columns()
    for col in feature_cols:
        assert col in out.columns, f"Feature column missing: {col}"
        assert pd.api.types.is_numeric_dtype(out[col]), f"Feature column {col} must be numeric."

    print("PASS: Full feature matrix pipeline test passed.")


if __name__ == "__main__":
    test_calendar_features()
    test_demand_lag_leakage_prevention()
    test_rolling_statistics_leakage_prevention()
    test_weather_interactions()
    test_build_features_clean_nans()
    print("\nAll feature engineering unit tests completed successfully!")
