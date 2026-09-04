"""Unit tests for preprocessing and validation utilities."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validator import run_data_validation, validate_dataset
from src.data.preprocessing import (
    normalize_to_kolkata_timestamp,
    clean_and_resample_hourly,
    align_demand_and_weather,
    preprocess_and_save_dataset,
)


def test_validator_clean_data():
    """Verify validator accepts clean, continuous data with expected columns."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-05-01 00:00:00", periods=48, freq="1h"),
        "demand_mw": np.linspace(4500, 7500, 48),
        "temperature": np.linspace(25.0, 42.0, 48),
    })
    result = run_data_validation(df)
    assert result.is_valid, f"Validation should pass but got errors: {result.errors}"
    assert len(result.errors) == 0

    valid, msgs = validate_dataset(df)
    assert valid is True
    print("PASS: Validator clean data test passed.")


def test_validator_corrupted_cases():
    """Verify validator catches non-positive demand, extreme temps, gaps, and missing columns."""
    base_ts = pd.date_range("2023-05-01 00:00:00", periods=24, freq="1h")

    # 1. Non-positive demand
    bad_demand_df = pd.DataFrame({
        "timestamp": base_ts,
        "demand_mw": [5000] * 23 + [-10],  # Negative demand
        "temperature": [30.0] * 24,
    })
    res1 = run_data_validation(bad_demand_df)
    assert not res1.is_valid
    assert any("non-positive" in e for e in res1.errors)

    # 2. Extreme temperature (> 55°C or < -5°C)
    bad_temp_df = pd.DataFrame({
        "timestamp": base_ts,
        "demand_mw": [5000] * 24,
        "temperature": [30.0] * 23 + [68.0],  # 68°C is unrealistic for Delhi
    })
    res2 = run_data_validation(bad_temp_df)
    assert not res2.is_valid
    assert any("Temperature out of reasonable Delhi range" in e for e in res2.errors)

    # 3. Temporal gap in timestamps
    gap_ts = list(pd.date_range("2023-05-01 00:00:00", periods=5, freq="1h")) + list(
        pd.date_range("2023-05-01 08:00:00", periods=5, freq="1h")
    )
    gap_df = pd.DataFrame({
        "timestamp": gap_ts,
        "demand_mw": [5000] * len(gap_ts),
        "temperature": [30.0] * len(gap_ts),
    })
    res3 = run_data_validation(gap_df)
    assert not res3.is_valid
    assert any("Non-continuous timestamps" in e for e in res3.errors)

    # 4. Missing required column
    missing_col_df = pd.DataFrame({
        "timestamp": base_ts,
        "demand_mw": [5000] * 24,
    })
    res4 = run_data_validation(missing_col_df)
    assert not res4.is_valid
    assert any("Missing required 'temperature' column" in e for e in res4.errors)

    print("PASS: Validator corrupted cases tests passed.")


def test_timestamp_normalization():
    """Verify timestamp normalization to Asia/Kolkata."""
    # Test timezone-naive series
    naive_series = pd.Series(["2023-06-01 12:00:00", "2023-06-01 13:00:00"])
    normalized_naive = normalize_to_kolkata_timestamp(naive_series)
    assert pd.api.types.is_datetime64_any_dtype(normalized_naive)

    # Test timezone-aware UTC series
    utc_series = pd.Series(pd.date_range("2023-06-01 00:00:00", periods=3, freq="1h", tz="UTC"))
    normalized_utc = normalize_to_kolkata_timestamp(utc_series)
    # UTC 00:00 should be 05:30 in Asia/Kolkata
    assert normalized_utc.iloc[0].hour == 5
    assert normalized_utc.iloc[0].minute == 30

    print("PASS: Timestamp normalization test passed.")


def test_clean_and_resample_hourly():
    """Verify gap interpolation and 1h resampling."""
    # Create series with a 2-hour missing gap
    ts = [
        pd.Timestamp("2023-06-01 10:00:00"),
        pd.Timestamp("2023-06-01 11:00:00"),
        pd.Timestamp("2023-06-01 14:00:00"),  # Missing 12:00 and 13:00
    ]
    df = pd.DataFrame({
        "timestamp": ts,
        "demand_mw": [5000.0, 5200.0, 5800.0],
    })

    resampled = clean_and_resample_hourly(df, max_interpolation_gap=3)
    assert len(resampled) == 5  # 10:00, 11:00, 12:00, 13:00, 14:00
    assert not resampled["demand_mw"].isna().any()
    # 12:00 should be linearly interpolated between 5200 and 5800 (i.e. 5400)
    assert np.isclose(resampled.loc[resampled["timestamp"] == "2023-06-01 12:00:00", "demand_mw"].iloc[0], 5400.0)

    print("PASS: Clean and resample hourly test passed.")


def test_preprocess_and_save_pipeline():
    """Verify full end-to-end preprocessing, alignment, validation, and file saving."""
    test_output_path = PROJECT_ROOT / "data" / "processed" / "test_merged_features.csv"
    try:
        merged_df, val_result = preprocess_and_save_dataset(output_path=test_output_path)

        assert isinstance(merged_df, pd.DataFrame)
        assert not merged_df.empty
        assert "timestamp" in merged_df.columns
        assert "demand_mw" in merged_df.columns
        assert "temperature" in merged_df.columns
        assert val_result.is_valid, f"Pipeline validation failed: {val_result.errors}"
        assert test_output_path.exists()
        print("PASS: Full preprocessing and saving pipeline test passed.")
    finally:
        if test_output_path.exists():
            test_output_path.unlink()


if __name__ == "__main__":
    test_validator_clean_data()
    test_validator_corrupted_cases()
    test_timestamp_normalization()
    test_clean_and_resample_hourly()
    test_preprocess_and_save_pipeline()
    print("\nAll preprocessing and validator unit tests passed successfully!")
