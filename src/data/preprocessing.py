"""Data preprocessing and alignment pipeline for Delhi Electricity Demand Prediction System.

Responsibilities:
- Normalize timestamps to Asia/Kolkata timezone.
- Resample time series to an exact hourly resolution (1h).
- Handle missing demand and weather data using linear interpolation and forward-fill.
- Sort records in strict chronological order.
- Align demand and weather records on timestamp into a continuous DataFrame.
- Validate dataset integrity and save to data/processed/merged_features.csv.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.data.data_loader import load_historical_demand, load_weather
from src.data.validator import run_data_validation, ValidationResult

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "merged_features.csv"

DEFAULT_TIMEZONE = "Asia/Kolkata"


def normalize_to_kolkata_timestamp(
    series: pd.Series,
    target_tz: str = DEFAULT_TIMEZONE,
) -> pd.Series:
    """Normalize datetime series to Asia/Kolkata local time.

    If naive, assumes Asia/Kolkata. If aware, converts to Asia/Kolkata.
    Returns timezone-naive timestamps representing local Asia/Kolkata time for consistent modeling.
    """
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        # Assume local Kolkata time if not specified
        ts_kolkata = ts.dt.tz_localize(target_tz, ambiguous="NaT", nonexistent="shift_forward")
    else:
        ts_kolkata = ts.dt.tz_convert(target_tz)

    # Return as naive datetime in Kolkata local time for CSV/modeling compatibility
    return ts_kolkata.dt.tz_localize(None)


def clean_and_resample_hourly(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    max_interpolation_gap: int = 6,
) -> pd.DataFrame:
    """Clean, sort, deduplicate, and resample DataFrame to an exact continuous 1-hour resolution.

    Args:
        df: Input DataFrame containing timestamp and numeric values.
        timestamp_col: Name of the timestamp column.
        max_interpolation_gap: Maximum consecutive missing hours to interpolate.

    Returns:
        Continuous hourly DataFrame with missing values filled.
    """
    if df.empty:
        return df

    cleaned = df.copy()
    cleaned[timestamp_col] = normalize_to_kolkata_timestamp(cleaned[timestamp_col])

    # Deduplicate timestamps by averaging multiple observations in the same hour
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in cleaned.columns if c not in numeric_cols and c != timestamp_col]

    agg_dict = {col: "mean" for col in numeric_cols}
    for col in non_numeric_cols:
        agg_dict[col] = "first"

    if agg_dict:
        cleaned = cleaned.groupby(timestamp_col, as_index=False).agg(agg_dict)
    else:
        cleaned = cleaned.drop_duplicates(subset=[timestamp_col])

    # Sort chronologically
    cleaned = cleaned.sort_values(timestamp_col).reset_index(drop=True)

    # Build exact continuous hourly index
    start_time = cleaned[timestamp_col].min().floor("h")
    end_time = cleaned[timestamp_col].max().ceil("h")
    full_index = pd.date_range(start=start_time, end=end_time, freq="1h", name=timestamp_col)

    # Reindex to full continuous hourly range
    cleaned = cleaned.set_index(timestamp_col).reindex(full_index)

    # Handle missing numeric data via linear interpolation with limits, then ffill/bfill for edges
    for col in numeric_cols:
        if col in cleaned.columns:
            # Linear interpolation for small gaps
            cleaned[col] = cleaned[col].interpolate(
                method="linear",
                limit=max_interpolation_gap,
                limit_direction="both",
            )
            # Forward-fill and backward-fill remaining small edge gaps
            cleaned[col] = cleaned[col].ffill().bfill()

    # Restore timestamp column
    cleaned = cleaned.reset_index().rename(columns={"index": timestamp_col})
    return cleaned


def align_demand_and_weather(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align historical demand and weather dataframes on exact hourly timestamps.

    Standardizes columns:
    - timestamp
    - demand_mw
    - temperature / temperature_2m
    - humidity / relative_humidity_2m
    - apparent_temperature
    - precipitation
    - wind_speed / wind_speed_10m
    """
    # Clean and resample each dataset independently
    d_clean = clean_and_resample_hourly(demand_df)
    w_clean = clean_and_resample_hourly(weather_df)

    # Merge on timestamp
    merged = pd.merge(d_clean, w_clean, on="timestamp", how="inner")

    # If weather did not overlap completely, use outer join and interpolate weather
    if len(merged) < len(d_clean):
        merged = pd.merge(d_clean, w_clean, on="timestamp", how="left")
        # Interpolate weather columns across any gaps
        weather_numeric = [
            c for c in w_clean.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(w_clean[c])
        ]
        for col in weather_numeric:
            if col in merged.columns:
                merged[col] = merged[col].interpolate(method="linear", limit=12, limit_direction="both").ffill().bfill()

    # Standardize column naming aliases for convenience
    if "temperature_2m" in merged.columns and "temperature" not in merged.columns:
        merged["temperature"] = merged["temperature_2m"]
    elif "temperature" in merged.columns and "temperature_2m" not in merged.columns:
        merged["temperature_2m"] = merged["temperature"]

    if "relative_humidity_2m" in merged.columns and "humidity" not in merged.columns:
        merged["humidity"] = merged["relative_humidity_2m"]
    elif "humidity" in merged.columns and "relative_humidity_2m" not in merged.columns:
        merged["relative_humidity_2m"] = merged["humidity"]

    if "wind_speed_10m" in merged.columns and "wind_speed" not in merged.columns:
        merged["wind_speed"] = merged["wind_speed_10m"]
    elif "wind_speed" in merged.columns and "wind_speed_10m" not in merged.columns:
        merged["wind_speed_10m"] = merged["wind_speed"]

    # Final sort
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def preprocess_and_save_dataset(
    demand_df: Optional[pd.DataFrame] = None,
    weather_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, ValidationResult]:
    """Execute full preprocessing pipeline: load -> clean & resample -> align -> validate -> save.

    Returns:
        Tuple of (processed DataFrame, ValidationResult).
    """
    # 1. Load data if not supplied
    if demand_df is None:
        demand_df = load_historical_demand(aggregate_total=True)

    if weather_df is None:
        start_date = str(demand_df["timestamp"].min())
        end_date = str(demand_df["timestamp"].max())
        weather_df = load_weather(start_date=start_date, end_date=end_date)

    # 2. Align and clean
    merged_df = align_demand_and_weather(demand_df, weather_df)

    # 3. Format timestamp to standard string format before saving
    merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # 4. Validate output
    val_result = run_data_validation(merged_df)
    if not val_result.is_valid:
        print(f"Warning: Validation found errors: {val_result.errors}")
    if val_result.warnings:
        print(f"Validation notices: {val_result.warnings}")

    # 5. Save to disk
    target_path = output_path or OUTPUT_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(target_path, index=False)
    print(f"Preprocessing complete: {len(merged_df)} continuous hourly rows saved to {target_path}")

    return merged_df, val_result


if __name__ == "__main__":
    df, result = preprocess_and_save_dataset()
    print("Preview of processed features:")
    print(df.head())
    print(f"Validation passed: {result.is_valid}")
