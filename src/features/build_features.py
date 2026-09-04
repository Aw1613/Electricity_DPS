"""Feature engineering pipeline for Delhi Electricity Demand Prediction System.

Constructs high-signal predictive features strictly WITHOUT target data leakage:
1. Calendar / Time features:
   - hour (0-23)
   - day_of_week (0-6)
   - day_of_month (1-31)
   - month (1-12)
   - is_weekend (0 or 1)
   - season (1: Winter, 2: Summer, 3: Monsoon, 4: Autumn)

2. Demand Lag features (prior historical observations only):
   - lag_1h, lag_2h, lag_3h, lag_24h, lag_48h, lag_168h

3. Rolling Demand Statistics (computed on strictly shifted historical demand to prevent leakage):
   - rolling_mean_3h, rolling_mean_6h, rolling_mean_24h, rolling_max_24h

4. Weather Interactions:
   - temperature_squared, temperature_x_hour, temperature_x_weekend
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_INPUT_FILE = PROCESSED_DIR / "merged_features.csv"
DEFAULT_OUTPUT_FILE = PROCESSED_DIR / "feature_matrix.csv"


def get_season_code(month: int) -> int:
    """Return Delhi meteorological season code.

    1: Winter (Dec, Jan, Feb)
    2: Summer (Mar, Apr, May, Jun)
    3: Monsoon (Jul, Aug, Sep)
    4: Post-Monsoon / Autumn (Oct, Nov)
    """
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5, 6]:
        return 2
    elif month in [7, 8, 9]:
        return 3
    else:
        return 4


def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Extract temporal and calendar features from timestamp."""
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])

    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["day_of_month"] = ts.dt.day
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["season"] = out["month"].apply(get_season_code)

    return out


def add_demand_lag_features(
    df: pd.DataFrame,
    demand_col: str = "demand_mw",
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Create lagged demand features strictly referencing historical observations.

    No data leakage: lag_1h refers to demand at t-1, lag_24h to t-24, etc.
    """
    if lags is None:
        lags = [1, 2, 3, 24, 48, 168]

    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}h"] = out[demand_col].shift(lag)

    return out


def add_rolling_features(
    df: pd.DataFrame,
    demand_col: str = "demand_mw",
) -> pd.DataFrame:
    """Compute rolling window demand statistics without target leakage.

    CRITICAL LEAKAGE PREVENTION:
    All rolling statistics are calculated on demand_mw.shift(1), ensuring that
    the current hour's target demand (t) is NEVER included in the rolling window.
    """
    out = df.copy()
    shifted_demand = out[demand_col].shift(1)

    out["rolling_mean_3h"] = shifted_demand.rolling(window=3).mean()
    out["rolling_mean_6h"] = shifted_demand.rolling(window=6).mean()
    out["rolling_mean_24h"] = shifted_demand.rolling(window=24).mean()
    out["rolling_max_24h"] = shifted_demand.rolling(window=24).max()

    return out


def add_weather_interaction_features(
    df: pd.DataFrame,
    temp_col: str = "temperature",
) -> pd.DataFrame:
    """Construct non-linear weather and temporal interaction features."""
    out = df.copy()

    # Fall back to temperature_2m if temperature not present
    actual_temp_col = temp_col if temp_col in out.columns else "temperature_2m"
    if actual_temp_col not in out.columns:
        raise ValueError(f"Neither '{temp_col}' nor 'temperature_2m' found in DataFrame.")

    # Ensure required calendar components exist
    if "hour" not in out.columns:
        out = add_calendar_features(out)

    temp = out[actual_temp_col].astype(float)
    out["temperature_squared"] = temp ** 2
    out["temperature_x_hour"] = temp * out["hour"]
    out["temperature_x_weekend"] = temp * out["is_weekend"]

    return out


def build_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    demand_col: str = "demand_mw",
    temp_col: str = "temperature",
    drop_na: bool = True,
) -> pd.DataFrame:
    """Transform preprocessed DataFrame into full feature matrix.

    Args:
        df: Clean preprocessed continuous hourly DataFrame.
        timestamp_col: Name of timestamp column.
        demand_col: Name of electricity demand target column.
        temp_col: Name of temperature column.
        drop_na: Whether to drop rows containing NaN values produced by 168h lags/rolling.

    Returns:
        DataFrame containing original columns + calendar, lag, rolling, and interaction features.
    """
    out = df.copy()

    # 1. Calendar features
    out = add_calendar_features(out, timestamp_col=timestamp_col)

    # 2. Demand Lags (without leakage)
    out = add_demand_lag_features(out, demand_col=demand_col)

    # 3. Rolling Statistics (without leakage)
    out = add_rolling_features(out, demand_col=demand_col)

    # 4. Weather Interactions
    out = add_weather_interaction_features(out, temp_col=temp_col)

    # 5. Clean NaN values
    if drop_na:
        initial_len = len(out)
        out = out.dropna().reset_index(drop=True)
        dropped_len = initial_len - len(out)
        # Note: 168 initial rows are expected to drop due to 168h lookback
        print(f"Features created: Dropped {dropped_len} initial burn-in rows due to 168h lags. Remaining rows: {len(out)}.")

    return out


def get_feature_columns() -> List[str]:
    """Return standard ordered list of model predictor column names."""
    return [
        # Calendar
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "season",
        # Weather
        "temperature",
        "humidity",
        "apparent_temperature",
        "precipitation",
        "wind_speed",
        # Lags
        "lag_1h",
        "lag_2h",
        "lag_3h",
        "lag_24h",
        "lag_48h",
        "lag_168h",
        # Rolling stats
        "rolling_mean_3h",
        "rolling_mean_6h",
        "rolling_mean_24h",
        "rolling_max_24h",
        # Interactions
        "temperature_squared",
        "temperature_x_hour",
        "temperature_x_weekend",
    ]


def build_and_save_feature_matrix(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Load preprocessed features, compute all engineered features, and save to CSV."""
    source_path = input_path or DEFAULT_INPUT_FILE
    if not source_path.exists():
        from src.data.preprocessing import preprocess_and_save_dataset
        print(f"{source_path} not found. Running preprocessing pipeline first...")
        preprocess_and_save_dataset(output_path=source_path)

    df_raw = pd.read_csv(source_path)
    df_features = build_features(df_raw, drop_na=drop_na)

    target_path = output_path or DEFAULT_OUTPUT_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(target_path, index=False)
    print(f"Feature matrix saved: {len(df_features)} rows x {len(df_features.columns)} columns to {target_path}")

    return df_features


if __name__ == "__main__":
    matrix = build_and_save_feature_matrix()
    print("Engineered features preview:")
    print(matrix[get_feature_columns()].head())
