"""Data validation utilities for Delhi Electricity Demand Prediction System.

Performs data sanity checks ensuring:
- Required columns ('timestamp', 'demand_mw', 'temperature') are present.
- Demand is strictly positive (> 0 MW) and within realistic grid boundaries (< 12,000 MW).
- Temperature is within plausible physical limits for Delhi (-5°C to 55°C).
- Timestamps are continuous without temporal gaps (strictly 1-hour resolution).
- Returns boolean validation results, error lists, and explicit warnings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


@dataclass
class ValidationResult:
    """Encapsulates validation findings."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.is_valid


def validate_dataset(
    df: pd.DataFrame,
    min_demand_mw: float = 0.0,
    max_demand_mw: float = 12000.0,
    min_temp_c: float = -5.0,
    max_temp_c: float = 55.0,
    expected_freq_hours: int = 1,
) -> Tuple[bool, List[str]]:
    """Validate DataFrame integrity and return (is_valid, warnings/errors).

    Convenience wrapper returning a boolean and list of combined messages.
    """
    result = run_data_validation(
        df,
        min_demand_mw=min_demand_mw,
        max_demand_mw=max_demand_mw,
        min_temp_c=min_temp_c,
        max_temp_c=max_temp_c,
        expected_freq_hours=expected_freq_hours,
    )
    all_messages = [f"[ERROR] {e}" for e in result.errors] + [f"[WARNING] {w}" for w in result.warnings]
    return result.is_valid, all_messages


def run_data_validation(
    df: pd.DataFrame,
    min_demand_mw: float = 0.0,
    max_demand_mw: float = 12000.0,
    min_temp_c: float = -5.0,
    max_temp_c: float = 55.0,
    expected_freq_hours: int = 1,
) -> ValidationResult:
    """Perform comprehensive data integrity checks on demand and weather datasets."""
    errors: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, Any] = {}

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return ValidationResult(
            is_valid=False,
            errors=["DataFrame is empty or None."],
            warnings=[],
            metrics={"row_count": 0},
        )

    metrics["row_count"] = len(df)

    # 1. Required Columns Check
    col_mapping = {c.lower(): c for c in df.columns}

    # Timestamp column check
    time_col = None
    for candidate in ["timestamp", "time", "datetime", "date"]:
        if candidate in col_mapping:
            time_col = col_mapping[candidate]
            break

    if not time_col:
        errors.append("Missing required 'timestamp' column.")

    # Demand column check
    demand_col = None
    for candidate in ["demand_mw", "demand", "load_mw", "total_demand_mw"]:
        if candidate in col_mapping:
            demand_col = col_mapping[candidate]
            break

    if not demand_col:
        errors.append("Missing required 'demand_mw' column.")

    # Temperature column check (accept 'temperature' or 'temperature_2m')
    temp_col = None
    for candidate in ["temperature", "temperature_2m", "temp"]:
        if candidate in col_mapping:
            temp_col = col_mapping[candidate]
            break

    if not temp_col:
        errors.append("Missing required 'temperature' column.")

    # If fundamental columns are missing, return early
    if errors:
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings, metrics=metrics)

    # 2. Timestamp Continuity & Ordering Checks
    try:
        ts_series = pd.to_datetime(df[time_col])
        if not ts_series.is_monotonic_increasing:
            warnings.append("Timestamps are not in strictly increasing chronological order.")
            ts_series = ts_series.sort_values().reset_index(drop=True)

        # Check for duplicates
        duplicate_count = ts_series.duplicated().sum()
        if duplicate_count > 0:
            errors.append(f"Found {duplicate_count} duplicate timestamps.")

        # Check hourly step continuity
        time_diffs = ts_series.diff().dropna()
        expected_step = pd.Timedelta(hours=expected_freq_hours)
        irregular_steps = time_diffs[time_diffs != expected_step]

        if not irregular_steps.empty:
            gaps_count = len(irregular_steps)
            max_gap = time_diffs.max()
            errors.append(
                f"Non-continuous timestamps detected: {gaps_count} gaps found. Largest gap: {max_gap}."
            )
            metrics["gap_count"] = gaps_count
            metrics["max_gap"] = str(max_gap)
        else:
            metrics["gap_count"] = 0
    except Exception as e:
        errors.append(f"Failed to parse timestamps: {e}")

    # 3. Demand Validity Checks
    try:
        demand_values = pd.to_numeric(df[demand_col], errors="coerce")
        null_demand = demand_values.isna().sum()
        if null_demand > 0:
            errors.append(f"Found {null_demand} null/NaN values in '{demand_col}'.")

        non_null_demand = demand_values.dropna()
        if not non_null_demand.empty:
            min_val = float(non_null_demand.min())
            max_val = float(non_null_demand.max())
            metrics["demand_min_mw"] = min_val
            metrics["demand_max_mw"] = max_val

            # Strictly positive check
            if min_val <= min_demand_mw:
                non_positive_count = int((non_null_demand <= min_demand_mw).sum())
                errors.append(
                    f"Found {non_positive_count} non-positive demand values (min value: {min_val:.2f} MW <= {min_demand_mw} MW)."
                )

            # Plausible upper boundary check
            if max_val > max_demand_mw:
                excess_count = int((non_null_demand > max_demand_mw).sum())
                warnings.append(
                    f"Found {excess_count} demand values exceeding normal grid bounds ({max_val:.2f} MW > {max_demand_mw} MW)."
                )
    except Exception as e:
        errors.append(f"Error validating demand: {e}")

    # 4. Temperature Validity Checks
    try:
        temp_values = pd.to_numeric(df[temp_col], errors="coerce")
        null_temp = temp_values.isna().sum()
        if null_temp > 0:
            errors.append(f"Found {null_temp} null/NaN values in '{temp_col}'.")

        non_null_temp = temp_values.dropna()
        if not non_null_temp.empty:
            min_temp = float(non_null_temp.min())
            max_temp = float(non_null_temp.max())
            metrics["temp_min_c"] = min_temp
            metrics["temp_max_c"] = max_temp

            if min_temp < min_temp_c or max_temp > max_temp_c:
                out_of_range = int(((non_null_temp < min_temp_c) | (non_null_temp > max_temp_c)).sum())
                errors.append(
                    f"Temperature out of reasonable Delhi range [{min_temp_c}°C, {max_temp_c}°C]: "
                    f"{out_of_range} anomalies detected (min: {min_temp:.1f}°C, max: {max_temp:.1f}°C)."
                )
    except Exception as e:
        errors.append(f"Error validating temperature: {e}")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, metrics=metrics)


if __name__ == "__main__":
    # Self-test demonstration
    sample_df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=24, freq="h"),
        "demand_mw": [5000 + i * 50 for i in range(24)],
        "temperature": [20.0 + i * 0.5 for i in range(24)],
    })
    valid, msgs = validate_dataset(sample_df)
    print(f"Sample data validation passed: {valid}, messages: {msgs}")
