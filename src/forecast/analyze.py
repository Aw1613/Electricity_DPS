"""Peak demand detection, capacity utilization, and uncertainty analysis engine (Task 5.3).

Exposes:
- detect_peaks: Finds max predicted demand MW, peak timestamp, capacity utilization %,
  top 5 peak demand periods, and validation error-based uncertainty bands.
- add_uncertainty_bounds: Computes lower and upper confidence intervals across the forecast timeline.
- get_top_peaks: Returns ranked list of top peak hours.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

import config
from src.models.predict import load_demand_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_default_validation_mape(model_path: Optional[Union[str, Path]] = None) -> float:
    """Retrieve validation MAPE from trained model metadata, defaulting to 3.0% if unavailable."""
    try:
        _, _, metadata = load_demand_model(model_path=model_path)
        val_metrics = metadata.get("val_metrics", {})
        mape = val_metrics.get("mape", 3.0)
        return float(mape)
    except Exception:
        return 3.0


def add_uncertainty_bounds(
    forecast_df: pd.DataFrame,
    uncertainty_mape: Optional[float] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Compute and append lower and upper estimated forecast bounds to each hourly prediction.

    Args:
        forecast_df: DataFrame containing 'predicted_demand_mw'.
        uncertainty_mape: Validation MAPE percentage (e.g. 2.8 for 2.8%). If None, loaded from model.
        model_path: Optional custom path to model joblib bundle.

    Returns:
        DataFrame with 'predicted_lower_mw' and 'predicted_upper_mw' columns added.
    """
    df = forecast_df.copy()
    if "predicted_demand_mw" not in df.columns:
        raise ValueError("forecast_df must contain 'predicted_demand_mw' column.")

    mape = uncertainty_mape if uncertainty_mape is not None else get_default_validation_mape(model_path)
    margin_ratio = mape / 100.0

    df["predicted_lower_mw"] = (df["predicted_demand_mw"] * (1.0 - margin_ratio)).round(2)
    df["predicted_upper_mw"] = (df["predicted_demand_mw"] * (1.0 + margin_ratio)).round(2)
    df["uncertainty_margin_mw"] = (df["predicted_demand_mw"] * margin_ratio).round(2)

    return df


def get_top_peaks(
    forecast_df: pd.DataFrame,
    top_n: int = 5,
    capacity_mw: Optional[float] = None,
) -> pd.DataFrame:
    """Identify the top N highest electricity demand periods in the forecast."""
    df = forecast_df.copy()
    cap = float(capacity_mw if capacity_mw is not None else config.GRID_CAPACITY_MW)

    if "predicted_demand_mw" not in df.columns:
        raise ValueError("forecast_df must contain 'predicted_demand_mw' column.")

    top_df = df.sort_values("predicted_demand_mw", ascending=False).head(top_n).copy()
    top_df["capacity_utilization_pct"] = ((top_df["predicted_demand_mw"] / cap) * 100.0).round(2)
    top_df["rank"] = range(1, len(top_df) + 1)

    output_cols = [
        "rank",
        "timestamp",
        "predicted_demand_mw",
        "capacity_utilization_pct",
    ]
    for extra_col in ["hour", "temperature", "day_name"]:
        if extra_col in top_df.columns:
            output_cols.append(extra_col)

    return top_df[output_cols].reset_index(drop=True)


def detect_peaks(
    forecast_df: pd.DataFrame,
    capacity_mw: Optional[float] = None,
    warning_threshold: Optional[float] = None,
    critical_threshold: Optional[float] = None,
    top_n: int = 5,
    uncertainty_mape: Optional[float] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Analyze forecast to identify peak demand, grid capacity risk, and uncertainty intervals.

    Args:
        forecast_df: DataFrame containing at least ['timestamp', 'predicted_demand_mw'].
        capacity_mw: Total Delhi grid transmission capacity (defaults to config.GRID_CAPACITY_MW).
        warning_threshold: Proportion for warning alert (defaults to config.WARNING_THRESHOLD, e.g. 0.85).
        critical_threshold: Proportion for critical alert (defaults to config.CRITICAL_THRESHOLD, e.g. 0.95).
        top_n: Number of highest peak intervals to extract.
        uncertainty_mape: Validation error MAPE % used for uncertainty bounds.
        model_path: Optional custom path to trained model.

    Returns:
        Structured dictionary with peak demand, peak time, capacity utilization,
        alert status, top peaks list, and uncertainty metrics.
    """
    if forecast_df.empty or "predicted_demand_mw" not in forecast_df.columns:
        raise ValueError("forecast_df is empty or missing 'predicted_demand_mw'.")

    cap = float(capacity_mw if capacity_mw is not None else config.GRID_CAPACITY_MW)
    warn_thresh = float(warning_threshold if warning_threshold is not None else config.WARNING_THRESHOLD)
    crit_thresh = float(critical_threshold if critical_threshold is not None else config.CRITICAL_THRESHOLD)

    # 1. Identify overall absolute peak
    peak_idx = forecast_df["predicted_demand_mw"].idxmax()
    peak_row = forecast_df.loc[peak_idx]
    peak_demand_mw = float(peak_row["predicted_demand_mw"])
    peak_timestamp = str(peak_row["timestamp"])

    # 2. Capacity utilization
    utilization_pct = round((peak_demand_mw / cap) * 100.0, 2)

    # 3. Alert classification
    if utilization_pct >= (crit_thresh * 100.0):
        status = "CRITICAL"
        status_message = f"Critical Alert: Forecasted peak of {peak_demand_mw:,.0f} MW exceeds {crit_thresh*100:.0f}% capacity ({cap:,.0f} MW)!"
    elif utilization_pct >= (warn_thresh * 100.0):
        status = "WARNING"
        status_message = f"Warning Alert: Forecasted peak of {peak_demand_mw:,.0f} MW reaches {utilization_pct}% of grid capacity."
    else:
        status = "NORMAL"
        status_message = f"Grid Normal: Forecasted peak of {peak_demand_mw:,.0f} MW operates safely within operating margins ({utilization_pct}%)."

    # 4. Uncertainty analysis
    mape = uncertainty_mape if uncertainty_mape is not None else get_default_validation_mape(model_path)
    error_margin_mw = round(peak_demand_mw * (mape / 100.0), 2)
    peak_lower_bound_mw = round(peak_demand_mw - error_margin_mw, 2)
    peak_upper_bound_mw = round(peak_demand_mw + error_margin_mw, 2)

    # 5. Extract top N peaks
    top_peaks_df = get_top_peaks(forecast_df, top_n=top_n, capacity_mw=cap)

    return {
        "peak_demand_mw": round(peak_demand_mw, 2),
        "peak_timestamp": peak_timestamp,
        "peak_hour": int(peak_row.get("hour", pd.to_datetime(peak_timestamp).hour)),
        "peak_temperature_c": float(peak_row.get("temperature", np.nan)),
        "capacity_mw": cap,
        "capacity_utilization_pct": utilization_pct,
        "warning_threshold_pct": round(warn_thresh * 100.0, 1),
        "critical_threshold_pct": round(crit_thresh * 100.0, 1),
        "status": status,
        "status_message": status_message,
        "uncertainty": {
            "validation_mape_pct": round(mape, 2),
            "error_margin_mw": error_margin_mw,
            "peak_lower_bound_mw": peak_lower_bound_mw,
            "peak_upper_bound_mw": peak_upper_bound_mw,
        },
        "top_peaks": top_peaks_df.to_dict(orient="records"),
        "top_peaks_df": top_peaks_df,
    }


if __name__ == "__main__":
    from src.forecast.predict_24h import predict_next_24h

    print("Analyzing 24-Hour Forecast Peaks...")
    df_fc = predict_next_24h()
    df_fc = add_uncertainty_bounds(df_fc)
    peak_analysis = detect_peaks(df_fc)

    print(f"Peak Demand: {peak_analysis['peak_demand_mw']} MW at {peak_analysis['peak_timestamp']}")
    print(f"Capacity Utilization: {peak_analysis['capacity_utilization_pct']}% -> Status: {peak_analysis['status']}")
    print(f"Uncertainty Band (±{peak_analysis['uncertainty']['validation_mape_pct']}%): {peak_analysis['uncertainty']['peak_lower_bound_mw']} MW to {peak_analysis['uncertainty']['peak_upper_bound_mw']} MW")
    print("\n--- Top 5 Peak Periods ---")
    print(peak_analysis["top_peaks_df"].to_string(index=False))
