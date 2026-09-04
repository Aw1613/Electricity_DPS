"""Renewable and solar energy integration module for Delhi (Task 8.1 / Prompt 8).

Computes:
- Net Demand = Gross Demand - Solar Generation
- Renewable and solar contribution percentages: (Solar Generation / Gross Demand) * 100
- Robust fallback when renewable telemetry is unavailable.
- Future solar generation simulation for net demand forecasting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.data.data_loader import load_renewable_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def calculate_net_demand(
    gross_demand_df: pd.DataFrame,
    renewable_df: Optional[pd.DataFrame] = None,
    demand_col: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate Net Demand by deducting solar generation from gross electricity load.

    Formula:
        Net Demand = Gross Demand - Solar Generation
        Solar Contribution % = (Solar Generation / Gross Demand) * 100

    Args:
        gross_demand_df: DataFrame with timestamps and electricity demand.
        renewable_df: Optional DataFrame with ['timestamp', 'solar_generation_mw', 'renewable_generation_mw'].
        demand_col: Optional column name for demand. Defaults to auto-detecting
                    ('demand_mw', 'predicted_demand_mw', 'gross_demand_mw').

    Returns:
        DataFrame with net demand, solar metrics, and availability flag.
    """
    if gross_demand_df.empty:
        raise ValueError("gross_demand_df cannot be empty.")

    out = gross_demand_df.copy()

    # Identify demand column
    target_col = None
    if demand_col and demand_col in out.columns:
        target_col = demand_col
    else:
        for candidate in ["predicted_demand_mw", "demand_mw", "gross_demand_mw", "load_mw"]:
            if candidate in out.columns:
                target_col = candidate
                break

    if target_col is None:
        raise ValueError("Could not find a valid electricity demand column in gross_demand_df.")

    out["gross_demand_mw"] = pd.to_numeric(out[target_col], errors="coerce")
    out["timestamp_dt"] = pd.to_datetime(out["timestamp"])

    # Load renewable dataset if not passed explicitly
    if renewable_df is None:
        try:
            renewable_df = load_renewable_data()
        except Exception:
            renewable_df = None

    # Check availability
    if (
        renewable_df is None
        or renewable_df.empty
        or "solar_generation_mw" not in renewable_df.columns
    ):
        out["is_renewable_available"] = False
        out["renewable_status"] = "Renewable adjustment unavailable"
        out["solar_generation_mw"] = 0.0
        out["renewable_generation_mw"] = 0.0
        out["net_demand_mw"] = out["gross_demand_mw"]
        out["solar_contribution_pct"] = 0.0
        out["renewable_contribution_pct"] = 0.0
        out = out.drop(columns=["timestamp_dt"], errors="ignore")
        return out

    # Prepare renewable lookup
    ren_df = renewable_df.copy()
    ren_df["timestamp_dt"] = pd.to_datetime(ren_df["timestamp"])
    ren_df = ren_df.sort_values("timestamp_dt")

    # Ensure existing renewable columns are dropped if re-calculating
    out = out.drop(
        columns=[
            "solar_generation_mw",
            "renewable_generation_mw",
            "net_demand_mw",
            "solar_contribution_pct",
            "renewable_contribution_pct",
            "is_renewable_available",
            "renewable_status",
        ],
        errors="ignore",
    )

    # Merge on timestamp
    merged = pd.merge_asof(
        out.sort_values("timestamp_dt"),
        ren_df[["timestamp_dt", "solar_generation_mw", "renewable_generation_mw"]],
        on="timestamp_dt",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )

    # Fill unaligned gaps
    merged["solar_generation_mw"] = merged["solar_generation_mw"].fillna(0.0).round(2)
    merged["renewable_generation_mw"] = (
        merged["renewable_generation_mw"].fillna(merged["solar_generation_mw"]).round(2)
    )

    # Compute Net Demand: Gross - Solar
    merged["net_demand_mw"] = (merged["gross_demand_mw"] - merged["solar_generation_mw"]).clip(lower=0.0).round(2)

    # Contribution percentages
    denom = np.where(merged["gross_demand_mw"] > 0, merged["gross_demand_mw"], 1.0)
    merged["solar_contribution_pct"] = np.clip(
        (merged["solar_generation_mw"] / denom) * 100.0, 0.0, 100.0
    ).round(2)
    merged["renewable_contribution_pct"] = np.clip(
        (merged["renewable_generation_mw"] / denom) * 100.0, 0.0, 100.0
    ).round(2)

    merged["is_renewable_available"] = True
    merged["renewable_status"] = "Active"

    return merged.drop(columns=["timestamp_dt"], errors="ignore").reset_index(drop=True)


def simulate_solar_generation_profile(
    timestamps: Union[pd.Series, List[Any]],
    installed_capacity_mw: float = 450.0,
    performance_ratio: float = 0.82,
) -> pd.DataFrame:
    """Generate diurnal solar output curve for future forecast horizons.

    Used when forecasting future periods where historical solar meter logs do not yet exist.
    """
    ts_series = pd.to_datetime(timestamps)
    records = []

    for ts in ts_series:
        hour = ts.hour
        # Delhi daylight envelope ~6:00 to 18:00, peak solar noon at 12:30
        if 6 <= hour <= 18:
            solar_factor = np.sin((hour - 6) * np.pi / 12) ** 2
            solar_mw = round(float(installed_capacity_mw * performance_ratio * solar_factor), 2)
        else:
            solar_mw = 0.0

        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "solar_generation_mw": solar_mw,
            "renewable_generation_mw": round(solar_mw + 40.0, 2),  # + baseline biomass/waste-to-energy
        })

    return pd.DataFrame(records)


def adjust_forecast_for_renewables(
    forecast_df: pd.DataFrame,
    installed_solar_capacity_mw: float = 450.0,
) -> pd.DataFrame:
    """Convenience helper to augment any 24h or 7d forecast DataFrame with net demand."""
    if forecast_df.empty:
        return forecast_df

    simulated_solar = simulate_solar_generation_profile(
        timestamps=forecast_df["timestamp"],
        installed_capacity_mw=installed_solar_capacity_mw,
    )

    return calculate_net_demand(
        gross_demand_df=forecast_df,
        renewable_df=simulated_solar,
    )


def get_renewable_summary(renewable_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Calculate key renewable metrics and capacity status."""
    if renewable_df is None:
        try:
            renewable_df = load_renewable_data()
        except Exception:
            renewable_df = None

    if (
        renewable_df is None
        or renewable_df.empty
        or "solar_generation_mw" not in renewable_df.columns
    ):
        return {
            "is_available": False,
            "status_message": "Renewable adjustment unavailable",
            "peak_solar_mw": 0.0,
            "peak_renewable_mw": 0.0,
            "mean_solar_mw": 0.0,
        }

    solar = pd.to_numeric(renewable_df["solar_generation_mw"], errors="coerce").fillna(0.0)
    ren = pd.to_numeric(renewable_df["renewable_generation_mw"], errors="coerce").fillna(0.0)

    # Daytime slice (solar > 1 MW)
    daytime_solar = solar[solar > 1.0]
    mean_daytime = float(daytime_solar.mean()) if len(daytime_solar) > 0 else 0.0

    return {
        "is_available": True,
        "status_message": "Active",
        "peak_solar_mw": round(float(solar.max()), 2),
        "peak_renewable_mw": round(float(ren.max()), 2),
        "mean_daytime_solar_mw": round(mean_daytime, 2),
        "total_records": len(renewable_df),
    }


if __name__ == "__main__":
    from src.forecast.predict_24h import predict_next_24h

    print("Evaluating Renewable Net Demand...")
    fc_24h = predict_next_24h()
    adjusted_fc = adjust_forecast_for_renewables(fc_24h, installed_solar_capacity_mw=450.0)

    print("--- 24-Hour Net Demand Adjustment Preview ---")
    print(adjusted_fc[["timestamp", "gross_demand_mw", "solar_generation_mw", "net_demand_mw", "solar_contribution_pct"]].iloc[10:16].to_string(index=False))

    ren_summary = get_renewable_summary()
    print(f"\nRenewable Summary: Peak Solar = {ren_summary['peak_solar_mw']} MW | Daytime Mean = {ren_summary['mean_daytime_solar_mw']} MW")
