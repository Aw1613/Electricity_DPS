"""Unit tests for secondary analytics: area analysis and renewable net demand (Prompt 8)."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.area_analysis import (
    get_area_demand_summary,
    rank_zones_by_peak,
    calculate_zone_proportions,
)
from src.features.renewables import (
    calculate_net_demand,
    simulate_solar_generation_profile,
    adjust_forecast_for_renewables,
    get_renewable_summary,
)


def test_area_demand_summary():
    """Verify geographic area grouping, aggregations, and ranking."""
    df_area = get_area_demand_summary()
    assert isinstance(df_area, pd.DataFrame)
    assert len(df_area) >= 5
    assert "area" in df_area.columns
    assert "peak_demand_mw" in df_area.columns
    assert "mean_demand_mw" in df_area.columns
    assert "share_pct" in df_area.columns

    # Verify ranked strictly descending by peak demand
    peaks = df_area["peak_demand_mw"].values
    assert all(peaks[i] >= peaks[i + 1] for i in range(len(peaks) - 1))


def test_area_demand_summary_fallback_when_empty():
    """Verify graceful fallback to demonstration DISCOM figures when data is absent."""
    empty_df = pd.DataFrame()
    fallback_summary = get_area_demand_summary(empty_df)
    assert len(fallback_summary) == 5
    assert fallback_summary["is_demonstration_data"].all()
    assert set(fallback_summary["area"]).issuperset({"North Delhi", "South Delhi", "West Delhi", "East Delhi", "Central Delhi"})


def test_calculate_zone_proportions():
    """Verify disaggregation of Delhi total load into geographic zones."""
    total_mw = 8000.0
    proportions = calculate_zone_proportions(total_demand_mw=total_mw)
    assert len(proportions) == 5
    assert "allocated_demand_mw" in proportions.columns
    total_allocated = proportions["allocated_demand_mw"].sum()
    # Should sum to approximately total_mw (allowing slight rounding)
    assert abs(total_allocated - total_mw) < 5.0


def test_calculate_net_demand_with_data():
    """Verify Net Demand = Gross Demand - Solar Generation."""
    timestamps = pd.date_range("2024-06-01 10:00:00", periods=5, freq="h")
    gross_df = pd.DataFrame({
        "timestamp": timestamps,
        "predicted_demand_mw": [7000.0, 7500.0, 7800.0, 7600.0, 7200.0],
    })
    solar_df = pd.DataFrame({
        "timestamp": timestamps,
        "solar_generation_mw": [200.0, 350.0, 400.0, 300.0, 150.0],
        "renewable_generation_mw": [240.0, 390.0, 440.0, 340.0, 190.0],
    })

    net_df = calculate_net_demand(gross_df, renewable_df=solar_df)
    assert "net_demand_mw" in net_df.columns
    assert "solar_contribution_pct" in net_df.columns
    assert net_df["is_renewable_available"].all()

    # Verify noon hour (7800 gross - 400 solar = 7400 net)
    assert net_df.loc[2, "net_demand_mw"] == 7400.0
    assert net_df.loc[2, "solar_contribution_pct"] == round((400 / 7800) * 100, 2)


def test_calculate_net_demand_missing_fallback():
    """Verify graceful handling when renewable data is absent or unaligned."""
    gross_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-06-01 00:00:00", periods=5, freq="h"),
        "demand_mw": [5000.0, 4800.0, 4600.0, 4700.0, 5200.0],
    })
    empty_ren_df = pd.DataFrame()

    res = calculate_net_demand(gross_df, renewable_df=empty_ren_df)
    assert not res["is_renewable_available"].all()
    assert res["renewable_status"].iloc[0] == "Renewable adjustment unavailable"
    assert (res["net_demand_mw"] == res["gross_demand_mw"]).all()


def test_simulate_solar_generation_profile():
    """Verify simulated daylight solar curve peaks at solar noon."""
    timestamps = pd.date_range("2024-06-01 00:00:00", periods=24, freq="h")
    sim = simulate_solar_generation_profile(timestamps, installed_capacity_mw=450.0)
    assert len(sim) == 24
    # Solar at night should be 0
    assert sim.loc[sim["timestamp"].str.endswith("02:00:00"), "solar_generation_mw"].values[0] == 0.0
    # Solar at noon (12:00) should be highest
    noon_solar = sim.loc[sim["timestamp"].str.endswith("12:00:00"), "solar_generation_mw"].values[0]
    assert noon_solar > 300.0
