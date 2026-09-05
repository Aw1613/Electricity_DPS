"""Unit tests for Point-in-Time & Historical Replay Telemetry Service."""

import pytest
import pandas as pd
from src.services.demand_service import get_point_in_time_telemetry_service


def test_get_point_in_time_telemetry_valid_date():
    """Verify that a known historical date (2024-06-19 15:00) retrieves accurate telemetry."""
    res = get_point_in_time_telemetry_service("2024-06-19 15:00:00")

    assert res["target_timestamp"] == "2024-06-19 15:00:00"
    assert res["target_date"] == "2024-06-19"
    assert res["target_hour"] == 15
    assert 8000 <= res["actual_demand_mw"] <= 9000
    assert res["predicted_demand_mw"] > 0
    assert res["temperature_c"] > 35.0  # High summer heatwave
    assert res["alert_status"] in ["NORMAL", "WARNING", "CRITICAL"]

    # Check 24-hour day profile
    day_df = res["day_profile_24h"]
    assert isinstance(day_df, pd.DataFrame)
    assert len(day_df) == 24
    assert "actual_demand_mw" in day_df.columns
    assert "predicted_demand_mw" in day_df.columns
    assert "temperature_c" in day_df.columns

    # Check 7-day week context
    week_df = res["week_context_7d"]
    assert isinstance(week_df, pd.DataFrame)
    assert len(week_df) > 24
    assert "demand_mw" in week_df.columns
    assert "is_target_day" in week_df.columns


def test_get_point_in_time_solar_day_night():
    """Verify that rooftop solar PV generates at noon and produces 0 MW at night."""
    res_noon = get_point_in_time_telemetry_service("2024-06-19 12:00:00", solar_capacity_mw=500.0)
    res_night = get_point_in_time_telemetry_service("2024-06-19 02:00:00", solar_capacity_mw=500.0)

    assert res_noon["renewable"]["solar_generation_mw"] > 0.0
    assert res_noon["renewable"]["net_demand_mw"] < res_noon["renewable"]["gross_demand_mw"]

    assert res_night["renewable"]["solar_generation_mw"] == 0.0
    assert res_night["renewable"]["net_demand_mw"] == res_night["renewable"]["gross_demand_mw"]


def test_get_point_in_time_area_breakdown():
    """Verify that discom area apportionment correctly distributes instant load."""
    res = get_point_in_time_telemetry_service("2023-05-15 14:00:00")
    area_data = res["area_breakdown"]

    assert "area_summary_df" in area_data
    summary_df = area_data["area_summary_df"]
    assert len(summary_df) == 5  # 5 Delhi zones

    # Sum of apportioned zones should match total demand within floating point rounding
    assert abs(summary_df["demand_mw"].sum() - res["actual_demand_mw"]) < 1.0


def test_get_point_in_time_boundary_clamping():
    """Verify that out-of-range dates clamp safely to available boundaries."""
    res_past = get_point_in_time_telemetry_service("2015-01-01 00:00:00")
    res_future = get_point_in_time_telemetry_service("2035-01-01 00:00:00")

    assert res_past["target_date"] >= res_past["min_available_date"]
    assert res_future["target_date"] <= res_future["max_available_date"]
