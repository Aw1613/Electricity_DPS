"""Unit tests for Alert Manager and Demand Service (Prompt 7 / Phase 6)."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.alerts.alert_manager import (
    evaluate_demand_alert,
    evaluate_forecast_alerts,
    get_active_thresholds,
)
from src.services.demand_service import (
    generate_24h_forecast_service,
    get_current_grid_snapshot_service,
    get_historical_demand_service,
    get_model_info_service,
)


def test_get_active_thresholds():
    """Verify threshold calculation and default config fallback."""
    cfg = get_active_thresholds(capacity_mw=10000.0, warning_threshold=0.80, critical_threshold=0.90)
    assert cfg["grid_capacity_mw"] == 10000.0
    assert cfg["warning_demand_mw"] == 8000.0
    assert cfg["critical_demand_mw"] == 9000.0
    assert cfg["warning_threshold_pct"] == 80.0
    assert cfg["critical_threshold_pct"] == 90.0


def test_evaluate_demand_alert_normal():
    """Verify NORMAL status when utilization < 85%."""
    res = evaluate_demand_alert(demand_mw=7000.0, capacity_mw=9000.0, warning_threshold=0.85, critical_threshold=0.95)
    assert res["status"] == "NORMAL"
    assert res["utilization_pct"] == round((7000 / 9000) * 100, 2)
    assert res["badge_class"] == "success"
    assert "🟢" in res["icon"]


def test_evaluate_demand_alert_warning():
    """Verify WARNING status when utilization is between 85% and 95%."""
    res = evaluate_demand_alert(demand_mw=8000.0, capacity_mw=9000.0, warning_threshold=0.85, critical_threshold=0.95)
    assert res["status"] == "WARNING"
    assert res["utilization_pct"] == round((8000 / 9000) * 100, 2)
    assert res["badge_class"] == "warning"
    assert "⚠️" in res["icon"]


def test_evaluate_demand_alert_critical():
    """Verify CRITICAL status when utilization >= 95%."""
    res = evaluate_demand_alert(demand_mw=8700.0, capacity_mw=9000.0, warning_threshold=0.85, critical_threshold=0.95)
    assert res["status"] == "CRITICAL"
    assert res["utilization_pct"] == round((8700 / 9000) * 100, 2)
    assert res["badge_class"] == "danger"
    assert "🚨" in res["icon"]


def test_evaluate_forecast_alerts():
    """Verify timeline alerts on a multi-hour forecast DataFrame."""
    demands = [7000.0] * 10 + [8000.0] * 5 + [8600.0] * 3
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-06-01", periods=len(demands), freq="h"),
        "predicted_demand_mw": demands,
    })

    eval_result = evaluate_forecast_alerts(
        forecast_df=df,
        capacity_mw=9000.0,
        warning_threshold=0.85,
        critical_threshold=0.95,
    )

    assert eval_result["overall_status"] == "CRITICAL"
    assert eval_result["critical_hours_count"] == 3
    assert eval_result["warning_hours_count"] == 5
    assert eval_result["normal_hours_count"] == 10
    assert "alert_status" in eval_result["annotated_forecast_df"].columns


def test_demand_service_integration():
    """Verify demand service endpoints produce valid structured outputs."""
    # Snapshot
    snapshot = get_current_grid_snapshot_service()
    assert "current_demand_mw" in snapshot
    assert "alert" in snapshot
    assert snapshot["alert"]["status"] in ["NORMAL", "WARNING", "CRITICAL"]

    # Model info
    model_info = get_model_info_service()
    assert model_info["is_loaded"] is True
    assert "val_metrics" in model_info

    # 24h forecast service
    fc_res = generate_24h_forecast_service()
    assert fc_res["horizon"] == "24h"
    assert len(fc_res["forecast_df"]) == 24
    assert "peak_demand_mw" in fc_res
    assert fc_res["overall_status"] in ["NORMAL", "WARNING", "CRITICAL"]
