"""Configurable grid capacity and alert evaluation engine for Delhi Electricity Grid (Phase 6 / Prompt 7).

Monitors predicted demand against dynamic transmission and substation capacity thresholds.
Classifies operational state into:
- NORMAL: Utilization < 85%
- WARNING: 85% to < 95%
- CRITICAL: Utilization >= 95%
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

import config


def get_active_thresholds(
    capacity_mw: Optional[float] = None,
    warning_threshold: Optional[float] = None,
    critical_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Retrieve active grid capacity and thresholds dynamically from configuration or runtime overrides."""
    active_config = config.load_config()

    cap = float(capacity_mw if capacity_mw is not None else active_config.get("grid_capacity_mw", config.GRID_CAPACITY_MW))
    warn_ratio = float(warning_threshold if warning_threshold is not None else active_config.get("warning_threshold", config.WARNING_THRESHOLD))
    crit_ratio = float(critical_threshold if critical_threshold is not None else active_config.get("critical_threshold", config.CRITICAL_THRESHOLD))

    return {
        "grid_capacity_mw": cap,
        "warning_threshold_ratio": warn_ratio,
        "critical_threshold_ratio": crit_ratio,
        "warning_threshold_pct": round(warn_ratio * 100.0, 1),
        "critical_threshold_pct": round(crit_ratio * 100.0, 1),
        "warning_demand_mw": round(cap * warn_ratio, 1),
        "critical_demand_mw": round(cap * crit_ratio, 1),
    }


def evaluate_demand_alert(
    demand_mw: float,
    capacity_mw: Optional[float] = None,
    warning_threshold: Optional[float] = None,
    critical_threshold: Optional[float] = None,
    context_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate grid alert status for a given demand value in MW.

    Args:
        demand_mw: Peak or current demand in MW.
        capacity_mw: Optional override for grid capacity.
        warning_threshold: Optional override for warning threshold ratio (e.g. 0.85).
        critical_threshold: Optional override for critical threshold ratio (e.g. 0.95).
        context_timestamp: Optional timestamp of the observation or peak.

    Returns:
        Structured dictionary with status, utilization, thresholds, color, and operational guidance.
    """
    cfg = get_active_thresholds(
        capacity_mw=capacity_mw,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
    )
    cap = cfg["grid_capacity_mw"]
    warn_ratio = cfg["warning_threshold_ratio"]
    crit_ratio = cfg["critical_threshold_ratio"]

    demand = float(demand_mw)
    utilization_ratio = demand / cap if cap > 0 else 0.0
    utilization_pct = round(utilization_ratio * 100.0, 2)

    # Classification
    if utilization_ratio >= crit_ratio:
        status = "CRITICAL"
        color = "#EF4444"
        badge_class = "danger"
        icon = "🚨"
        title = "CRITICAL GRID OVERLOAD RISK"
        message = (
            f"Demand ({demand:,.0f} MW) breaches critical threshold of {cfg['critical_threshold_pct']}% "
            f"({cfg['critical_demand_mw']:,.0f} MW / {cap:,.0f} MW capacity). Grid stress is imminent."
        )
        action_recommended = (
            "Immediate SLDC action required: Dispatch spinning reserves, alert battery storage discharging, "
            "and prepare feeder-level rotational load balancing / demand response."
        )
    elif utilization_ratio >= warn_ratio:
        status = "WARNING"
        color = "#F59E0B"
        badge_class = "warning"
        icon = "⚠️"
        title = "HIGH DEMAND CAPACITY WARNING"
        message = (
            f"Demand ({demand:,.0f} MW) reached {utilization_pct}% of total grid capacity, "
            f"exceeding warning threshold of {cfg['warning_threshold_pct']}% ({cfg['warning_demand_mw']:,.0f} MW)."
        )
        action_recommended = (
            "Notify distribution companies (BRPL, BYPL, TPDDL), monitor substation thermal limits, "
            "and ready peaker generation units."
        )
    else:
        status = "NORMAL"
        color = "#10B981"
        badge_class = "success"
        icon = "🟢"
        title = "NORMAL OPERATING MARGINS"
        message = (
            f"Grid operating comfortably at {utilization_pct}% capacity ({demand:,.0f} MW / {cap:,.0f} MW). "
            f"Adequate reserve margins available."
        )
        action_recommended = "Continue standard economic dispatch and scheduled transmission monitoring."

    return {
        "status": status,
        "demand_mw": round(demand, 2),
        "grid_capacity_mw": cap,
        "utilization_pct": utilization_pct,
        "utilization_ratio": round(utilization_ratio, 4),
        "warning_threshold_pct": cfg["warning_threshold_pct"],
        "critical_threshold_pct": cfg["critical_threshold_pct"],
        "warning_demand_mw": cfg["warning_demand_mw"],
        "critical_demand_mw": cfg["critical_demand_mw"],
        "color": color,
        "badge_class": badge_class,
        "icon": icon,
        "title": title,
        "message": message,
        "action_recommended": action_recommended,
        "timestamp": context_timestamp,
    }


def evaluate_forecast_alerts(
    forecast_df: pd.DataFrame,
    capacity_mw: Optional[float] = None,
    warning_threshold: Optional[float] = None,
    critical_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate grid alert status across all hourly forecast intervals.

    Returns timeline of alert levels, count of critical/warning hours, and the overall peak alert.
    """
    if forecast_df.empty or "predicted_demand_mw" not in forecast_df.columns:
        raise ValueError("forecast_df must contain 'predicted_demand_mw' column.")

    cfg = get_active_thresholds(
        capacity_mw=capacity_mw,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
    )
    cap = cfg["grid_capacity_mw"]
    warn_ratio = cfg["warning_threshold_ratio"]
    crit_ratio = cfg["critical_threshold_ratio"]

    df = forecast_df.copy()
    demands = df["predicted_demand_mw"].values
    util_ratios = demands / cap
    util_pcts = util_ratios * 100.0

    statuses: List[str] = []
    colors: List[str] = []
    icons: List[str] = []

    for ratio in util_ratios:
        if ratio >= crit_ratio:
            statuses.append("CRITICAL")
            colors.append("#EF4444")
            icons.append("🚨")
        elif ratio >= warn_ratio:
            statuses.append("WARNING")
            colors.append("#F59E0B")
            icons.append("⚠️")
        else:
            statuses.append("NORMAL")
            colors.append("#10B981")
            icons.append("🟢")

    df["capacity_utilization_pct"] = np.round(util_pcts, 2)
    df["alert_status"] = statuses
    df["alert_color"] = colors

    # Summary counts
    total_hours = len(df)
    critical_hours = statuses.count("CRITICAL")
    warning_hours = statuses.count("WARNING")
    normal_hours = statuses.count("NORMAL")

    # Overall peak evaluation
    peak_idx = int(np.argmax(demands))
    peak_row = df.iloc[peak_idx]
    peak_demand = float(demands[peak_idx])
    peak_ts = str(peak_row.get("timestamp", ""))

    peak_alert = evaluate_demand_alert(
        demand_mw=peak_demand,
        capacity_mw=cap,
        warning_threshold=warn_ratio,
        critical_threshold=crit_ratio,
        context_timestamp=peak_ts,
    )

    # Timeline of only elevated risk hours
    elevated_hours_df = df[df["alert_status"].isin(["WARNING", "CRITICAL"])].copy()

    return {
        "overall_status": peak_alert["status"],
        "peak_alert": peak_alert,
        "grid_capacity_mw": cap,
        "warning_threshold_pct": cfg["warning_threshold_pct"],
        "critical_threshold_pct": cfg["critical_threshold_pct"],
        "total_hours": total_hours,
        "critical_hours_count": critical_hours,
        "warning_hours_count": warning_hours,
        "normal_hours_count": normal_hours,
        "has_critical_hours": critical_hours > 0,
        "has_warning_hours": warning_hours > 0,
        "annotated_forecast_df": df,
        "elevated_hours_df": elevated_hours_df,
    }


if __name__ == "__main__":
    print("Testing Alert Manager with synthetic load values...")
    normal_case = evaluate_demand_alert(6800.0)
    warning_case = evaluate_demand_alert(7900.0)
    critical_case = evaluate_demand_alert(8600.0)

    print(f"Normal (6800 MW): {normal_case['status']} ({normal_case['utilization_pct']}%)")
    print(f"Warning (7900 MW): {warning_case['status']} ({warning_case['utilization_pct']}%)")
    print(f"Critical (8600 MW): {critical_case['status']} ({critical_case['utilization_pct']}%)")
