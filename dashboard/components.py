"""Dashboard UI components for Delhi Electricity Demand Prediction System."""

import streamlit as st
from typing import Optional


def render_header(title: str = "Delhi Electricity Demand Prediction System", subtitle: Optional[str] = None):
    """Render standardized application header."""
    st.title(f"⚡ {title}")
    if subtitle:
        st.caption(subtitle)


def render_kpi_card(label: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None):
    """Render a styled KPI metric card."""
    st.metric(label=label, value=value, delta=delta, help=help_text)


def render_alert_banner(current_demand_mw: float, capacity_mw: float, warning_pct: float, critical_pct: float):
    """Render an alert banner depending on grid load ratio."""
    ratio = current_demand_mw / capacity_mw if capacity_mw > 0 else 0.0
    pct = ratio * 100

    if ratio >= critical_pct:
        st.error(f"🚨 **CRITICAL GRID ALERT**: Load is at {pct:.1f}% of total capacity ({current_demand_mw:,.0f} / {capacity_mw:,.0f} MW)!")
    elif ratio >= warning_pct:
        st.warning(f"⚠️ **HIGH LOAD WARNING**: Load is at {pct:.1f}% of total capacity ({current_demand_mw:,.0f} / {capacity_mw:,.0f} MW).")
    else:
        st.success(f"✅ **NORMAL STATUS**: Grid operating stably at {pct:.1f}% load ({current_demand_mw:,.0f} / {capacity_mw:,.0f} MW).")
