"""Streamlit main entry point for Delhi Electricity Demand Prediction System."""

import streamlit as st
import config
from dashboard.components import render_header, render_kpi_card, render_alert_banner


def main():
    st.set_page_config(
        page_title="Delhi Electricity Demand Prediction System",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_header(
        title="Delhi Electricity Demand Prediction System",
        subtitle="AI-driven short-term load forecasting, peak alert engine, and grid stability analytics.",
    )

    # Sidebar
    st.sidebar.header("⚙️ System Configuration")
    st.sidebar.markdown(f"**Grid Capacity:** `{config.GRID_CAPACITY_MW:,} MW`")
    st.sidebar.markdown(f"**Warning Level:** `{config.WARNING_THRESHOLD * 100:.0f}%`")
    st.sidebar.markdown(f"**Critical Level:** `{config.CRITICAL_THRESHOLD * 100:.0f}%`")
    st.sidebar.markdown(f"**Forecast Horizon:** `{config.FORECAST_HORIZON_HOURS} Hours`")

    # Overview Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Total Grid Capacity", f"{config.GRID_CAPACITY_MW:,} MW")
    with col2:
        render_kpi_card("Warning Threshold", f"{config.GRID_CAPACITY_MW * config.WARNING_THRESHOLD:,.0f} MW")
    with col3:
        render_kpi_card("Critical Threshold", f"{config.GRID_CAPACITY_MW * config.CRITICAL_THRESHOLD:,.0f} MW")
    with col4:
        render_kpi_card("Forecast Horizon", f"{config.FORECAST_HORIZON_HOURS} Hours")

    st.divider()

    # System Status Placeholder
    render_alert_banner(
        current_demand_mw=6200.0,
        capacity_mw=config.GRID_CAPACITY_MW,
        warning_pct=config.WARNING_THRESHOLD,
        critical_pct=config.CRITICAL_THRESHOLD,
    )

    st.info("System initialized successfully. Ready for data generation and forecasting modules.")


if __name__ == "__main__":
    main()
