"""Plotly chart utilities for Delhi Electricity Demand Prediction System (Phase 9 / Prompt 9).

Provides interactive, publication-grade visualizations for:
1. Actual vs. Predicted Demand with shaded uncertainty intervals and capacity thresholds.
2. 7-Day multi-step recursive forecast trends with daily peak markers.
3. Temperature vs. Demand relationship (cooling load sensitivity curve).
4. Capacity Utilization Gauge with operational color bands.
5. Area & Feeder Demand breakdown (ranked bar chart and zonal share donut).
6. Renewable Gross vs. Net Demand chart with solar generation shading.
7. Hourly Alert Timeline risk classification chart.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Color palette for operational grid control
THEME_COLORS = {
    "actual": "#3B82F6",         # Electric blue
    "forecast": "#F97316",       # Amber / orange
    "uncertainty": "rgba(249, 115, 22, 0.18)",
    "net_demand": "#10B981",     # Emerald green
    "solar": "#FACC15",          # Solar gold
    "capacity": "#EF4444",       # Red
    "warning": "#F59E0B",        # Amber
    "normal": "#10B981",         # Green
    "dark_bg": "#000000",
    "card_bg": "#050505",
    "grid_line": "rgba(255, 255, 255, 0.08)",
    "text": "#D8D8D8",
}


def plot_actual_vs_predicted(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    capacity_mw: Optional[float] = None,
    warning_mw: Optional[float] = None,
    title: str = "Actual vs. Predicted Electricity Demand (24-Hour Horizon)",
) -> go.Figure:
    """Create interactive time-series comparing historical load with 24h predictions and uncertainty bounds."""
    fig = go.Figure()

    # 1. Historical Actual Demand
    if not historical_df.empty and "demand_mw" in historical_df.columns:
        hist_df = historical_df.copy()
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(hist_df["timestamp"]),
                y=hist_df["demand_mw"],
                mode="lines",
                name="Historical Demand (MW)",
                line=dict(color=THEME_COLORS["actual"], width=2.5),
                hovertemplate="<b>Historical Demand</b>: %{y:,.1f} MW<br>Time: %{x}<extra></extra>",
            )
        )

    # 2. Uncertainty Band (Lower and Upper bounds)
    has_bounds = "predicted_lower_mw" in forecast_df.columns and "predicted_upper_mw" in forecast_df.columns
    if has_bounds:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(forecast_df["timestamp"]),
                y=forecast_df["predicted_upper_mw"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(forecast_df["timestamp"]),
                y=forecast_df["predicted_lower_mw"],
                mode="lines",
                fill="tonexty",
                fillcolor=THEME_COLORS["uncertainty"],
                name="Forecast Confidence (±MAPE)",
                line=dict(width=0),
                hovertemplate="Range: %{y:,.0f} - " + "%{text:,.0f} MW<extra></extra>",
                text=forecast_df["predicted_upper_mw"],
            )
        )

    # 3. Forecast Line
    if not forecast_df.empty and "predicted_demand_mw" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(forecast_df["timestamp"]),
                y=forecast_df["predicted_demand_mw"],
                mode="lines+markers",
                name="Predicted Demand (MW)",
                line=dict(color=THEME_COLORS["forecast"], width=3, dash="dash"),
                marker=dict(size=5, color=THEME_COLORS["forecast"]),
                hovertemplate="<b>Forecast Demand</b>: %{y:,.1f} MW<br>Time: %{x}<extra></extra>",
            )
        )

        # Highlight Peak Hour
        peak_idx = forecast_df["predicted_demand_mw"].idxmax()
        peak_row = forecast_df.loc[peak_idx]
        fig.add_trace(
            go.Scatter(
                x=[pd.to_datetime(peak_row["timestamp"])],
                y=[peak_row["predicted_demand_mw"]],
                mode="markers+text",
                name="Forecast Peak",
                marker=dict(size=14, color="#EF4444", symbol="star"),
                text=[f"Peak: {peak_row['predicted_demand_mw']:,.0f} MW"],
                textposition="top center",
                hovertemplate="<b>Peak Demand</b>: %{y:,.1f} MW<br>Hour: %{x}<extra></extra>",
            )
        )

    # 4. Threshold Lines
    if capacity_mw:
        fig.add_hline(
            y=capacity_mw,
            line_dash="dot",
            line_color=THEME_COLORS["capacity"],
            line_width=1.5,
            annotation_text=f"Grid Capacity ({capacity_mw:,.0f} MW)",
            annotation_position="top right",
            annotation_font_color=THEME_COLORS["capacity"],
        )

    if warning_mw:
        fig.add_hline(
            y=warning_mw,
            line_dash="dashdot",
            line_color=THEME_COLORS["warning"],
            line_width=1.5,
            annotation_text=f"Warning Threshold ({warning_mw:,.0f} MW)",
            annotation_position="bottom right",
            annotation_font_color=THEME_COLORS["warning"],
        )

    fig.update_layout(
        title=dict(text=title, font=dict(family="'Inter', sans-serif", size=16, color="#FFFFFF")),
        paper_bgcolor="#000000",
        plot_bgcolor="#050505",
        font=dict(family="'Inter', sans-serif", color="#D8D8D8"),
        xaxis=dict(title="Timeline", gridcolor="rgba(255, 255, 255, 0.08)", zerolinecolor="rgba(255, 255, 255, 0.12)"),
        yaxis=dict(title="Electricity Demand (MW)", gridcolor="rgba(255, 255, 255, 0.08)", zerolinecolor="rgba(255, 255, 255, 0.12)"),
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#D8D8D8")),
        margin=dict(l=50, r=40, t=60, b=40),
        height=440,
    )

    return fig


def plot_7d_forecast_trend(
    forecast_7d_df: pd.DataFrame,
    daily_summary_df: Optional[pd.DataFrame] = None,
    capacity_mw: Optional[float] = None,
    warning_mw: Optional[float] = None,
) -> go.Figure:
    """Create interactive 7-day (168-hour) multi-step recursive trend chart with daily peak markers."""
    fig = go.Figure()

    if forecast_7d_df.empty or "predicted_demand_mw" not in forecast_7d_df.columns:
        return fig

    # 1. 168-hour demand line
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(forecast_7d_df["timestamp"]),
            y=forecast_7d_df["predicted_demand_mw"],
            mode="lines",
            name="7-Day Predicted Demand (MW)",
            line=dict(color="#6366F1", width=2.5),
            hovertemplate="<b>Predicted Load</b>: %{y:,.1f} MW<br>Date/Time: %{x}<extra></extra>",
        )
    )

    # 2. Daily peak markers
    if daily_summary_df is not None and not daily_summary_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(daily_summary_df["peak_timestamp"]),
                y=daily_summary_df["peak_demand_mw"],
                mode="markers+text",
                name="Daily Peaks",
                marker=dict(size=11, color="#EF4444", symbol="diamond"),
                text=[f"{val:,.0f} MW" for val in daily_summary_df["peak_demand_mw"]],
                textposition="top center",
                textfont=dict(size=10, color="#B91C1C"),
                hovertemplate="<b>Daily Peak</b>: %{y:,.1f} MW<br>Peak Time: %{x}<extra></extra>",
            )
        )

    # 3. Capacity lines
    if capacity_mw:
        fig.add_hline(
            y=capacity_mw,
            line_dash="dot",
            line_color=THEME_COLORS["capacity"],
            annotation_text=f"Capacity ({capacity_mw:,.0f} MW)",
            annotation_position="top right",
        )
    if warning_mw:
        fig.add_hline(
            y=warning_mw,
            line_dash="dashdot",
            line_color=THEME_COLORS["warning"],
            annotation_text=f"Warning ({warning_mw:,.0f} MW)",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=dict(text="7-Day Electricity Demand Forecast (168-Hour Horizon)", font=dict(size=16)),
        xaxis=dict(title="Date & Time", gridcolor="rgba(255, 255, 255, 0.08)"),
        yaxis=dict(title="Demand (MW)", gridcolor="rgba(255, 255, 255, 0.08)"),
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#050505",
        font=dict(family="'Inter', sans-serif", color="#D8D8D8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=40, t=60, b=40),
        height=440,
    )

    return fig


def plot_temperature_vs_demand(
    df: pd.DataFrame,
    temp_col: str = "temperature",
    demand_col: str = "demand_mw",
) -> go.Figure:
    """Scatter plot demonstrating temperature-driven cooling load sensitivity in Delhi."""
    fig = go.Figure()

    actual_temp = temp_col if temp_col in df.columns else "temperature_2m"
    actual_demand = demand_col if demand_col in df.columns else "predicted_demand_mw"

    if actual_temp not in df.columns or actual_demand not in df.columns:
        return fig

    sub_df = df.dropna(subset=[actual_temp, actual_demand]).copy()

    # Color by hour if available
    color_var = sub_df["hour"] if "hour" in sub_df.columns else None

    scatter = go.Scatter(
        x=sub_df[actual_temp],
        y=sub_df[actual_demand],
        mode="markers",
        name="Hourly Observations",
        marker=dict(
            size=6,
            color=color_var if color_var is not None else sub_df[actual_temp],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Hour of Day" if color_var is not None else "Temp (°C)"),
            opacity=0.65,
        ),
        hovertemplate="Temperature: %{x:.1f}°C<br>Demand: %{y:,.0f} MW<extra></extra>",
    )
    fig.add_trace(scatter)

    # Polynomial trendline showing non-linear cooling surge above 28°C
    if len(sub_df) > 20 and sub_df[actual_temp].nunique() > 3:
        poly_fit = np.polyfit(sub_df[actual_temp], sub_df[actual_demand], deg=2)
        x_trend = np.linspace(sub_df[actual_temp].min(), sub_df[actual_temp].max(), 100)
        y_trend = np.polyval(poly_fit, x_trend)

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=y_trend,
                mode="lines",
                name="Thermal Response Curve",
                line=dict(color="#EF4444", width=3),
                hoverinfo="skip",
            )
        )

    # Cooling threshold vertical reference line at 28°C
    fig.add_vline(
        x=28.0,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text="Air Conditioning Threshold (28°C)",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(text="Weather Correlation: Ambient Temperature vs. Electricity Demand", font=dict(size=16)),
        xaxis=dict(title="Ambient Temperature (°C)", gridcolor="rgba(255, 255, 255, 0.08)"),
        yaxis=dict(title="Electricity Demand (MW)", gridcolor="rgba(255, 255, 255, 0.08)"),
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#050505",
        font=dict(family="'Inter', sans-serif", color="#D8D8D8"),
        margin=dict(l=50, r=40, t=60, b=40),
        height=420,
    )

    return fig


def plot_capacity_gauge(
    current_or_peak_mw: float,
    capacity_mw: float = 9000.0,
    warning_pct: float = 0.85,
    critical_pct: float = 0.95,
    title: str = "Peak Grid Capacity Utilization",
) -> go.Figure:
    """Plotly indicator gauge displaying current or peak load relative to transmission capacity."""
    utilization_pct = (current_or_peak_mw / capacity_mw * 100.0) if capacity_mw > 0 else 0.0

    warn_val = capacity_mw * warning_pct
    crit_val = capacity_mw * critical_pct

    # Determine bar color
    if current_or_peak_mw >= crit_val:
        bar_color = "#EF4444"
    elif current_or_peak_mw >= warn_val:
        bar_color = "#F59E0B"
    else:
        bar_color = "#10B981"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=current_or_peak_mw,
            number={"suffix": " MW", "valueformat": ",.0f"},
            delta={"reference": warn_val, "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#10B981"}},
            title={"text": f"<b>{title}</b><br><span style='font-size:0.8em;color:gray'>{utilization_pct:.1f}% Capacity</span>"},
            gauge={
                "axis": {"range": [0, max(capacity_mw * 1.1, current_or_peak_mw * 1.1)], "tickformat": ",.0f"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "steps": [
                    {"range": [0, warn_val], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [warn_val, crit_val], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [crit_val, max(capacity_mw * 1.1, current_or_peak_mw * 1.1)], "color": "rgba(239, 68, 68, 0.35)"},
                ],
                "threshold": {
                    "line": {"color": "#DC2626", "width": 4},
                    "thickness": 0.8,
                    "value": crit_val,
                },
            },
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#050505",
        font=dict(family="'Inter', sans-serif", color="#D8D8D8"),
        margin=dict(l=30, r=30, t=60, b=20),
        height=320,
    )

    return fig


def plot_area_breakdown_bars(area_summary_df: pd.DataFrame) -> go.Figure:
    """Ranked horizontal bar chart of peak electricity load across Delhi DISCOM zones."""
    fig = go.Figure()

    if area_summary_df.empty:
        return fig

    df = area_summary_df.sort_values("peak_demand_mw", ascending=True)

    fig.add_trace(
        go.Bar(
            y=df["area"],
            x=df["peak_demand_mw"],
            orientation="h",
            marker=dict(
                color=df["peak_demand_mw"],
                colorscale="Blues",
                showscale=False,
            ),
            text=[f"{val:,.0f} MW ({share}%)" for val, share in zip(df["peak_demand_mw"], df["share_pct"])],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Peak Load: %{x:,.1f} MW<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="Geographic Demand by Delhi Zone (Ranked by Peak Load)", font=dict(size=15)),
        xaxis=dict(title="Peak Demand (MW)", gridcolor="rgba(255, 255, 255, 0.08)"),
        yaxis=dict(title=""),
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#050505",
        font=dict(family="'Inter', sans-serif", color="#D8D8D8"),
        margin=dict(l=100, r=40, t=50, b=40),
        height=340,
    )

    return fig


def plot_area_breakdown_pie(area_summary_df: pd.DataFrame) -> go.Figure:
    """Donut chart illustrating zonal consumption share across Delhi DISCOMs."""
    fig = go.Figure()

    if area_summary_df.empty:
        return fig

    fig.add_trace(
        go.Pie(
            labels=area_summary_df["area"],
            values=area_summary_df["share_pct"],
            hole=0.45,
            marker=dict(colors=["#3B82F6", "#6366F1", "#0EA5E9", "#F59E0B", "#10B981"]),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Share: %{percent}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="Zonal Demand Distribution Share (%)", font=dict(size=15)),
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#050505",
        font=dict(family="'Inter', sans-serif", color="#D8D8D8"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=340,
    )

    return fig


def plot_renewable_net_demand_chart(forecast_df: pd.DataFrame) -> go.Figure:
    """Interactive chart demonstrating Gross Demand vs. Solar Generation and Net Grid Load."""
    fig = go.Figure()

    if forecast_df.empty or "gross_demand_mw" not in forecast_df.columns:
        return fig

    x_vals = pd.to_datetime(forecast_df["timestamp"])

    # Gross Demand line
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=forecast_df["gross_demand_mw"],
            mode="lines",
            name="Gross Grid Demand (MW)",
            line=dict(color="#3B82F6", width=2.5),
            hovertemplate="Gross Demand: %{y:,.0f} MW<extra></extra>",
        )
    )

    # Net Demand line
    if "net_demand_mw" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=forecast_df["net_demand_mw"],
                mode="lines",
                name="Net Demand (After Solar)",
                line=dict(color="#10B981", width=3, dash="solid"),
                hovertemplate="Net Demand: %{y:,.0f} MW<extra></extra>",
            )
        )

    # Solar Generation shaded area
    if "solar_generation_mw" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=forecast_df["solar_generation_mw"],
                mode="lines",
                name="Rooftop Solar Offset",
                fill="tozeroy",
                fillcolor="rgba(250, 204, 21, 0.35)",
                line=dict(color="#EAB308", width=1.5),
                hovertemplate="Solar Generation: %{y:,.1f} MW<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="Renewable Shaving: Gross Load vs. Solar Generation vs. Net Demand", font=dict(size=16)),
        xaxis=dict(title="Timeline", gridcolor="rgba(255, 255, 255, 0.08)"),
        yaxis=dict(title="Power (MW)", gridcolor="rgba(255, 255, 255, 0.08)"),
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#050505",
        font=dict(family="'Inter', sans-serif", color="#D8D8D8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=40, t=60, b=40),
        height=420,
    )

    return fig


def plot_hourly_alert_timeline(forecast_df: pd.DataFrame) -> go.Figure:
    """Timeline bar chart displaying hourly alert risk classification across the forecast."""
    fig = go.Figure()

    if forecast_df.empty or "predicted_demand_mw" not in forecast_df.columns:
        return fig

    df = forecast_df.copy()
    status_colors = {"NORMAL": "#10B981", "WARNING": "#F59E0B", "CRITICAL": "#EF4444"}
    bar_colors = [status_colors.get(s, "#10B981") for s in df.get("alert_status", ["NORMAL"] * len(df))]

    fig.add_trace(
        go.Bar(
            x=pd.to_datetime(df["timestamp"]),
            y=df["predicted_demand_mw"],
            marker_color=bar_colors,
            name="Hourly Load Risk",
            hovertemplate="Time: %{x}<br>Demand: %{y:,.0f} MW<br>Risk: %{text}<extra></extra>",
            text=df.get("alert_status", ["NORMAL"] * len(df)),
        )
    )

    fig.update_layout(
        title=dict(text="Hourly Grid Alert Classification Timeline", font=dict(size=15)),
        xaxis=dict(title="Time", gridcolor="rgba(255, 255, 255, 0.08)"),
        yaxis=dict(title="Demand (MW)", gridcolor="rgba(255, 255, 255, 0.08)"),
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#050505",
        font=dict(family="'Inter', sans-serif", color="#D8D8D8"),
        margin=dict(l=50, r=40, t=50, b=40),
        height=320,
    )

    return fig
