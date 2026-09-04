"""Feature engineering and domain analytics package for Delhi Electricity Demand Prediction System."""

from src.features.build_features import (
    build_features,
    get_feature_columns,
    build_and_save_feature_matrix,
    add_calendar_features,
    add_demand_lag_features,
    add_rolling_features,
    add_weather_interaction_features,
    get_season_code,
)
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

__all__ = [
    "build_features",
    "get_feature_columns",
    "build_and_save_feature_matrix",
    "add_calendar_features",
    "add_demand_lag_features",
    "add_rolling_features",
    "add_weather_interaction_features",
    "get_season_code",
    "get_area_demand_summary",
    "rank_zones_by_peak",
    "calculate_zone_proportions",
    "calculate_net_demand",
    "simulate_solar_generation_profile",
    "adjust_forecast_for_renewables",
    "get_renewable_summary",
]
