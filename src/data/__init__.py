"""Data ingestion, synthetic generation, and weather acquisition package for Delhi Electricity Demand."""

from src.data.generate_synthetic import (
    generate_synthetic_demand,
    generate_synthetic_renewables,
    DELHI_AREAS,
)
from src.data.weather_api import (
    fetch_weather_data,
    fetch_weather_from_api,
    generate_mock_weather,
)
from src.data.data_loader import (
    load_historical_demand,
    load_weather,
    load_renewable_data,
    load_area_data,
)

__all__ = [
    "generate_synthetic_demand",
    "generate_synthetic_renewables",
    "DELHI_AREAS",
    "fetch_weather_data",
    "fetch_weather_from_api",
    "generate_mock_weather",
    "load_historical_demand",
    "load_weather",
    "load_renewable_data",
    "load_area_data",
]
