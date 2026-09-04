"""Data ingestion, synthetic generation, weather acquisition, preprocessing, and validation."""

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
from src.data.validator import (
    validate_dataset,
    run_data_validation,
    ValidationResult,
)
from src.data.preprocessing import (
    normalize_to_kolkata_timestamp,
    clean_and_resample_hourly,
    align_demand_and_weather,
    preprocess_and_save_dataset,
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
    "validate_dataset",
    "run_data_validation",
    "ValidationResult",
    "normalize_to_kolkata_timestamp",
    "clean_and_resample_hourly",
    "align_demand_and_weather",
    "preprocess_and_save_dataset",
]
