"""Test configuration loading."""

import sys
from pathlib import Path

# Add project root to sys.path so tests can be run from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CONFIG,
    GRID_CAPACITY_MW,
    WARNING_THRESHOLD,
    CRITICAL_THRESHOLD,
    FORECAST_HORIZON_HOURS,
    load_config,
)


def test_config_values():
    """Verify default and YAML configuration parameters."""
    cfg = load_config()
    assert cfg["grid_capacity_mw"] == 9000, f"Expected 9000, got {cfg.get('grid_capacity_mw')}"
    assert cfg["warning_threshold"] == 0.85, f"Expected 0.85, got {cfg.get('warning_threshold')}"
    assert cfg["critical_threshold"] == 0.95, f"Expected 0.95, got {cfg.get('critical_threshold')}"
    assert cfg["forecast_horizon_hours"] == 24, f"Expected 24, got {cfg.get('forecast_horizon_hours')}"

    assert GRID_CAPACITY_MW == 9000
    assert WARNING_THRESHOLD == 0.85
    assert CRITICAL_THRESHOLD == 0.95
    assert FORECAST_HORIZON_HOURS == 24


if __name__ == "__main__":
    test_config_values()
    print("All configuration tests passed successfully!")
