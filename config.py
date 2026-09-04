"""Configuration loader for Delhi Electricity Demand Prediction System."""

from pathlib import Path
from typing import Any, Dict

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "grid_capacity_mw": 9000,
    "warning_threshold": 0.85,
    "critical_threshold": 0.95,
    "forecast_horizon_hours": 24,
}


def _fallback_yaml_parse(content: str) -> Dict[str, Any]:
    """Simple parser for basic key-value YAML files when pyyaml is not yet installed."""
    data: Dict[str, Any] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            # Try parsing numeric types
            try:
                if "." in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                pass
            data[key] = val
    return data


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML file, falling back to defaults if not found."""
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except ImportError:
        with open(config_path, "r", encoding="utf-8") as f:
            data = _fallback_yaml_parse(f.read())

    config = DEFAULT_CONFIG.copy()
    config.update(data)
    return config


# Global configuration instance
CONFIG = load_config()

# Convenience exports
GRID_CAPACITY_MW = CONFIG["grid_capacity_mw"]
WARNING_THRESHOLD = CONFIG["warning_threshold"]
CRITICAL_THRESHOLD = CONFIG["critical_threshold"]
FORECAST_HORIZON_HOURS = CONFIG["forecast_horizon_hours"]
