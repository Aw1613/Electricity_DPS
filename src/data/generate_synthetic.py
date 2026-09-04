"""Synthetic historical electricity demand dataset generator for Delhi.

Generates realistic hourly electricity demand patterns incorporating:
- Daily seasonality: morning ramp-up, afternoon load, evening peak, and night reduction.
- Weekly seasonality: weekday vs. weekend consumption profiles.
- Annual / seasonal seasonality: extreme summer cooling peak (May-July),
  winter heating peaks (Dec-Jan), and moderate spring/autumn baselines.
- Delhi DISCOM zones and feeders (BRPL, BYPL, TPDDL, NDMC).
- Controlled random noise.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MOCK_DIR = DATA_DIR / "mock"

# Delhi DISCOM areas, feeders, and demand distribution weights
DELHI_AREAS = [
    {"area": "South Delhi", "feeder": "BRPL-South", "weight": 0.30},
    {"area": "North Delhi", "feeder": "TPDDL-North", "weight": 0.24},
    {"area": "West Delhi", "feeder": "BRPL-West", "weight": 0.22},
    {"area": "East Delhi", "feeder": "BYPL-East", "weight": 0.16},
    {"area": "Central Delhi", "feeder": "NDMC-Central", "weight": 0.08},
]


def calculate_daily_factor(hour: int) -> float:
    """Return multiplier based on typical Delhi daily load profile."""
    # Hours 0-4: Night reduction (lowest load)
    if hour <= 4:
        return 0.65 + 0.05 * np.cos((hour - 3) * np.pi / 6)
    # Hours 5-9: Morning ramp-up (rising commercial, domestic activities)
    elif hour <= 9:
        return 0.70 + 0.22 * ((hour - 5) / 4.0)
    # Hours 10-16: Sustained afternoon load (offices, schools, cooling)
    elif hour <= 16:
        return 0.92 + 0.06 * np.sin((hour - 10) * np.pi / 6)
    # Hours 17-21: Evening peak (lighting, cooking, maximum domestic load)
    elif hour <= 21:
        return 0.98 + 0.08 * np.sin((hour - 17) * np.pi / 4)
    # Hours 22-23: Late night ramp-down
    else:
        return 0.85 - 0.15 * ((hour - 22) / 2.0)


def calculate_seasonal_factor(day_of_year: int) -> float:
    """Return multiplier based on Delhi climate (summer heatwave vs pleasant winter/spring)."""
    # Delhi peak summer: May to July (~Day 120 to 200) -> peak factor ~1.30 to 1.45
    # Winter heating: December to mid-January (~Day 335 to 365 and 1 to 20) -> factor ~1.05
    # Moderate spring/autumn: March-April, Oct-Nov -> factor ~0.85 to 0.95
    rad = 2 * np.pi * (day_of_year - 170) / 365.25
    summer_effect = 0.28 * np.exp(-0.5 * ((day_of_year - 165) / 40.0) ** 2)
    winter_effect = 0.08 * np.exp(-0.5 * ((day_of_year - 10) / 25.0) ** 2)
    winter_effect += 0.08 * np.exp(-0.5 * ((day_of_year - 355) / 25.0) ** 2)
    base_annual = 0.92 + 0.05 * np.cos(rad)
    return float(base_annual + summer_effect + winter_effect)


def generate_synthetic_demand(
    start_date: str = "2023-01-01 00:00:00",
    end_date: str = "2023-12-31 23:00:00",
    output_path: Optional[Path] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic hourly electricity demand dataset for Delhi.

    Args:
        start_date: Start datetime string (YYYY-MM-DD HH:MM:SS)
        end_date: End datetime string (YYYY-MM-DD HH:MM:SS)
        output_path: Optional file path to save CSV. Defaults to data/mock/synthetic_demand.csv
        seed: Random seed for reproducibility

    Returns:
        DataFrame containing ['timestamp', 'demand_mw', 'area', 'feeder']
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")

    # Baseline Delhi demand in MW (approx 5,200 MW nominal)
    base_demand = 5200.0

    all_records = []

    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek  # 0=Mon, 6=Sun
        day_of_year = ts.dayofyear

        # Weekend factor (Saturdays -4%, Sundays -9% due to commercial closures)
        if day_of_week == 5:
            weekend_factor = 0.96
        elif day_of_week == 6:
            weekend_factor = 0.91
        else:
            weekend_factor = 1.00

        daily_factor = calculate_daily_factor(hour)
        seasonal_factor = calculate_seasonal_factor(day_of_year)

        # Total grid hourly demand with subtle smooth noise
        grid_noise = rng.normal(0, 75.0)
        total_grid_mw = base_demand * daily_factor * seasonal_factor * weekend_factor + grid_noise

        # Ensure realistic boundaries (min ~3,000 MW, max bounded below 8,900 MW)
        total_grid_mw = float(np.clip(total_grid_mw, 3000.0, 8850.0))

        # Distribute into areas and feeders
        for area_info in DELHI_AREAS:
            area_weight = area_info["weight"]
            area_noise = rng.normal(0, 15.0)
            area_demand = round(max(50.0, total_grid_mw * area_weight + area_noise), 2)

            all_records.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "demand_mw": area_demand,
                "area": area_info["area"],
                "feeder": area_info["feeder"],
            })

    df = pd.DataFrame(all_records)

    # Save to disk
    if output_path is None:
        MOCK_DIR.mkdir(parents=True, exist_ok=True)
        output_path = MOCK_DIR / "synthetic_demand.csv"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Generated synthetic demand dataset: {len(df)} records saved to {output_path}")

    return df


def generate_synthetic_renewables(
    start_date: str = "2023-01-01 00:00:00",
    end_date: str = "2023-12-31 23:00:00",
    output_path: Optional[Path] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic hourly solar and renewable generation dataset for Delhi."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")

    records = []
    # Delhi rooftop & regional solar capacity ~ 400 MW max peak
    max_solar_mw = 400.0

    for ts in timestamps:
        hour = ts.hour
        day_of_year = ts.dayofyear

        # Solar curve (sunlight between 6:00 and 18:00, peak at 12:00 - 13:00)
        if 6 <= hour <= 18:
            solar_shape = np.sin((hour - 6) * np.pi / 12) ** 2
            # Monsoon cloud cover in July-August
            cloud_cover_factor = 0.55 if 190 <= day_of_year <= 240 else 0.92
            noise = rng.uniform(0.85, 1.05)
            solar_mw = round(max_solar_mw * solar_shape * cloud_cover_factor * noise, 2)
        else:
            solar_mw = 0.0

        # Small wind/biomass baseline (~30-60 MW)
        other_renewable_mw = round(float(rng.uniform(30.0, 60.0)), 2)
        total_renewable_mw = round(solar_mw + other_renewable_mw, 2)

        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "solar_generation_mw": solar_mw,
            "renewable_generation_mw": total_renewable_mw,
        })

    df = pd.DataFrame(records)

    if output_path is None:
        MOCK_DIR.mkdir(parents=True, exist_ok=True)
        output_path = MOCK_DIR / "synthetic_renewable.csv"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Generated synthetic renewable dataset: {len(df)} records saved to {output_path}")

    return df


if __name__ == "__main__":
    generate_synthetic_demand()
    generate_synthetic_renewables()
