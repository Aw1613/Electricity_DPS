"""Synthetic historical electricity demand and weather dataset generator for Delhi.

Generates realistic hourly electricity demand patterns incorporating:
- Base daily pattern: minimum load ~2,000 MW (winter night), summer peak ~8,500-9,000 MW
- Two daily peaks in summer: ~15:00-15:30 (commercial & AC load) and a secondary peak after 23:00 (residential night AC)
- Winter: flatter curve, much lower peak (~4,000-5,000 MW), no strong afternoon spike
- Seasonal trend: gradual ramp-up March-June (pre-monsoon heatwave), dip in monsoon (Jul-Sep) due to humidity/rain cooling,
  brief rise in autumn (Oct), low in winter (Dec-Feb)
- Weekday vs weekend: slightly lower demand on weekends (less commercial/office load)
- Realistic random noise (~2-4% of value) with smooth autocorrelation
- Correlated synthetic temperature series (higher temp = higher demand, especially above 35°C non-linear surge)
- Clear labeling as SYNTHETIC in metadata and dataset columns per project spec Data Mode requirements
- Delhi DISCOM zones and feeders (BRPL, BYPL, TPDDL, NDMC)
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MOCK_DIR = DATA_DIR / "mock"
RAW_DIR = DATA_DIR / "raw"

# Data Mode label constant
DATA_MODE_LABEL = "DEMONSTRATION / SYNTHETIC"

# Delhi DISCOM areas, feeders, and demand distribution weights
DELHI_AREAS = [
    {"area": "South Delhi", "feeder": "BRPL-South", "weight": 0.30},
    {"area": "North Delhi", "feeder": "TPDDL-North", "weight": 0.24},
    {"area": "West Delhi", "feeder": "BRPL-West", "weight": 0.22},
    {"area": "East Delhi", "feeder": "BYPL-East", "weight": 0.16},
    {"area": "Central Delhi", "feeder": "NDMC-Central", "weight": 0.08},
]


def calculate_daily_factor(hour: int) -> float:
    """Return multiplier based on typical Delhi daily load profile (legacy compatibility)."""
    if hour <= 4:
        return 0.65 + 0.05 * np.cos((hour - 3) * np.pi / 6)
    elif hour <= 9:
        return 0.70 + 0.22 * ((hour - 5) / 4.0)
    elif hour <= 16:
        return 0.92 + 0.06 * np.sin((hour - 10) * np.pi / 6)
    elif hour <= 21:
        return 0.98 + 0.08 * np.sin((hour - 17) * np.pi / 4)
    else:
        return 0.85 - 0.15 * ((hour - 22) / 2.0)


def calculate_seasonal_factor(day_of_year: int) -> float:
    """Return multiplier based on Delhi climate (legacy compatibility)."""
    rad = 2 * np.pi * (day_of_year - 170) / 365.25
    summer_effect = 0.28 * np.exp(-0.5 * ((day_of_year - 165) / 40.0) ** 2)
    winter_effect = 0.08 * np.exp(-0.5 * ((day_of_year - 10) / 25.0) ** 2)
    winter_effect += 0.08 * np.exp(-0.5 * ((day_of_year - 355) / 25.0) ** 2)
    base_annual = 0.92 + 0.05 * np.cos(rad)
    return float(base_annual + summer_effect + winter_effect)


def generate_synthetic_weather_series(
    timestamps: pd.DatetimeIndex,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate physically realistic hourly Delhi weather series.

    Matches Delhi's historical climate patterns:
    - Extreme dry summer heatwave (May-June): daytime peaks 42-46°C, nights 28-33°C
    - Monsoon (July-September): rain cooling, highs 31-36°C, relative humidity 70-95%
    - Autumn (October): pleasant warmth, highs 30-34°C, dry air
    - Winter (December-February): cold nights 5-9°C, pleasant afternoons 18-24°C
    - Diurnal cycle: minimum temp around 05:30, peak around 15:00-15:30
    """
    rng = np.random.default_rng(seed)
    n = len(timestamps)

    hours = timestamps.hour.values
    days_of_year = timestamps.dayofyear.values
    months = timestamps.month.values

    # Seasonal mean temperature baseline by month center
    month_centers = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    month_means = np.array([13.8, 17.8, 24.2, 31.0, 36.2, 37.2, 31.5, 30.5, 29.8, 26.2, 20.2, 14.5])

    extended_centers = np.concatenate([month_centers - 365, month_centers, month_centers + 365])
    extended_means = np.concatenate([month_means, month_means, month_means])
    daily_mean_temps = np.interp(days_of_year, extended_centers, extended_means)

    is_monsoon = (months >= 7) & (months <= 9)
    is_summer = (months >= 4) & (months <= 6)
    is_winter = (months == 12) | (months == 1) | (months == 2)

    # Diurnal temperature swing: wider in dry summer, narrower in humid monsoon
    diurnal_amp = np.where(is_monsoon, 3.8, np.where(is_summer, 7.0, 5.8))
    # Peak temperature occurs around 15:00-15:30 (hour 15.25), lowest at 05:30 (hour 5.5)
    diurnal_cycle = diurnal_amp * np.sin(2 * np.pi * (hours - 9.375) / 24.0)

    # Autoregressive weather wave noise (heatwaves, cool breezes lasting 2-4 days)
    weather_noise = np.zeros(n)
    w_curr = 0.0
    for i in range(n):
        w_curr = 0.95 * w_curr + rng.normal(0, 0.40)
        weather_noise[i] = w_curr

    temperature = np.round(daily_mean_temps + diurnal_cycle + weather_noise, 2)
    temperature = np.clip(temperature, 4.0, 47.5)

    # Humidity: high during monsoon (70-95%), low in dry summer (20-40%), moderate in winter (50-80%)
    base_humidity = np.where(is_monsoon, 78.0, np.where(is_summer, 32.0, np.where(is_winter, 68.0, 48.0)))
    daily_humidity_cycle = -14.0 * np.sin(2 * np.pi * (hours - 9.375) / 24.0)
    humidity_noise = rng.normal(0, 4.0, size=n)
    humidity = np.round(np.clip(base_humidity + daily_humidity_cycle + humidity_noise, 15.0, 99.0), 1)

    # Apparent temperature (heat index calculation)
    apparent_temp = np.zeros(n)
    for i in range(n):
        t = temperature[i]
        rh = humidity[i]
        if t > 26.0:
            heat_index_bonus = (rh / 100.0) * (t - 20.0) * 0.46
            apparent_temp[i] = round(t + heat_index_bonus, 2)
        elif t < 14.0:
            apparent_temp[i] = round(t - 1.2, 2)
        else:
            apparent_temp[i] = t

    # Precipitation: primarily during monsoon months
    precip = np.zeros(n)
    for i in range(n):
        if is_monsoon[i] and rng.random() < 0.20:
            precip[i] = round(float(rng.exponential(3.2)), 2)
        elif rng.random() < 0.015:
            precip[i] = round(float(rng.uniform(0.1, 1.2)), 2)

    # Wind speed in km/h
    wind_speed = np.round(np.clip(rng.gamma(shape=3.0, scale=3.6, size=n), 2.0, 36.0), 2)

    weather_df = pd.DataFrame({
        "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature_2m": temperature,
        "relative_humidity_2m": humidity,
        "apparent_temperature": apparent_temp,
        "precipitation": precip,
        "wind_speed_10m": wind_speed,
        "is_synthetic": True,
    })
    return weather_df


def generate_synthetic_demand(
    start_date: str = "2023-01-01 00:00:00",
    end_date: str = "2024-12-31 23:00:00",
    output_path: Optional[Path] = None,
    seed: int = 42,
    sync_weather: bool = True,
) -> pd.DataFrame:
    """Generate realistic hourly electricity demand dataset for Delhi over a 2-year horizon.

    Incorporates exact Delhi grid characteristics:
    - Base daily pattern: minimum load ~2,000 MW (winter night), summer peak ~8,500-9,000 MW
    - Two daily peaks in summer: ~15:00-15:30 (AC & commercial load) and secondary peak after 23:00 (residential sleep AC)
    - Winter: flatter curve, much lower peak (~4,000-5,000 MW), no strong afternoon spike
    - Seasonal trend: gradual ramp-up March-June, dip in monsoon (Jul-Sep), brief autumn rise (Oct), low in winter (Dec-Feb)
    - Weekday vs weekend: slightly lower demand on weekends
    - Realistic random noise (~2-4% of value)
    - Strong non-linear correlation with synthetic temperature (hockey-stick surge above 35°C)
    - Clear SYNTHETIC labeling in metadata and data column

    Args:
        start_date: Start datetime string (YYYY-MM-DD HH:MM:SS)
        end_date: End datetime string (YYYY-MM-DD HH:MM:SS)
        output_path: Optional file path to save CSV. Defaults to data/mock/synthetic_demand.csv
        seed: Random seed for reproducibility
        sync_weather: If True, also generates matching weather data to data/mock/weather_mock.csv

    Returns:
        DataFrame containing ['timestamp', 'demand_mw', 'area', 'feeder', 'is_synthetic']
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    n = len(timestamps)

    # Generate aligned synthetic weather series
    weather_df = generate_synthetic_weather_series(timestamps, seed=seed)
    temperature = weather_df["temperature_2m"].values

    hours = timestamps.hour.values
    days_of_year = timestamps.dayofyear.values
    days_of_week = timestamps.dayofweek.values
    months = timestamps.month.values

    # Base seasonal temperature progression for profile weighting
    month_centers = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    month_means = np.array([13.8, 17.8, 24.2, 31.0, 36.2, 37.2, 31.5, 30.5, 29.8, 26.2, 20.2, 14.5])
    extended_centers = np.concatenate([month_centers - 365, month_centers, month_centers + 365])
    extended_means = np.concatenate([month_means, month_means, month_means])
    daily_mean_temps = np.interp(days_of_year, extended_centers, extended_means)

    is_monsoon = (months >= 7) & (months <= 9)
    is_autumn = (months == 10)

    # Baseload non-weather electricity: essential services, water pumping, base domestic/industrial
    base_load = 1950.0

    # Summer vs Winter seasonal blending
    summer_factor = np.clip((daily_mean_temps - 16.0) / 20.0, 0.0, 1.0)
    winter_factor = 1.0 - summer_factor

    # Winter daily profile:
    # Night minimum ~2,000 MW total (hours 02-04)
    # Flatter afternoon (no AC spike), evening heating/lighting peak ~4,000-5,000 MW total
    winter_hourly = np.array([
        -20, -70, -130, -150, -120, -40,     # 00-05: Night minimum
        280, 850, 1400, 1500, 1250, 1050,    # 06-11: Morning routine/geysers
        950, 900, 900, 920, 1000, 1350,      # 12-17: Flatter afternoon (no AC spike)
        1900, 2150, 2050, 1600, 800, 300     # 18-23: Evening peak (~4,000-4,800 MW)
    ])

    # Summer daily profile:
    # Two distinct peaks:
    # 1. Primary peak ~15:00-15:30 (Commercial HVAC + peak afternoon ambient temperature)
    # 2. Secondary peak after 23:00 (Residential bedroom AC as families sleep)
    # Night minimum in summer is sustained (>= 3000 MW in June)
    summer_hourly = np.array([
        1850, 1650, 1500, 1380, 1300, 1250,  # 00-05: Night sustained AC
        1400, 1750, 2150, 2600, 2900, 3150,  # 06-11: Commercial morning ramp
        3350, 3550, 3650, 3650, 3350, 2950,  # 12-17: Peak 1 at 15:00 (AC load)
        2550, 2350, 2450, 2750, 3100, 3250   # 18-23: Secondary peak after 23:00!
    ])

    base_hourly_profile = (winter_factor * winter_hourly[hours] +
                           summer_factor * summer_hourly[hours])

    # Temperature correlation:
    # Moderate cooling above 24°C; sharp non-linear surge above 35°C (hockey-stick curve)
    cooling_linear = np.maximum(0.0, temperature - 24.0) * 40.0
    cooling_above_35 = np.maximum(0.0, temperature - 35.0) ** 1.60 * 48.0
    total_cooling = cooling_linear + cooling_above_35

    # Winter mild heating degree load when temperature drops below 15°C
    heating_load = np.maximum(0.0, 15.0 - temperature) * 35.0

    # Monsoon dip (Jul-Sep): rain cooling and humidity reduce daytime AC load
    monsoon_cooling_dip = np.where(is_monsoon, 0.88, 1.0)

    # Brief autumn rise (Oct): post-monsoon clear skies, warm afternoons, festive commercial preparation
    autumn_boost = np.where(is_autumn, 1.05, 1.0)

    # Weekday vs weekend: commercial and institutional demand drop
    # Saturday: -4%, Sunday: -8%
    weekend_factor = np.where(days_of_week == 6, 0.92, np.where(days_of_week == 5, 0.96, 1.00))

    # Base grid load before noise
    grid_demand_raw = (base_load + base_hourly_profile + total_cooling + heating_load) * monsoon_cooling_dip * autumn_boost * weekend_factor

    # Realistic random noise (~2-4% of value) with 3-hour smoothing
    noise_pct = rng.normal(0, 0.022, size=n)
    smoothed_noise = pd.Series(noise_pct).rolling(3, min_periods=1, center=True).mean().values
    total_grid_mw = grid_demand_raw * (1.0 + smoothed_noise)

    # Bound within realistic boundaries:
    # Winter night minimum load ~2,000 MW, summer peak ~8,500-9,000 MW (up to 8,880 MW)
    total_grid_mw = np.clip(total_grid_mw, 1980.0, 8880.0)

    # Distribute total load across Delhi DISCOM areas and feeders
    all_records = []
    for i, ts in enumerate(timestamps):
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        hour_total = total_grid_mw[i]

        for area_info in DELHI_AREAS:
            area_weight = area_info["weight"]
            # Small realistic local feeder variation (+- 1-2%)
            feeder_noise = rng.normal(0, 8.0)
            area_demand = round(max(50.0, hour_total * area_weight + feeder_noise), 2)

            all_records.append({
                "timestamp": ts_str,
                "demand_mw": area_demand,
                "area": area_info["area"],
                "feeder": area_info["feeder"],
                "is_synthetic": True,
            })

    df = pd.DataFrame(all_records)

    # Save demand dataset to disk
    if output_path is None:
        MOCK_DIR.mkdir(parents=True, exist_ok=True)
        output_path = MOCK_DIR / "synthetic_demand.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Generated synthetic demand dataset: {len(df)} records ({len(timestamps)} hours) saved to {output_path}")

    # Synchronize weather dataset only when generating the primary dataset
    if sync_weather and output_path == (MOCK_DIR / "synthetic_demand.csv"):
        weather_mock_path = MOCK_DIR / "weather_mock.csv"
        weather_df.to_csv(weather_mock_path, index=False)
        print(f"Synchronized correlated synthetic weather dataset saved to {weather_mock_path}")

    return df


def generate_synthetic_renewables(
    start_date: str = "2023-01-01 00:00:00",
    end_date: str = "2024-12-31 23:00:00",
    output_path: Optional[Path] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic hourly solar and renewable generation dataset for Delhi (2 years)."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")

    records = []
    # Delhi rooftop & regional solar capacity ~ 400 MW peak
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
            "is_synthetic": True,
        })

    df = pd.DataFrame(records)

    if output_path is None:
        MOCK_DIR.mkdir(parents=True, exist_ok=True)
        output_path = MOCK_DIR / "synthetic_renewable.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Generated synthetic renewable dataset: {len(df)} records saved to {output_path}")

    return df


if __name__ == "__main__":
    generate_synthetic_demand()
    generate_synthetic_renewables()

