# Delhi Electricity Demand Prediction System (delhi-electricity-demand)

An AI-driven short-term electricity load forecasting, peak detection, and grid stability monitoring platform designed specifically for the Delhi power grid (calibrated to Delhi SLDC 8,000+ MW summer peak load).

---

## 🚀 Quick Start

### 1. Requirements & Setup
Ensure Python 3.10+ is installed. Clone the repository and install dependencies:

```bash
git clone git@github.com:Aw1613/Electricity_DPS.git
cd Electricity_DPS
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard
Launch the interactive web UI:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 💾 Live Mode vs. Offline Demo Mode

The platform features a **Zero-Network Offline Demo Mode** accessible right from the sidebar:

- **🟢 Live Data Mode (Default)**:
  - Fetches real-time hourly temperature, humidity, wind speed, and precipitation forecasts for Delhi (`28.61° N, 77.23° E`) directly from the **Open-Meteo API**.
  - Merges real-time weather into feature transformations for rolling predictions.

- **💾 Offline Demo Mode**:
  - Activated via the sidebar switch: **"💾 Offline Demo Mode (Zero Network)"**.
  - Completely bypasses external HTTP requests.
  - Automatically loads local historical demand (`data/mock/synthetic_demand.csv`), pre-cached weather matrices (`data/raw/weather_cache.csv`), and local pre-trained ML models (`models/demand_model.joblib`).
  - Guarantees 100% reliability during pitch presentations and offline evaluations.

### Data Status Badges
The top navigation header displays dynamic status indicators:
- `🟢 Live Data` — Active Open-Meteo network API connection
- `🟡 Cached Weather` — Graceful fallback to cached weather matrix on network disruption
- `💾 Synthetic Demo Mode` — Full offline execution using local mock data

---

## 📊 Standardized Engineering Units

All analytics, charts, and key performance indicators adhere strictly to utility industry standards:
- **Electricity Demand & Capacity**: Megawatts (`MW`)
- **Grid Utilization**: Percentage (`%`)
- **Temperature & Apparent Temp**: Degrees Celsius (`°C`)
- **Relative Humidity**: Percentage (`%`)
- **Wind Speed**: Kilometers per Hour (`km/h`)
- **Precipitation**: Millimeters (`mm`)

---

## 📁 Project Architecture

```plaintext
delhi-electricity-demand/
├── app.py                         # Streamlit command center UI
├── config.py                      # YAML configuration loader & validation
├── config/
│   └── config.yaml                # Capacity limits & alert thresholds
├── data/
│   ├── raw/                       # Raw feeds & cached weather matrices
│   ├── processed/                 # Leak-free feature-engineered datasets
│   └── mock/                      # Calibrated synthetic data (demand, weather, solar)
├── models/
│   └── demand_model.joblib        # Pre-trained HistGradientBoosting model
├── src/
│   ├── data/
│   │   ├── weather_api.py         # Open-Meteo fetcher with offline fallbacks
│   │   ├── data_loader.py         # Resilient multi-source data loading
│   │   ├── preprocessor.py        # Timestamp parsing & gap filling
│   │   └── validator.py           # Range checks & data integrity validation
│   ├── features/
│   │   ├── build_features.py      # 24-feature pipeline (lags, rolling, interactions)
│   │   ├── area_analysis.py       # Zonal DISCOM disaggregation (BRPL, BYPL, TPDDL)
│   │   └── renewables.py          # Net Demand (Gross - Solar) analysis
│   ├── models/
│   │   ├── baseline.py            # Persistence & seasonal 24h baselines
│   │   ├── train.py               # TimeSeriesSplit cross-validation & training
│   │   └── evaluate.py            # MAE, RMSE, MAPE, R2 metrics
│   ├── forecast/
│   │   ├── predict_24h.py         # 24-hour day-ahead hourly forecast
│   │   ├── predict_7d.py          # 168-hour recursive multi-step forecaster
│   │   └── analyze.py             # Peak detection & uncertainty bands (±MAPE)
│   ├── alerts/
│   │   └── alert_manager.py       # Dynamic threshold engine (Normal, Warning, Critical)
│   └── services/
│       └── demand_service.py      # Unified orchestration service layer
├── dashboard/
│   ├── components.py              # KPI cards, alert banners, tables
│   └── charts.py                  # High-performance Plotly visualizations
└── tests/                         # 40-test automated verification suite
```

---

## 🧪 Test Suite

Run the full pytest suite with 40 unit and integration tests:

```bash
pytest tests/
```

Test coverage includes:
- **Alert Threshold Classification**: Boundary validations (`<85%`, `85%–95%`, `≥95%`).
- **Offline Resilience**: Simulated API failure, network timeout handling, and local mock fallback.
- **Feature Pipeline Dimensions**: Strict 24-column feature matrix integrity.
- **Recursive Forecasting**: 168-hour multi-step lag updates and daily summaries.
- **Renewables & Zonal Analytics**: Net demand calculations and DISCOM area disaggregations.

---

## ⚡ Key Grid Parameters (`config/config.yaml`)

| Parameter | Default Value | Description |
|---|---|---|
| `grid_capacity_mw` | `9000` | Delhi maximum transmission handling capacity |
| `warning_threshold` | `0.85` | Warning alert trigger (85% utilization / 7,650 MW) |
| `critical_threshold` | `0.95` | Critical alert trigger (95% utilization / 8,550 MW) |
| `forecast_horizon_hours` | `24` | Primary operational planning horizon |
