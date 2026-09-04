# Delhi Electricity Demand Prediction System (delhi-electricity-demand)

An AI-based short-term electricity demand prediction, alert, and grid stability monitoring system designed for the Delhi power grid.

---

## 📁 Project Structure

```plaintext
delhi-electricity-demand/
├── app.py                     # Streamlit application entry point
├── config.py                  # Configuration loader module
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore file
├── config/
│   └── config.yaml            # Grid parameters & threshold configurations
├── data/
│   ├── raw/                   # Raw historical data
│   ├── processed/             # Cleaned and feature-ready data
│   └── mock/                  # Synthetic/mock datasets
├── models/                    # Saved trained model artifacts (.joblib)
├── src/
│   ├── __init__.py
│   ├── data/                  # Data loaders & weather fetchers
│   ├── features/              # Feature engineering & transformations
│   ├── models/                # ML training & model logic
│   ├── forecast/              # Forecast orchestration engine
│   ├── alerts/                # Peak load & threshold alert system
│   └── dashboard/             # Dashboard data connectors
├── dashboard/
│   ├── components.py          # Streamlit UI metric cards and banners
│   └── charts.py              # Plotly chart generation utilities
└── tests/                     # Unit and integration test suite
```

---

## ⚙️ Configuration (`config/config.yaml`)

```yaml
grid_capacity_mw: 9000
warning_threshold: 0.85
critical_threshold: 0.95
forecast_horizon_hours: 24
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Configuration Tests
```bash
python tests/test_config.py
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
