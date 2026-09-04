# AI-Based Electricity Demand Prediction System - Development Task Plan
**For Execution by AI Agent (Google Antigravity)**

## IMPORTANT INSTRUCTIONS FOR AI AGENT
1. **Work task-by-task.** Do NOT attempt to build the entire project blindly in one step.
2. **Before implementing a task:** Inspect the existing project, check completed tasks, identify dependencies.
3. **Implement only the current task.** Run/test it, fix errors, confirm DONE CRITERIA, then move to the next task.
4. **Prioritize a WORKING DEMO.** If real-world data blocks progress, immediately use realistic synthetic data.
5. **No over-engineering.** The goal is a 10-hour Hackathon MVP. 

---

## EXECUTION ORDER & DEPENDENCY GRAPH
`PROJECT SETUP` -> `DATA GENERATION` -> `DATA INGESTION` -> `PREPROCESSING` -> `FEATURE ENGINEERING` -> `BASELINE MODEL` -> `ML MODEL` -> `FORECAST ENGINE` -> `ALERT ENGINE` -> `DASHBOARD` -> `AREA ANALYSIS` -> `RENEWABLE MODULE` -> `TESTING` -> `DEMO MODE` -> `FINAL POLISH`

---

## PHASE 0 — PROJECT INITIALIZATION

### TASK 0.1: Project Setup
* **PRIORITY:** P0
* **PURPOSE:** Initialize project structure and environment.
* **INPUTS:** None
* **OUTPUTS:** Standardized project structure.
* **FILES TO CREATE/MODIFY:** `requirements.txt`, `README.md`, `.gitignore`, `config.py` or `config.yaml`, folder structure (`data/`, `src/data/`, `src/features/`, `src/models/`, `src/forecast/`, `src/alerts/`, `src/dashboard/`, `tests/`).
* **IMPLEMENTATION DETAILS:** Create necessary directories and placeholder files. Define dependencies (pandas, numpy, scikit-learn, openmeteo-requests, streamlit).
* **DEPENDENCIES:** None
* **TESTING METHOD:** Verify folder structure and successful `pip install -r requirements.txt`.
* **DONE CRITERIA:** All folders exist, environment can be initialized without errors.

---

## PHASE 1 — DATA LAYER

### TASK 1.1: Create Synthetic Historical Dataset
* **PRIORITY:** P0
* **PURPOSE:** Generate realistic hourly electricity demand patterns to prevent blocking on data acquisition.
* **INPUTS:** Date range configuration.
* **OUTPUTS:** `synthetic_demand.csv`
* **FILES TO CREATE/MODIFY:** `src/data/generate_synthetic.py`
* **IMPLEMENTATION DETAILS:** Generate timestamp, total_demand_MW, optional zone/feeder. Must include morning increase, afternoon demand, evening peak, night reduction, weekday/weekend, and seasonal variations. Introduce slight random noise.
* **DEPENDENCIES:** TASK 0.1
* **TESTING METHOD:** Plot generated data to visually confirm realistic load curves.
* **DONE CRITERIA:** CSV generated with non-random-looking realistic daily/weekly seasonality.

### TASK 1.2: Weather Data Ingestion
* **PRIORITY:** P0
* **PURPOSE:** Fetch real or forecasted weather data using Open-Meteo.
* **INPUTS:** Latitude/Longitude of Delhi.
* **OUTPUTS:** Weather DataFrame (timestamps, temp, humidity).
* **FILES TO CREATE/MODIFY:** `src/data/weather_api.py`
* **IMPLEMENTATION DETAILS:** Build integration with Open-Meteo API. Implement fallback behavior returning cached/mock weather data if API fails.
* **DEPENDENCIES:** TASK 0.1
* **TESTING METHOD:** Execute API call and verify DataFrame structure. Disconnect internet and verify fallback works.
* **DONE CRITERIA:** Reliable retrieval of temperature and humidity matching demand timestamps.

### TASK 1.3: Data Loading Utilities
* **PRIORITY:** P0
* **PURPOSE:** Unified loader for historical and weather data.
* **INPUTS:** CSV paths, API connectors.
* **OUTPUTS:** Standardized pandas DataFrames.
* **FILES TO CREATE/MODIFY:** `src/data/data_loader.py`
* **IMPLEMENTATION DETAILS:** Functions to load mock data, real CSVs, and API data consistently.
* **DEPENDENCIES:** TASK 1.1, TASK 1.2
* **TESTING METHOD:** Run loader functions and assert output shapes and column names.
* **DONE CRITERIA:** Data can be loaded interchangeably via single function calls.

---

## PHASE 2 — PREPROCESSING

### TASK 2.1: Preprocessing Pipeline
* **PRIORITY:** P0
* **PURPOSE:** Clean and align data for modeling.
* **INPUTS:** Raw DataFrames from loader.
* **OUTPUTS:** Cleaned, combined DataFrame.
* **FILES TO CREATE/MODIFY:** `src/data/preprocessing.py`
* **IMPLEMENTATION DETAILS:** Timestamp conversion, sorting, deduplication, missing value imputation (forward fill/interpolate), hourly resampling, and merging weather with load data.
* **DEPENDENCIES:** TASK 1.3
* **TESTING METHOD:** Pass data with missing rows and duplicates; verify clean output.
* **DONE CRITERIA:** Pipeline outputs a continuous, gapless, hourly DataFrame containing both demand and weather.

### TASK 2.2: Validation Utilities
* **PRIORITY:** P0
* **PURPOSE:** Ensure data integrity before modeling.
* **INPUTS:** Cleaned DataFrame.
* **OUTPUTS:** Validation boolean and error messages.
* **FILES TO CREATE/MODIFY:** `src/data/validator.py`
* **IMPLEMENTATION DETAILS:** Check demand range (>0, <Max), timestamp continuity, missing values, required columns.
* **DEPENDENCIES:** TASK 2.1
* **TESTING METHOD:** Feed intentionally corrupted data and assert exceptions/warnings are raised.
* **DONE CRITERIA:** Clear validation results returned for good and bad data.

---

## PHASE 3 — FEATURE ENGINEERING

### TASK 3.1: Feature Engineering Module
* **PRIORITY:** P0
* **PURPOSE:** Create predictive features from raw data.
* **INPUTS:** Cleaned DataFrame.
* **OUTPUTS:** Feature-rich DataFrame for ML.
* **FILES TO CREATE/MODIFY:** `src/features/build_features.py`
* **IMPLEMENTATION DETAILS:** 
  - Calendar: hour, day, weekday, weekend, month, season.
  - Demand lags: lag_1h, lag_24h, lag_168h.
  - Rolling stats: rolling_3h, rolling_24h, rolling_168h.
  - Optional: solar gen, holiday boolean.
  *Crucial: Reusable for both training and single-step/multi-step prediction without data leakage.*
* **DEPENDENCIES:** TASK 2.1
* **TESTING METHOD:** Verify lag_24h on current row matches demand from 24 rows prior.
* **DONE CRITERIA:** Feature module transforms basic dataframe into model-ready X and y arrays.

---

## PHASE 4 — MODEL

### TASK 4.1: Baseline Model
* **PRIORITY:** P0
* **PURPOSE:** Establish a performance floor.
* **INPUTS:** Feature DataFrame.
* **OUTPUTS:** Baseline predictions, MAE, RMSE, MAPE.
* **FILES TO CREATE/MODIFY:** `src/models/baseline.py`
* **IMPLEMENTATION DETAILS:** Implement a naive model (e.g., previous-day same-hour demand).
* **DEPENDENCIES:** TASK 3.1
* **TESTING METHOD:** Run evaluation metrics on a hold-out test set.
* **DONE CRITERIA:** Baseline metrics are computed and logged.

### TASK 4.2: Train Primary ML Model
* **PRIORITY:** P0
* **PURPOSE:** Develop the core forecasting model.
* **INPUTS:** Feature DataFrame (Train/Test split).
* **OUTPUTS:** Trained model object.
* **FILES TO CREATE/MODIFY:** `src/models/train.py`
* **IMPLEMENTATION DETAILS:** Use scikit-learn (RandomForest, HistGradientBoosting) or Prophet for fast, explainable results. Optimize for MVP constraints (fast training).
* **DEPENDENCIES:** TASK 3.1
* **TESTING METHOD:** Train model and output predictions on test set.
* **DONE CRITERIA:** Model trains successfully without crashing.

### TASK 4.3: Model Evaluation
* **PRIORITY:** P1
* **PURPOSE:** Compare ML model against Baseline.
* **INPUTS:** True values, Baseline preds, ML preds.
* **OUTPUTS:** Metrics dictionary (MAE, RMSE, MAPE).
* **FILES TO CREATE/MODIFY:** `src/models/evaluate.py`
* **IMPLEMENTATION DETAILS:** Calculate evaluation metrics and format for dashboard consumption.
* **DEPENDENCIES:** TASK 4.1, TASK 4.2
* **TESTING METHOD:** Compare Baseline MAE vs ML MAE (ML should ideally be lower).
* **DONE CRITERIA:** Evaluation metrics successfully generated.

### TASK 4.4: Model Persistence
* **PRIORITY:** P0
* **PURPOSE:** Save model to disk to prevent retraining on dashboard reload.
* **INPUTS:** Trained model object.
* **OUTPUTS:** `model.pkl` or `model.joblib`
* **FILES TO CREATE/MODIFY:** `src/models/train.py` (update), `src/models/predict.py`
* **IMPLEMENTATION DETAILS:** Serialize using `joblib`. Add a load function for the inference pipeline.
* **DEPENDENCIES:** TASK 4.2
* **TESTING METHOD:** Save model, delete from memory, load from disk, and run prediction.
* **DONE CRITERIA:** Model seamlessly loaded from disk for inference.

---

## PHASE 5 — FORECAST ENGINE

### TASK 5.1: Next 24-Hour Forecasting Function
* **PRIORITY:** P0
* **PURPOSE:** Generate the core next-day forecast.
* **INPUTS:** Latest historical demand, weather forecast, loaded model.
* **OUTPUTS:** DataFrame with `timestamp` and `predicted_demand_MW`.
* **FILES TO CREATE/MODIFY:** `src/forecast/predict_24h.py`
* **IMPLEMENTATION DETAILS:** Iterative or direct multi-output prediction for 24 hours. Must cleanly handle feature engineering for future timesteps.
* **DEPENDENCIES:** TASK 3.1, TASK 4.4
* **TESTING METHOD:** Assert output contains exactly 24 rows with future timestamps.
* **DONE CRITERIA:** 24-hour forecast returns correct format reliably.

### TASK 5.2: Next 7-Day Forecasting Function
* **PRIORITY:** P1
* **PURPOSE:** Generate weekly trend forecast.
* **INPUTS:** Latest historical data, 7-day weather forecast.
* **OUTPUTS:** Daily/Hourly predictions for 7 days.
* **FILES TO CREATE/MODIFY:** `src/forecast/predict_7d.py`
* **IMPLEMENTATION DETAILS:** Similar to 24h but extended horizon. Daily aggregation is acceptable if hourly drift is too high.
* **DEPENDENCIES:** TASK 5.1
* **TESTING METHOD:** Assert output covers 7 days.
* **DONE CRITERIA:** Returns a stable 7-day trend.

### TASK 5.3: Peak Demand Detection
* **PRIORITY:** P0
* **PURPOSE:** Identify critical demand points in the forecast.
* **INPUTS:** 24h Forecast output.
* **OUTPUTS:** Dictionary (peak MW, peak timestamp).
* **FILES TO CREATE/MODIFY:** `src/forecast/analyze.py`
* **IMPLEMENTATION DETAILS:** `argmax()` on predicted demand.
* **DEPENDENCIES:** TASK 5.1
* **TESTING METHOD:** Verify returned peak matches maximum value in the forecast array.
* **DONE CRITERIA:** Accurately extracts peak demand and time.

---

## PHASE 6 — ALERT ENGINE

### TASK 6.1: Configurable Grid Capacity & Alert Logic
* **PRIORITY:** P0
* **PURPOSE:** Warn operators of grid stress.
* **INPUTS:** Peak demand, `GRID_CAPACITY_MW` (config).
* **OUTPUTS:** Alert status (Normal, Warning, Critical), utilization %.
* **FILES TO CREATE/MODIFY:** `src/alerts/alert_manager.py`
* **IMPLEMENTATION DETAILS:** 
  - Normal: <85%
  - Warning: 85-95%
  - Critical: >95%
  Do NOT hard-code `GRID_CAPACITY_MW`. Read from config.
* **DEPENDENCIES:** TASK 0.1, TASK 5.3
* **TESTING METHOD:** Pass synthetic peak values representing 80%, 90%, and 98% capacity; assert correct statuses.
* **DONE CRITERIA:** Alert engine returns proper severity and message based on config.

---

## PHASE 7 & 8 — SECONDARY MODULES (P1/P2)

### TASK 7.1: Area-wise Demand Analysis
* **PRIORITY:** P1
* **PURPOSE:** Show breakdown of demand by zone.
* **INPUTS:** Synthetic/Demo area data.
* **OUTPUTS:** Area aggregated metrics.
* **FILES TO CREATE/MODIFY:** `src/features/area_analysis.py`
* **IMPLEMENTATION DETAILS:** Create clear synthetic data if real is missing. Rank areas by demand.
* **DEPENDENCIES:** TASK 1.1
* **DONE CRITERIA:** Aggregated rankings available for UI.

### TASK 8.1: Renewable (Solar) Adjustment
* **PRIORITY:** P1
* **PURPOSE:** Calculate Net Demand (Gross - Solar).
* **INPUTS:** Gross demand, simulated solar generation curve.
* **OUTPUTS:** Net Demand MW.
* **FILES TO CREATE/MODIFY:** `src/features/renewables.py`
* **IMPLEMENTATION DETAILS:** Generate a daylight bell-curve for synthetic solar if real data is missing. Clearly label as simulated.
* **DEPENDENCIES:** TASK 1.1
* **DONE CRITERIA:** Accurate subtraction of solar from gross demand.

*Note: Phase 7 Feeder-level prediction (P2) is skipped for this MVP unless time permits.*

---

## PHASE 9 & 10 — UI / DASHBOARD

### TASK 9.1: Streamlit Dashboard UI
* **PRIORITY:** P0
* **PURPOSE:** Interactive UI for the Hackathon Demo.
* **INPUTS:** Outputs from all Phase 4, 5, 6, 7, 8 modules.
* **OUTPUTS:** Running Streamlit application.
* **FILES TO CREATE/MODIFY:** `src/dashboard/app.py`
* **IMPLEMENTATION DETAILS:** 
  - Header: AI-Based Electricity Demand Prediction
  - KPI Cards: Current Demand, Predicted Peak, Capacity, Utilization, Status.
  - Section 1: 24h Line chart (Actual vs Predicted).
  - Section 2 (P1): 7-day trend.
  - Section 3: Weather metrics/relationship.
  - Section 4: Peak & Alerts with visual severity colors.
  - Section 5 (P1): Area & Renewables charts.
  - Section 6 (P1): Model Performance.
  *Keep logic out of UI files.*
* **DEPENDENCIES:** TASK 5.1, 5.3, 6.1
* **TESTING METHOD:** Run `streamlit run src/dashboard/app.py` and interact with the UI.
* **DONE CRITERIA:** Fast, clean, error-free dashboard rendering local data.

---

## PHASE 11 & 12 — DEMO RELIABILITY

### TASK 11.1: Core Unit Tests
* **PRIORITY:** P1
* **PURPOSE:** Prevent regressions during rapid hacking.
* **FILES TO CREATE/MODIFY:** `tests/test_data.py`, `tests/test_models.py`, `tests/test_alerts.py`
* **IMPLEMENTATION DETAILS:** Test missing weather, empty datasets, threshold logic.
* **DONE CRITERIA:** `pytest` passes on core logic.

### TASK 12.1: DEMO MODE (Offline Fallback)
* **PRIORITY:** P0
* **PURPOSE:** Guarantee hackathon presentation works without internet.
* **INPUTS:** UI toggle switch.
* **OUTPUTS:** Dashboard running on 100% local mock/cached data.
* **FILES TO CREATE/MODIFY:** `src/dashboard/app.py`, `config.py`
* **IMPLEMENTATION DETAILS:** Implement a "DEMO MODE / LIVE MODE" toggle. In demo mode, bypass all APIs and use `synthetic_demand.csv` and a static mock weather file. Load pre-trained local model.
* **DEPENDENCIES:** TASK 9.1
* **TESTING METHOD:** Disable Wi-Fi. Turn on Demo mode. Verify dashboard functions perfectly.
* **DONE CRITERIA:** Seamless offline operation.

---

## PHASE 13 — FINAL POLISH

### TASK 13.1: Polish & Documentation
* **PRIORITY:** P0
* **PURPOSE:** Make the MVP look professional.
* **FILES TO CREATE/MODIFY:** `README.md`, `src/dashboard/app.py`
* **IMPLEMENTATION DETAILS:** 
  - Add clean UI labels, standard units (MW, °C).
  - Add setup instructions to README.
  - No unnecessary animations.
* **DONE CRITERIA:** Repository is ready for judges to review and run.
