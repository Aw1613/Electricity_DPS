# AI-Based Electricity Demand Prediction System for Delhi
# COMPLETE LOGICAL WORKFLOW
## Hackathon Implementation Workflow — Developer Ready

---

# 0. WORKFLOW OBJECTIVE

The application predicts Delhi electricity demand for:

- Next 24 hours
- Next 7 days

using:

- Historical electricity demand
- Temperature
- Humidity
- Time/calendar features
- Lagged demand
- Rolling demand statistics
- Optional renewable/solar generation
- Optional area/feeder data

The workflow is designed for a small student team and a hackathon prototype.

---

# 1. COMPLETE END-TO-END WORKFLOW

```text
APPLICATION START
      │
      ▼
Load Configuration
      │
      ├── Grid Capacity
      ├── Alert Thresholds
      ├── Forecast Horizon
      └── Data Paths
      │
      ▼
Check Data Sources
      │
      ├── Historical Load Data
      ├── Weather Data
      ├── Renewable Data (optional)
      └── Area/Feeder Data (optional)
      │
      ▼
Load / Fetch Data
      │
      ▼
DATA VALIDATION
      │
      ├── Required Columns?
      ├── Valid Timestamps?
      ├── Duplicate Records?
      ├── Missing Values?
      ├── Valid Demand?
      ├── Valid Temperature?
      └── Time-Series Gaps?
      │
      ▼
DATA ALIGNMENT
      │
      ├── Normalize timestamps
      ├── Convert to hourly frequency
      ├── Align demand + weather
      └── Handle missing hours
      │
      ▼
DATA PREPROCESSING
      │
      ├── Sort timestamps
      ├── Fill small gaps
      ├── Handle outliers
      ├── Remove unusable records
      └── Prevent future-data leakage
      │
      ▼
FEATURE ENGINEERING
      │
      ├── Time Features
      ├── Weather Features
      ├── Lag Features
      ├── Rolling Features
      └── Optional Solar/Holiday Features
      │
      ▼
MODEL TRAINING
      │
      ├── Naive Baseline
      ├── HistGradientBoostingRegressor
      ├── Time-Series Validation
      ├── MAE / RMSE / MAPE
      └── Select Model
      │
      ▼
Save Trained Model
      │
      ▼
FORECAST ENGINE
      │
      ├── Next 24 Hours
      └── Next 7 Days
      │
      ▼
PEAK DEMAND DETECTION
      │
      ├── Maximum Predicted Demand
      ├── Peak Date/Time
      └── Capacity Utilization
      │
      ▼
ALERT ENGINE
      │
      ├── Normal
      ├── Warning
      └── Critical
      │
      ▼
OPTIONAL ANALYTICS
      │
      ├── Area/Feeder Analysis
      └── Renewable Net Demand
      │
      ▼
FORECAST STATE / STORAGE
      │
      ▼
BACKEND / SERVICE LAYER
      │
      ▼
STREAMLIT DASHBOARD
      │
      ▼
USER SEES:
      ├── Current Demand
      ├── 24h Forecast
      ├── 7-Day Forecast
      ├── Predicted Peak
      ├── Actual vs Predicted
      ├── Temperature
      ├── Capacity Utilization
      ├── Alerts
      ├── Area Demand
      └── Net Demand
```

---

# 2. APPLICATION STARTUP WORKFLOW

## Step 1 — Start Application

User runs:

```bash
streamlit run app.py
```

`app.py` initializes the application.

---

## Step 2 — Load Configuration

Read:

```text
config/config.yaml
```

Example:

```yaml
grid_capacity_mw: 9000
warning_threshold: 0.85
critical_threshold: 0.95
forecast_hours: 24
forecast_days: 7
```

Configuration must contain:

```text
grid capacity
alert thresholds
data locations
forecast settings
```

Never hard-code these values inside the alert logic.

---

## Step 3 — Initialize Services

Initialize:

```text
Data Loader
Weather Loader
Preprocessor
Feature Engineer
Model
Forecast Engine
Alert Engine
Analytics
```

Then check whether a trained model already exists.

```text
models/demand_model.joblib
```

If it exists:

```text
Load model
```

If it does not exist:

```text
Prepare data
→ Train model
→ Validate model
→ Save model
```

---

# 3. STAGE 1 — DATA COLLECTION

## 3.1 Historical Electricity Demand

Preferred input:

```text
Delhi SLDC
Grid-India / POSOCO
data.gov.in
other reliable public historical load reports
```

Input may be:

```text
CSV
Excel
API
Downloaded report
```

Minimum required fields:

```text
timestamp
demand_mw
```

Example:

```text
timestamp,demand_mw
2026-08-01 00:00,4100
2026-08-01 01:00,3950
2026-08-01 02:00,3800
```

---

## 3.2 Weather Collection

Use:

```text
Open-Meteo API
```

Collect:

```text
temperature
humidity
apparent temperature
precipitation
wind speed
```

For this project, the most important weather variable is:

```text
temperature
```

Weather data is required for:

```text
historical training
future forecasting
```

---

## 3.3 Optional Renewable Data

If available:

```text
timestamp
solar_generation_mw
renewable_generation_mw
```

Otherwise:

```text
renewable module = disabled
```

---

## 3.4 Optional Area/Feeder Data

If available:

```text
timestamp
area
feeder
demand_mw
```

Otherwise:

```text
area module = disabled
```

---

## 3.5 Real Data Unavailable

Fallback hierarchy:

```text
Real historical load
       ↓ unavailable
Cached historical load
       ↓ unavailable
Synthetic/mock load
```

For weather:

```text
Open-Meteo
       ↓ unavailable
Cached weather
       ↓ unavailable
Historical weather profile
       ↓ unavailable
Synthetic/mock weather
```

Dashboard must clearly display:

```text
Data Mode: REAL
```

or:

```text
Data Mode: DEMONSTRATION / SYNTHETIC
```

Synthetic data must never be presented as official Delhi electricity data.

---

# 4. STAGE 2 — DATA VALIDATION

Validation happens immediately after loading data.

---

## 4.1 Required Column Check

Demand dataset must contain:

```text
timestamp
demand_mw
```

Weather dataset must contain at least:

```text
timestamp
temperature
```

If a required column is missing:

```text
Stop that data pipeline
      ↓
Log error
      ↓
Use fallback dataset if available
```

Do not silently continue with an incomplete dataset.

---

## 4.2 Timestamp Validation

Check:

```text
Is timestamp parseable?
Is timezone known?
Are timestamps in expected range?
```

Convert using:

```python
pd.to_datetime()
```

Invalid timestamp:

```text
flag/remove record
```

---

## 4.3 Duplicate Records

Check:

```text
duplicate timestamp
```

If exact duplicate:

```text
keep one
```

If multiple different demand values exist for the same timestamp:

```text
resolve using source rules
OR
aggregate if appropriate
OR
flag for review
```

Do not blindly average conflicting records without understanding the source.

---

## 4.4 Missing Demand Values

Small gaps:

```text
interpolate
```

Large gaps:

```text
exclude affected training period
```

For forecasting, if the latest demand value required for a lag is missing:

```text
use last reliable value
OR
fallback to previous-day/previous-week baseline
```

---

## 4.5 Invalid Demand Values

Examples:

```text
negative demand
NaN
infinite values
obviously corrupt values
```

Action:

```text
negative/invalid → flag as invalid
NaN → missing-value pipeline
infinite → replace/remove
```

Do not remove high demand simply because it is high. A genuine summer peak can be a valid observation.

---

## 4.6 Impossible Temperature Values

Check for clearly invalid values such as:

```text
NaN
infinite
physically impossible/corrupt values
```

If suspicious:

```text
mark missing
```

Then:

```text
interpolate
OR
historical average
OR
cached value
```

---

## 4.7 Time-Series Gap Detection

Expected frequency:

```text
1 hour
```

Check:

```text
timestamp[t] - timestamp[t-1]
```

If gap > 1 hour:

```text
record missing interval
```

Small gaps:

```text
fill/interpolate
```

Large gaps:

```text
keep flagged
exclude from training where necessary
```

---

# 5. STAGE 3 — DATA ALIGNMENT

Demand and weather can use different timestamps.

Example:

```text
Demand:
10:00
11:00
12:00

Weather:
09:45
10:45
11:45
12:45
```

They must be converted to a common hourly timeline.

---

## 5.1 Normalize Timestamp

Apply:

```text
parse datetime
→ normalize timezone
→ sort
```

All data must use the same timezone.

For Delhi:

```text
Asia/Kolkata
```

---

## 5.2 Create Common Hourly Timeline

Create an hourly index:

```text
10:00
11:00
12:00
13:00
...
```

Use hourly demand as the primary timeline.

---

## 5.3 Resample Demand

If source data is 15-minute:

```text
00:00
00:15
00:30
00:45
```

convert to:

```text
00:00
01:00
02:00
```

using:

```text
mean
```

If the dataset represents instantaneous/peak readings, use the source's appropriate aggregation rule instead of automatically averaging.

---

## 5.4 Resample Weather

Convert weather data to the same hourly timestamps.

Example:

```text
weather 10:00
weather 11:00
weather 12:00
```

Then merge:

```text
demand + weather
```

using timestamp.

---

## 5.5 Missing Weather Hour

If weather is missing:

```text
interpolate short gaps
```

If unavailable:

```text
historical average
```

If API is unavailable:

```text
cached weather
```

---

## 5.6 Final Aligned Dataset

Example:

```text
timestamp
demand_mw
temperature
humidity
```

Later feature engineering adds:

```text
hour
day_of_week
lag_1h
lag_24h
lag_168h
rolling_24h
...
```

---

# 6. STAGE 4 — DATA PREPROCESSING

## 6.1 Sort

Always:

```python
df = df.sort_values("timestamp")
```

---

## 6.2 Missing-Value Treatment

Recommended:

```text
Demand:
small gap → interpolation
large gap → exclude affected training section

Weather:
small gap → interpolation
larger gap → historical average/cache

Optional solar:
small gap → interpolation
unavailable → disable renewable feature
```

---

## 6.3 Outlier Handling

Use:

```text
IQR
rolling statistics
domain checks
```

But:

```text
Do NOT automatically delete real electricity peaks.
```

Recommended approach:

```text
Detect
   ↓
Flag
   ↓
Check plausibility
   ↓
Keep genuine peak
```

---

## 6.4 Scaling / Normalization

The recommended tree-based model:

```text
HistGradientBoostingRegressor
```

does not require feature scaling.

Therefore:

```text
No normalization required for MVP.
```

If Linear Regression or another scale-sensitive model is added:

```text
StandardScaler
```

can be used.

Scaler must be fitted only on training data.

---

## 6.5 Train/Test Split

Never randomly shuffle time-series data.

Correct:

```text
Earlier data
     ↓
TRAIN
     ↓
VALIDATION
     ↓
TEST
     ↓
Most recent data
```

Example:

```text
70% chronological → Train
15% chronological → Validation
15% chronological → Test
```

---

## 6.6 Prevent Data Leakage

This is critical.

At training time:

```text
Feature at time t
```

can only use information available at or before time `t`.

Never use:

```text
future demand
future actual weather
future target values
```

when generating training features.

For rolling features:

```python
rolling_mean = demand.shift(1).rolling(24).mean()
```

The shift ensures the current target is not accidentally included.

For future forecasting:

```text
future weather = weather forecast
```

not actual future weather observations.

---

# 7. STAGE 5 — FEATURE ENGINEERING

Feature engineering converts raw data into model inputs.

---

# 7.1 TIME FEATURES

## Hour

```text
0–23
```

Why:

Electricity demand follows daily usage patterns.

Example:

```text
night → low
morning → increasing
afternoon → high
evening → high
```

---

## Day of Week

```text
Monday
Tuesday
...
Sunday
```

Why:

Workdays and weekends have different demand patterns.

---

## Weekend

```text
0 = weekday
1 = weekend
```

Why:

Weekend electricity consumption can differ from working days.

---

## Month

```text
1–12
```

Why:

Electricity demand changes by season.

---

## Season

Example:

```text
Winter
Summer
Monsoon
```

Why:

Delhi electricity demand is strongly affected by seasonal weather.

---

## Day

Day of month can optionally capture monthly patterns.

---

# 7.2 WEATHER FEATURES

## Temperature

Most important weather feature.

High temperature can increase:

```text
air-conditioner usage
cooling demand
fan usage
```

Therefore:

```text
temperature ↑
→ electricity demand often ↑
```

---

## Humidity

High humidity can increase perceived heat and cooling requirements.

---

## Optional Apparent Temperature

Can represent perceived heat more directly than temperature alone.

---

# 7.3 DEMAND HISTORY FEATURES

## Previous Hour Demand

```text
lag_1h
```

Why:

Demand at the previous hour is a strong indicator of near-future demand.

---

## Previous Day Same-Hour Demand

```text
lag_24h
```

Why:

Demand at 3 PM today is often related to demand at 3 PM yesterday.

---

## Previous Week Same-Hour Demand

```text
lag_168h
```

Why:

Captures weekly patterns.

```text
168 hours = 7 days
```

---

# 7.4 ROLLING FEATURES

## Rolling 3-Hour Demand

```text
rolling_mean_3h
```

Captures recent short-term demand behavior.

---

## Rolling 24-Hour Demand

```text
rolling_mean_24h
```

Captures recent daily demand level.

---

## Rolling 7-Day Demand

```text
rolling_mean_168h
```

Captures broader demand trend.

---

# 7.5 OPTIONAL FEATURES

## Solar Generation

```text
solar_generation_mw
```

Can reduce net demand during daytime.

---

## Holiday Indicator

```text
is_holiday
```

Useful because holidays may behave differently from normal weekdays.

---

# 7.6 Final Feature Vector

Example:

```text
hour
day
day_of_week
is_weekend
month
season
temperature
humidity
lag_1h
lag_24h
lag_168h
rolling_mean_3h
rolling_mean_24h
rolling_mean_168h
solar_generation_mw (optional)
is_holiday (optional)
```

Target:

```text
demand_mw
```

---

# 8. STAGE 6 — MODEL TRAINING

## 8.1 Create Baseline First

Use a naive baseline:

```text
Predicted demand at time t
=
demand at t-24h
```

For weekly comparison:

```text
Predicted demand
=
demand at t-168h
```

This establishes whether the ML model actually improves forecasting.

---

## 8.2 Train Recommended Model

Primary model:

```text
HistGradientBoostingRegressor
```

Workflow:

```text
Clean historical data
       ↓
Feature engineering
       ↓
Chronological split
       ↓
Train model
       ↓
Validation
       ↓
Calculate metrics
```

---

## 8.3 Model Input

```text
X =
time features
+
weather features
+
lag features
+
rolling features
+
optional renewable features
```

Target:

```text
y = demand_mw
```

---

## 8.4 Validation

Calculate:

```text
MAE
RMSE
MAPE
```

Compare:

```text
Baseline metrics
vs
ML model metrics
```

Choose the model with better validation performance while considering stability and simplicity.

---

## 8.5 Save Model

Save:

```text
models/demand_model.joblib
```

Also save:

```text
feature_names.json
model_metrics.json
training_metadata.json
```

Example metadata:

```text
training_start
training_end
model_name
MAE
RMSE
MAPE
feature_list
```

---

## 8.6 Retraining

Do NOT retrain after every dashboard refresh.

Recommended:

```text
New data arrives
       ↓
Store new data
       ↓
Check retraining schedule
       ↓
Retrain periodically
```

For hackathon:

```text
Manual "Retrain Model" button
```

can be provided.

Production-like behavior:

```text
Daily or weekly retraining
```

depending on data availability and drift.

---

# 9. STAGE 7 — FORECAST GENERATION

# 9.1 NEXT-DAY FORECAST

Goal:

```text
24 hourly predictions
```

Workflow:

```text
Latest historical demand
        ↓
Latest weather
        ↓
Future 24-hour weather forecast
        ↓
Create future timestamps
        ↓
Create calendar features
        ↓
Create lag features
        ↓
Model prediction
        ↓
24-hour forecast
```

---

# 9.2 Future Feature Generation

For each future hour:

```text
timestamp
hour
day
day_of_week
weekend
month
season
temperature forecast
humidity forecast
```

Then generate demand-history features.

---

# 9.3 Recursive Lag Handling

At the first future timestamp:

```text
lag_1h = latest actual demand
lag_24h = actual demand from 24h earlier
lag_168h = actual demand from 168h earlier
```

After predicting the first future hour:

```text
prediction t+1
```

becomes available for the next recursive step.

Example:

```text
Predict t+1
    ↓
Use prediction t+1 as lag_1h
    ↓
Predict t+2
    ↓
Use prediction t+2 as lag_1h
    ↓
...
```

For lag_24h and lag_168h:

```text
Use actual historical values when available.
Use earlier predictions once the forecast horizon passes beyond the historical boundary.
```

---

# 9.4 Rolling Feature Handling During Forecast

At each future timestamp:

```text
rolling_mean_3h
rolling_mean_24h
rolling_mean_168h
```

must be calculated using:

```text
available historical demand
+
previously generated predictions
```

Never use actual future demand because it is not known at prediction time.

---

# 9.5 NEXT-WEEK FORECAST

Goal:

```text
168 hourly predictions
```

Workflow:

```text
Future 7-day timestamps
        ↓
Future weather forecast
        ↓
Calendar features
        ↓
Historical demand features
        ↓
Recursive predictions
        ↓
168 predictions
        ↓
7-day forecast
```

---

# 9.6 Forecast Output

Store:

```text
timestamp
predicted_demand_mw
forecast_horizon
model_version
```

Example:

```text
2026-09-05 10:00, 8200, 24h, v1
2026-09-05 11:00, 8350, 24h, v1
...
```

---

# 10. STAGE 8 — PEAK DEMAND DETECTION

After generating predictions:

```text
Predicted values
      ↓
Find maximum
```

Example:

```text
Predicted peak = 8700 MW
Peak time = 2026-09-05 15:00
```

---

## 10.1 Capacity Utilization

Given:

```text
Grid Capacity = 9000 MW
Predicted Peak = 8700 MW
```

Calculate:

```text
8700 / 9000 × 100
=
96.67%
```

Therefore:

```text
Capacity utilization = 96.67%
```

---

## 10.2 Configurable Classification

Use:

```text
warning_threshold = 0.85
critical_threshold = 0.95
```

Logic:

```text
if utilization < warning_threshold:
    NORMAL

elif utilization < critical_threshold:
    WARNING

else:
    CRITICAL
```

For example:

```text
< 85%       → Normal
85%–<95%    → Warning
≥ 95%       → Critical
```

These are demonstration defaults only. The actual configured grid-capacity and operational thresholds should come from the project configuration/source assumptions.

---

# 11. STAGE 9 — ALERT GENERATION

## 11.1 Alert Object

Create:

```text
alert_id
timestamp
forecast_timestamp
predicted_demand_mw
grid_capacity_mw
utilization_percent
severity
message
```

---

## 11.2 Normal Alert

Condition:

```text
utilization < warning threshold
```

Message:

```text
Demand is within safe operating range.
```

---

## 11.3 Warning Alert

Condition:

```text
warning threshold ≤ utilization < critical threshold
```

Message:

```text
Predicted demand is approaching grid capacity.
```

---

## 11.4 Critical Alert

Condition:

```text
utilization ≥ critical threshold
```

Message:

```text
Predicted demand is near/exceeding grid capacity. Load balancing may be required.
```

---

## 11.5 Alert Storage

For MVP:

```text
alerts.json
```

or:

```text
SQLite
```

is sufficient.

Store alert history so the dashboard can show:

```text
Current alert
Previous alerts
Alert timeline
```

---

# 12. STAGE 10 — AREA / FEEDER ANALYSIS

This feature is optional.

---

## 12.1 If Real Area Data Exists

Input:

```text
timestamp
area
feeder
demand_mw
```

Workflow:

```text
Total Delhi Demand
        ↓
Group by Area
        ↓
Calculate area demand
        ↓
Sort descending
        ↓
Identify high-demand areas
        ↓
Display chart
```

Example:

```text
Area           Demand
North Delhi    2100 MW
South Delhi    1900 MW
East Delhi     1700 MW
West Delhi     1600 MW
```

---

## 12.2 Area Ranking

Calculate:

```text
area_demand = sum(demand_mw)
```

Then:

```text
sort descending
```

Dashboard:

```text
1. North Delhi
2. South Delhi
3. East Delhi
4. West Delhi
```

---

## 12.3 If Real Area Data Does Not Exist

Use a clearly labeled synthetic/demo dataset.

Example:

```text
Delhi total demand
        ↓
Synthetic allocation percentages
        ↓
North / South / East / West
```

The dashboard must show:

```text
Area Data: DEMONSTRATION / SYNTHETIC
```

Never claim:

```text
North Delhi = 2100 MW
```

is a real feeder measurement unless supported by real data.

---

# 13. STAGE 11 — RENEWABLE ADJUSTMENT

This module is optional.

If solar generation exists:

```text
Gross Demand
      -
Solar Generation
      =
Net Demand
```

Example:

```text
Gross Demand      = 8500 MW
Solar Generation  =  900 MW
Net Demand        = 7600 MW
```

---

## 13.1 Dashboard Values

Show:

```text
Gross Demand
Solar Generation
Net Demand
Renewable Contribution %
```

Calculate:

```text
renewable contribution =
solar_generation / gross_demand × 100
```

---

## 13.2 If Renewable Data Is Missing

Do not estimate it silently.

Display:

```text
Renewable adjustment unavailable for current dataset.
```

---

# 14. STAGE 12 — DASHBOARD DATA FLOW

Final application flow:

```text
Prediction Engine
       ↓
Forecast State / Storage
       ↓
Service Layer / Backend
       ↓
Streamlit Dashboard
```

For the MVP, the backend can simply be Python service functions rather than a separate HTTP server.

---

# 15. DASHBOARD OUTPUT

## 15.1 Current Demand

Show:

```text
Current Demand
```

Value:

```text
latest actual demand
```

If no recent actual data exists:

```text
Latest available demand
```

---

## 15.2 Predicted Peak

Show:

```text
Predicted Peak Demand
Peak Date
Peak Time
```

---

## 15.3 Next 24-Hour Forecast

Plot:

```text
timestamp
actual demand
predicted demand
```

For future hours:

```text
predicted demand only
```

---

## 15.4 Next 7-Day Forecast

Show:

```text
7-day hourly/daily forecast
```

For readability, allow:

```text
hourly view
daily aggregated view
```

---

## 15.5 Actual vs Predicted

For historical validation:

```text
Actual Demand
vs
Predicted Demand
```

This demonstrates model performance.

---

## 15.6 Temperature

Show:

```text
Current Temperature
Forecast Temperature
Temperature vs Demand
```

---

## 15.7 Capacity Utilization

Show:

```text
Predicted Peak
Grid Capacity
Utilization %
Risk Level
```

---

## 15.8 Alerts

Show:

```text
Normal
Warning
Critical
```

and the corresponding message.

---

## 15.9 Area Demand

If available:

```text
Area ranking
Area demand chart
```

---

## 15.10 Renewable Net Demand

If available:

```text
Gross Demand
Solar
Net Demand
```

---

# 16. STAGE 13 — CONTINUOUS UPDATE

When new actual electricity demand arrives:

```text
New Actual Demand
       ↓
Validate
       ↓
Store
       ↓
Merge with historical dataset
       ↓
Compare with previous prediction
       ↓
Calculate forecast error
       ↓
Update dashboard
       ↓
Check retraining schedule
       ↓
Retrain if required
```

---

# 17. FORECAST ERROR WORKFLOW

For each completed forecast:

```text
Predicted Demand
       +
Actual Demand
       ↓
Calculate Error
```

Example:

```text
Predicted = 8500 MW
Actual    = 8600 MW

Absolute Error = 100 MW
```

Store:

```text
timestamp
predicted
actual
absolute_error
percentage_error
```

Aggregate metrics:

```text
MAE
RMSE
MAPE
```

---

# 18. MODEL RETRAINING WORKFLOW

Recommended hackathon behavior:

```text
New data arrives
       ↓
Store new observation
       ↓
Update historical dataset
       ↓
Check "Retrain Model" button
       ↓
Rebuild features
       ↓
Chronological train/validation/test
       ↓
Train baseline
       ↓
Train HistGradientBoosting
       ↓
Evaluate
       ↓
Compare metrics
       ↓
Save improved model
       ↓
Update model metadata
```

Do not replace a working model with a worse model automatically.

---

# 19. DETAILED NUMBERED WORKFLOW

## 1. Application starts

```text
streamlit run app.py
```

## 2. Load configuration

Read:

```text
grid capacity
thresholds
forecast horizon
data paths
```

## 3. Check datasets

Check:

```text
historical load
weather
renewable
area
```

## 4. Load historical demand

Read CSV/API/report-converted dataset.

## 5. Fetch weather

Call Open-Meteo for historical/current/forecast weather.

## 6. Activate fallback if required

Use:

```text
cache
historical weather
mock data
```

when necessary.

## 7. Validate data

Check:

```text
columns
timestamps
duplicates
missing values
demand validity
temperature validity
time gaps
```

## 8. Normalize timestamps

Convert everything to:

```text
Asia/Kolkata
```

and standard datetime format.

## 9. Resample

Convert demand/weather to hourly resolution.

## 10. Align

Merge demand and weather by hourly timestamp.

## 11. Clean

Handle:

```text
missing values
outliers
invalid records
```

## 12. Build historical features

Create:

```text
time
weather
lag
rolling
optional solar/holiday
```

## 13. Split chronologically

Create:

```text
train
validation
test
```

## 14. Train baseline

Use previous-day/previous-week demand.

## 15. Train ML model

Use:

```text
HistGradientBoostingRegressor
```

## 16. Evaluate

Calculate:

```text
MAE
RMSE
MAPE
```

## 17. Select model

Choose best validated model.

## 18. Save model

Save:

```text
demand_model.joblib
```

## 19. Generate future timestamps

Create:

```text
next 24 hours
next 168 hours
```

## 20. Get future weather

Use Open-Meteo forecast.

## 21. Build future calendar features

Generate:

```text
hour
day
weekday
weekend
month
season
```

## 22. Generate future lag features

Use:

```text
historical actual demand
+
previous predictions
```

## 23. Generate predictions

Run model recursively.

## 24. Detect peak

Find maximum predicted demand.

## 25. Calculate utilization

```text
peak / grid_capacity × 100
```

## 26. Generate alert

Classify:

```text
Normal
Warning
Critical
```

## 27. Calculate optional analytics

If data exists:

```text
area demand
renewable net demand
```

## 28. Store forecast state

Save:

```text
forecast
peak
alerts
metrics
```

## 29. Backend/service layer returns data

Dashboard requests the prepared data.

## 30. Dashboard renders results

Display:

```text
Current Demand
24h Forecast
7d Forecast
Peak
Weather
Alerts
Actual vs Predicted
Area
Net Demand
```

---

# 20. NEXT-DAY FORECAST DATA FLOW

```text
LATEST ACTUAL DATA
       │
       ├── Latest Demand
       ├── Previous 24h
       └── Previous 168h
       │
       ▼
FUTURE 24 TIMESTAMPS
       │
       ▼
OPEN-METEO WEATHER FORECAST
       │
       ├── Temperature
       └── Humidity
       │
       ▼
CALENDAR FEATURES
       │
       ├── Hour
       ├── Day
       ├── Weekday
       ├── Weekend
       └── Month/Season
       │
       ▼
LAG FEATURES
       │
       ├── lag_1h
       ├── lag_24h
       └── lag_168h
       │
       ▼
ROLLING FEATURES
       │
       ├── 3h
       ├── 24h
       └── 7d
       │
       ▼
HISTGRADIENTBOOSTING MODEL
       │
       ▼
24 HOURLY PREDICTIONS
       │
       ▼
PEAK DETECTION
       │
       ▼
ALERT ENGINE
       │
       ▼
DASHBOARD
```

---

# 21. NEXT-WEEK FORECAST DATA FLOW

```text
LATEST HISTORICAL DATA
       │
       ▼
CREATE 168 FUTURE HOURLY TIMESTAMPS
       │
       ▼
GET 7-DAY WEATHER FORECAST
       │
       ▼
CREATE CALENDAR FEATURES
       │
       ▼
START RECURSIVE FORECAST
       │
       ▼
Predict Hour +1
       │
       ▼
Add Prediction to Temporary Demand History
       │
       ▼
Generate Features for Hour +2
       │
       ▼
Predict Hour +2
       │
       ▼
Repeat
       │
       ▼
Until Hour +168
       │
       ▼
168 HOURLY PREDICTIONS
       │
       ▼
Aggregate to Daily View if required
       │
       ▼
Find Weekly Peak
       │
       ▼
Capacity Utilization
       │
       ▼
Alerts
       │
       ▼
Dashboard
```

---

# 22. ALERT WORKFLOW

```text
Forecast Generated
       ↓
Find Peak Demand
       ↓
Load Grid Capacity
       ↓
Calculate:

utilization =
predicted_peak /
grid_capacity
× 100
       ↓
Compare Threshold
       │
       ├── < 85%
       │       ↓
       │    NORMAL
       │
       ├── 85%–<95%
       │       ↓
       │    WARNING
       │
       └── ≥95%
               ↓
            CRITICAL
       │
       ▼
Create Alert Object
       ↓
Store Alert
       ↓
Display Dashboard Alert
```

---

# 23. AREA-WISE WORKFLOW

```text
Area/Feeder Dataset
       ↓
Validate Location Fields
       ↓
Align Timestamps
       ↓
Aggregate Demand
       ↓
Group by Area
       ↓
Calculate Total/Peak Demand
       ↓
Sort Descending
       ↓
Identify High-Demand Areas
       ↓
Generate Ranking
       ↓
Dashboard Chart
```

If no real area data:

```text
No real area data
       ↓
Synthetic/demo allocation
       ↓
Clearly label as DEMONSTRATION DATA
       ↓
Dashboard
```

---

# 24. RENEWABLE WORKFLOW

```text
Gross Electricity Demand
       +
Solar/Renewable Generation
       ↓
Timestamp Alignment
       ↓
Validate Generation Data
       ↓
Calculate:

Net Demand =
Gross Demand - Renewable Generation
       ↓
Calculate Renewable Contribution
       ↓
Store
       ↓
Dashboard
       │
       ├── Gross Demand
       ├── Renewable Generation
       └── Net Demand
```

---

# 25. ERROR / RETRAINING WORKFLOW

```text
Forecast Generated
       ↓
Wait for Actual Demand
       ↓
Actual Demand Arrives
       ↓
Match Timestamp
       ↓
Predicted vs Actual
       ↓
Calculate Error
       ↓
Store Error
       ↓
Update MAE/RMSE/MAPE
       ↓
Check Model Performance
       ↓
Is Retraining Required?
       │
       ├── NO
       │    ↓
       │  Continue Current Model
       │
       └── YES
            ↓
        Rebuild Dataset
            ↓
        Rebuild Features
            ↓
        Train Model
            ↓
        Validate
            ↓
        Compare With Existing Model
            ↓
        Save Only If Better/Acceptable
```

---

# 26. FAILURE / FALLBACK WORKFLOW

## 26.1 Open-Meteo Failure

```text
Open-Meteo API
      ↓ failure
Cached Weather
      ↓ unavailable
Historical Weather Profile
      ↓ unavailable
Mock Weather
      ↓
Continue Forecast
```

Dashboard:

```text
Weather Source: FALLBACK
```

---

## 26.2 Historical Load Failure

```text
Real Load Data
      ↓ unavailable
Cached Load Data
      ↓ unavailable
Synthetic Load Data
      ↓
Train/Demo Mode
```

Dashboard:

```text
Data Mode: SYNTHETIC DEMO
```

---

## 26.3 Model Prediction Failure

```text
ML Model
   ↓ failure
Naive Previous-Day Baseline
   ↓
Generate Forecast
```

Dashboard:

```text
Forecast Mode: BASELINE FALLBACK
```

---

## 26.4 Missing Weather Values

```text
Missing Weather
      ↓
Short Gap?
      │
      ├── YES → Interpolate
      │
      └── NO → Historical Average / Cache
```

---

## 26.5 Missing Demand Values

```text
Missing Demand
      ↓
Small Gap?
      │
      ├── YES → Interpolate
      │
      └── NO → Exclude Affected Training Window
```

---

# 27. RECOMMENDED CODE MODULE MAPPING

Each workflow stage should map directly to code.

```text
STAGE                  MODULE

Configuration          config/config.yaml

Load ingestion         src/ingestion/load_loader.py

Weather ingestion      src/ingestion/weather_loader.py

Renewable ingestion    src/ingestion/renewable_loader.py

Area ingestion         src/ingestion/area_loader.py

Mock fallback          src/ingestion/mock_data_generator.py

Validation             src/preprocessing/validator.py

Cleaning               src/preprocessing/cleaner.py

Alignment              src/preprocessing/alignment.py

Features               src/features/feature_engineering.py

Baseline               src/models/baseline.py

Training               src/models/train.py

Evaluation             src/models/evaluate.py

Prediction             src/models/predict.py

Forecast               src/forecasting/forecast_engine.py

Peak analysis          src/analytics/peak_analysis.py

Area analysis          src/analytics/area_analysis.py

Renewable analysis     src/analytics/renewable_analysis.py

Alerts                 src/alerts/alert_engine.py

Service layer          src/services/demand_service.py

Dashboard              app.py

Charts                 dashboard/charts.py

UI components          dashboard/components.py
```

---

# 28. RECOMMENDED INTERNAL FUNCTION FLOW

```python
main()
    ↓
load_config()
    ↓
load_demand_data()
    ↓
load_weather_data()
    ↓
load_optional_data()
    ↓
validate_datasets()
    ↓
align_demand_weather()
    ↓
clean_data()
    ↓
build_features()
    ↓
if model_missing:
    train_model()
    evaluate_model()
    save_model()
    ↓
generate_next_day_forecast()
    ↓
generate_next_week_forecast()
    ↓
detect_peaks()
    ↓
calculate_capacity_utilization()
    ↓
generate_alerts()
    ↓
calculate_area_analysis()
    ↓
calculate_renewable_adjustment()
    ↓
prepare_dashboard_state()
    ↓
render_dashboard()
```

---

# 29. RECOMMENDED FORECAST STATE

Keep a single structured forecast state.

Example:

```python
forecast_state = {
    "current_demand_mw": ...,
    "next_24h": ...,
    "next_7d": ...,
    "predicted_peak_mw": ...,
    "peak_timestamp": ...,
    "capacity_utilization": ...,
    "alert_level": ...,
    "alert_message": ...,
    "temperature_forecast": ...,
    "model_metrics": ...,
    "area_analysis": ...,
    "renewable_analysis": ...
}
```

The dashboard reads from this state instead of independently recalculating the forecast.

---

# 30. FINAL LOGICAL WORKFLOW

```text
START
  │
  ▼
CONFIGURATION
  │
  ▼
DATA SOURCE CHECK
  │
  ├───────────────┬────────────────┐
  ▼               ▼                ▼
LOAD DATA       WEATHER        OPTIONAL DATA
  │               │                │
  └───────────────┴────────────────┘
                  │
                  ▼
             VALIDATION
                  │
                  ▼
             ALIGNMENT
                  │
                  ▼
            PREPROCESSING
                  │
                  ▼
         FEATURE ENGINEERING
                  │
                  ▼
             BASELINE
                  │
                  ▼
          ML MODEL TRAINING
                  │
                  ▼
              VALIDATION
                  │
                  ▼
           MODEL SELECTION
                  │
                  ▼
             SAVE MODEL
                  │
                  ▼
        ┌─────────┴─────────┐
        ▼                   ▼
    NEXT 24H             NEXT 7 DAYS
        │                   │
        └─────────┬─────────┘
                  ▼
          PEAK DETECTION
                  │
                  ▼
       CAPACITY UTILIZATION
                  │
                  ▼
            ALERT ENGINE
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
   AREA ANALYSIS       RENEWABLE ANALYSIS
   (optional)              (optional)
        │                    │
        └─────────┬──────────┘
                  ▼
          FORECAST STATE
                  │
                  ▼
         SERVICE/BACKEND
                  │
                  ▼
          STREAMLIT UI
                  │
                  ▼
        FINAL PREDICTION
                  │
                  ▼
     USER SEES FORECAST + RISK
                  │
                  ▼
       NEW ACTUAL DATA ARRIVES
                  │
                  ▼
        ERROR CALCULATION
                  │
                  ▼
        PERIODIC RETRAINING
                  │
                  └──────────────► UPDATE MODEL
```

---

# 31. 10-HOUR HACKATHON IMPLEMENTATION WORKFLOW

Only implement the following first:

```text
1. Load historical demand CSV
2. Load/fetch Open-Meteo weather
3. Add fallback mock data
4. Validate timestamps/columns
5. Resample to hourly
6. Merge demand + weather
7. Create essential features
8. Train naive baseline
9. Train HistGradientBoostingRegressor
10. Calculate MAE/RMSE/MAPE
11. Save model
12. Generate 24h forecast
13. Generate 7-day forecast
14. Detect peak
15. Calculate capacity utilization
16. Generate Normal/Warning/Critical alert
17. Build Streamlit dashboard
18. Add actual-vs-predicted chart
19. Add weather chart
20. Add failure/fallback indicators
```

Then add only if time remains:

```text
21. Area/feeder analysis
22. Renewable adjustment
23. Prediction uncertainty range
24. Advanced explainability
25. SQLite storage
26. FastAPI
```

---

# 32. CORE SUCCESS CRITERIA

The prototype is successful if this complete path works reliably:

```text
Historical Demand
       +
Temperature
       +
Time Features
       +
Lag Features
       +
Rolling Features
       ↓
ML Model
       ↓
24-Hour Forecast
       +
7-Day Forecast
       ↓
Peak Detection
       ↓
Capacity Utilization
       ↓
Alert
       ↓
Professional Dashboard
```

The most important principle is:

```text
WORKING END-TO-END SYSTEM
>
COMPLEX INFRASTRUCTURE
```

For the hackathon, prioritize:

```text
Accuracy
Reliability
Explainability
Fast execution
Clear visualization
Graceful fallback
```

Avoid unnecessary:

```text
Microservices
Kubernetes
Cloud infrastructure
Authentication
Message queues
Separate React frontend
Complex databases
Advanced MLOps
```

The final system should be modular enough that each stage can be implemented, tested, and replaced independently while still allowing the complete pipeline to run locally with one command.
