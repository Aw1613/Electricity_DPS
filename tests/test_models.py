"""Unit tests for baseline forecaster, evaluation metrics, model training, and prediction engine."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline import NaiveBaselineForecaster, predict_naive_baseline
from src.models.evaluate import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    evaluate_predictions,
    compare_models,
)
from src.models.train import chronological_train_val_split, train_demand_model
from src.models.predict import load_demand_model, predict_demand


def test_baseline_forecaster():
    """Verify naive persistence forecaster on day and week strategies."""
    # Test day baseline (t-24h)
    df_day = pd.DataFrame({
        "demand_mw": list(range(48)),
        "lag_24h": [np.nan] * 24 + list(range(24)),
    })
    preds_day = predict_naive_baseline(df_day, strategy="day")
    assert np.isnan(preds_day[0])
    assert preds_day[24] == 0.0
    assert preds_day[47] == 23.0

    # Test week baseline (t-168h)
    forecaster_week = NaiveBaselineForecaster(strategy="week")
    df_week = pd.DataFrame({
        "demand_mw": list(range(200)),
        "lag_168h": [np.nan] * 168 + list(range(32)),
    })
    preds_week = forecaster_week.predict(df_week)
    assert np.isnan(preds_week[100])
    assert preds_week[168] == 0.0
    assert preds_week[199] == 31.0

    print("PASS: Baseline forecaster tests passed.")


def test_evaluation_metrics():
    """Verify MAE, RMSE, and MAPE calculations with known inputs."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 315.0, 380.0])
    # Errors: +10, -10, +15, -20
    # Absolute errors: 10, 10, 15, 20 -> Mean = 55 / 4 = 13.75
    # Squared errors: 100, 100, 225, 400 = 825 -> RMSE = sqrt(825/4) = sqrt(206.25) ~ 14.3614
    # Pct errors: 10/100, 10/200, 15/300, 20/400 = 0.10, 0.05, 0.05, 0.05 -> Mean = 0.0625 = 6.25%

    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)

    assert np.isclose(mae, 13.75), f"Expected 13.75, got {mae}"
    assert np.isclose(rmse, np.sqrt(206.25)), f"Expected {np.sqrt(206.25)}, got {rmse}"
    assert np.isclose(mape, 6.25), f"Expected 6.25, got {mape}"

    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["mae"] == 13.75
    assert metrics["mape"] == 6.25

    # Test compare_models table
    table = compare_models(
        y_true,
        {"Model A": y_pred, "Model B": y_true},
    )
    assert len(table) == 2
    assert "MAE (MW)" in table.columns
    # Model B has 0 error and should be first
    assert table.iloc[0]["Model"] == "Model B"
    assert table.iloc[0]["MAE (MW)"] == 0.0

    print("PASS: Evaluation metrics test passed.")


def test_chronological_split():
    """Verify chronological split does not shuffle data."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=100, freq="1h"),
        "demand_mw": np.arange(100),
        "feature_1": np.arange(100) * 2,
    })

    X_train, X_val, y_train, y_val = chronological_train_val_split(
        df,
        feature_cols=["feature_1"],
        target_col="demand_mw",
        train_ratio=0.80,
    )

    assert len(X_train) == 80
    assert len(X_val) == 20
    # Strict order check
    assert (X_train["feature_1"].values == np.arange(0, 160, 2)).all()
    assert (X_val["feature_1"].values == np.arange(160, 200, 2)).all()
    assert y_train.max() < y_val.min()

    print("PASS: Chronological split test passed.")


def test_train_and_predict_pipeline():
    """Verify end-to-end model training, file persistence, and inference."""
    test_model_path = PROJECT_ROOT / "models" / "test_demand_model.joblib"
    try:
        # Run training
        train_result = train_demand_model(
            output_model_path=test_model_path,
            max_iter=30,  # Fast test iteration
        )

        assert test_model_path.exists()
        assert "model" in train_result
        assert "metadata" in train_result
        assert train_result["metadata"]["val_metrics"]["mae"] > 0

        # Test predict.py loading
        loaded_model, feature_names, metadata = load_demand_model(model_path=test_model_path)
        assert loaded_model is not None
        assert len(feature_names) > 0

        # Load feature matrix row to test inference
        matrix = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "feature_matrix.csv")
        sample_batch = matrix.iloc[:5]

        # 1. Batch DataFrame prediction
        batch_preds = predict_demand(sample_batch, model_path=test_model_path)
        assert len(batch_preds) == 5
        assert (batch_preds > 0).all()

        # 2. Single row dict prediction
        single_dict = sample_batch.iloc[0].to_dict()
        single_pred = predict_demand(single_dict, model_path=test_model_path)
        assert len(single_pred) == 1
        assert np.isclose(single_pred[0], batch_preds[0])

        print("PASS: Train and predict pipeline test passed.")
    finally:
        if test_model_path.exists():
            test_model_path.unlink()


def test_feature_dimensions():
    """Verify feature dimension alignment, required columns, and schema validation."""
    from src.features.build_features import get_feature_columns

    feature_cols = get_feature_columns()
    assert len(feature_cols) == 24, f"Expected 24 features, got {len(feature_cols)}"

    # Verify model loaded from default path expects matching feature dimensions
    model, model_features, metadata = load_demand_model()
    assert len(model_features) == 24
    assert set(model_features) == set(feature_cols)

    # Test error raised when feature dimension is missing
    incomplete_features = {f: 1.0 for f in feature_cols if f != "temperature"}
    with pytest.raises(ValueError, match="missing required columns"):
        predict_demand(incomplete_features)


if __name__ == "__main__":
    test_baseline_forecaster()
    test_evaluation_metrics()
    test_chronological_split()
    test_train_and_predict_pipeline()
    test_feature_dimensions()
    print("\nAll model layer unit tests completed successfully!")
