"""Model training pipeline for Delhi Electricity Demand Prediction System.

Performs:
1. Chronological (non-shuffled) train/validation split.
2. Training of scikit-learn's HistGradientBoostingRegressor.
3. Evaluation against naive baseline on the validation split.
4. Model persistence to models/demand_model.joblib with metadata and feature lists.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features.build_features import build_and_save_feature_matrix, get_feature_columns
from src.models.baseline import predict_naive_baseline
from src.models.evaluate import compare_models, evaluate_predictions

# Storage paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_MATRIX_PATH = PROCESSED_DIR / "feature_matrix.csv"
MODEL_SAVE_PATH = MODELS_DIR / "demand_model.joblib"


def chronological_train_val_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "demand_mw",
    train_ratio: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a strictly chronological, non-shuffled train/validation split.

    Preserves temporal ordering to prevent lookahead bias.
    """
    n_total = len(df)
    split_idx = int(n_total * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    return X_train, X_val, y_train, y_val


def train_demand_model(
    feature_matrix_path: Optional[Path] = None,
    output_model_path: Optional[Path] = None,
    train_ratio: float = 0.80,
    random_state: int = 42,
    max_iter: int = 150,
    learning_rate: float = 0.08,
) -> Dict[str, Any]:
    """Train HistGradientBoostingRegressor and persist to joblib with metadata.

    Returns:
        Dictionary containing the trained model, feature names, and evaluation report.
    """
    matrix_file = feature_matrix_path or FEATURE_MATRIX_PATH
    if not matrix_file.exists():
        print(f"Feature matrix {matrix_file} not found. Building features first...")
        df = build_and_save_feature_matrix(output_path=matrix_file)
    else:
        df = pd.read_csv(matrix_file)

    # Determine available feature columns
    standard_features = get_feature_columns()
    available_features = [f for f in standard_features if f in df.columns]

    if not available_features:
        raise ValueError("None of the expected feature columns found in feature matrix.")

    target_col = "demand_mw"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in feature matrix.")

    print(f"Splitting dataset chronologically ({int(train_ratio * 100)}% train, {int((1 - train_ratio) * 100)}% val)...")
    X_train, X_val, y_train, y_val = chronological_train_val_split(
        df,
        feature_cols=available_features,
        target_col=target_col,
        train_ratio=train_ratio,
    )

    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")
    print(f"Features utilized ({len(available_features)}): {available_features}")

    # Initialize and train model
    print("Training HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Baseline predictions on validation set for comparison
    val_slice = df.iloc[len(X_train):].copy()
    y_val_baseline_24h = predict_naive_baseline(val_slice, strategy="day")
    y_val_baseline_168h = predict_naive_baseline(val_slice, strategy="week")

    # Metrics
    train_metrics = evaluate_predictions(y_train, y_train_pred)
    val_metrics = evaluate_predictions(y_val, y_val_pred)
    baseline_24h_metrics = evaluate_predictions(y_val, y_val_baseline_24h)
    baseline_168h_metrics = evaluate_predictions(y_val, y_val_baseline_168h)

    # Comparison Table
    comparison_df = compare_models(
        y_val,
        {
            "Naive Baseline (t-24h)": y_val_baseline_24h,
            "Naive Baseline (t-168h)": y_val_baseline_168h,
            "HistGradientBoostingRegressor": y_val_pred,
        },
    )

    print("\n--- Validation Performance Comparison ---")
    print(comparison_df.to_string(index=False))
    print(f"Train MAE: {train_metrics['mae']} MW | Validation MAE: {val_metrics['mae']} MW (MAPE: {val_metrics['mape']}%)")

    # Prepare model payload
    target_path = output_model_path or MODEL_SAVE_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_type": "HistGradientBoostingRegressor",
        "trained_at": datetime.now().isoformat(),
        "target_col": target_col,
        "feature_names": available_features,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "baseline_24h_metrics": baseline_24h_metrics,
        "baseline_168h_metrics": baseline_168h_metrics,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "random_state": random_state,
        },
    }

    bundle = {
        "model": model,
        "feature_names": available_features,
        "metadata": metadata,
    }

    try:
        joblib.dump(bundle, target_path)
        print(f"\nModel and metadata saved successfully to {target_path}")
    except Exception as save_err:
        print(f"\nWarning: could not write {target_path} ({save_err}). In-memory model bundle will be used.")

    return {
        "model": model,
        "feature_names": available_features,
        "metadata": metadata,
        "comparison_table": comparison_df,
    }


if __name__ == "__main__":
    train_demand_model()
