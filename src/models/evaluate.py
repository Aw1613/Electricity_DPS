"""Evaluation metrics engine for electricity demand forecasting models.

Calculates:
- MAE  (Mean Absolute Error in MW): Average magnitude of forecast errors.
- RMSE (Root Mean Squared Error in MW): Penalizes large peak forecast misses.
- MAPE (Mean Absolute Percentage Error in %): Relative accuracy as a percentage of demand.
"""

from typing import Any, Dict, Union
import numpy as np
import pandas as pd


def _clean_arrays(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to 1D float arrays and filter out any NaN entries."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true ({len(yt)}) vs y_pred ({len(yp)}).")

    valid_mask = (~np.isnan(yt)) & (~np.isnan(yp))
    if not np.any(valid_mask):
        raise ValueError("No valid overlapping non-NaN pairs found to evaluate.")

    return yt[valid_mask], yp[valid_mask]


def calculate_mae(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """Calculate Mean Absolute Error (MAE) in MW."""
    yt, yp = _clean_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def calculate_rmse(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """Calculate Root Mean Squared Error (RMSE) in MW."""
    yt, yp = _clean_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def calculate_mape(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    epsilon: float = 1e-6,
) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE) in %."""
    yt, yp = _clean_arrays(y_true, y_pred)
    # Avoid zero division
    denominator = np.where(np.abs(yt) < epsilon, epsilon, np.abs(yt))
    return float(np.mean(np.abs((yt - yp) / denominator)) * 100.0)


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """Calculate MAE, RMSE, and MAPE and return as a dictionary."""
    return {
        "mae": round(calculate_mae(y_true, y_pred), 2),
        "rmse": round(calculate_rmse(y_true, y_pred), 2),
        "mape": round(calculate_mape(y_true, y_pred), 2),
    }


def compare_models(
    y_true: Union[pd.Series, np.ndarray],
    predictions_dict: Dict[str, Union[pd.Series, np.ndarray]],
) -> pd.DataFrame:
    """Generate a side-by-side comparison table of models evaluated against ground truth."""
    rows = []
    for model_name, y_pred in predictions_dict.items():
        metrics = evaluate_predictions(y_true, y_pred)
        rows.append({
            "Model": model_name,
            "MAE (MW)": metrics["mae"],
            "RMSE (MW)": metrics["rmse"],
            "MAPE (%)": metrics["mape"],
        })

    df = pd.DataFrame(rows)
    return df.sort_values("MAE (MW)").reset_index(drop=True)


if __name__ == "__main__":
    y_actual = np.array([5000.0, 5200.0, 4800.0, 5100.0])
    y_pred_baseline = np.array([4900.0, 5000.0, 4700.0, 4950.0])
    y_pred_ml = np.array([4980.0, 5190.0, 4810.0, 5090.0])

    comparison = compare_models(
        y_actual,
        {
            "Naive Baseline": y_pred_baseline,
            "HistGradientBoosting": y_pred_ml,
        },
    )
    print("Sample Model Evaluation Comparison:")
    print(comparison.to_string(index=False))
