"""Naive baseline forecasters for electricity demand benchmarking.

Provides simple, transparent baseline predictions:
- Previous day persistence (t-24h): Today's 14:00 demand = Yesterday's 14:00 demand.
- Previous week persistence (t-168h): Today's 14:00 demand = Same day last week's 14:00 demand.
"""

from typing import Literal, Optional, Union
import numpy as np
import pandas as pd


class NaiveBaselineForecaster:
    """Naive persistence baseline model for time series demand forecasting.

    Parameters:
        strategy: 'day' for 24-hour lookback (t-24h),
                  'week' for 168-hour lookback (t-168h).
    """

    def __init__(self, strategy: Literal["day", "week"] = "day"):
        if strategy not in ["day", "week"]:
            raise ValueError("Strategy must be either 'day' (24h) or 'week' (168h).")
        self.strategy = strategy
        self.lag_hours = 24 if strategy == "day" else 168
        self.lag_col = f"lag_{self.lag_hours}h"

    def fit(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None):
        """Fit method for scikit-learn API compatibility (no-op for naive baseline)."""
        return self

    def predict(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
        """Generate naive baseline predictions.

        If DataFrame contains precomputed lag column ('lag_24h' or 'lag_168h'),
        uses it directly. Otherwise, shifts demand_mw or series by lag_hours.
        """
        if isinstance(X, pd.DataFrame):
            if self.lag_col in X.columns:
                return X[self.lag_col].to_numpy(dtype=float)
            elif "demand_mw" in X.columns:
                return X["demand_mw"].shift(self.lag_hours).to_numpy(dtype=float)
            else:
                raise ValueError(
                    f"DataFrame must contain either '{self.lag_col}' or 'demand_mw' to compute naive baseline."
                )
        elif isinstance(X, pd.Series):
            return X.shift(self.lag_hours).to_numpy(dtype=float)
        elif isinstance(X, np.ndarray):
            if len(X) <= self.lag_hours:
                raise ValueError(f"Array length ({len(X)}) must be greater than lag ({self.lag_hours}).")
            result = np.full_like(X, fill_value=np.nan, dtype=float)
            result[self.lag_hours:] = X[:-self.lag_hours]
            return result
        else:
            raise TypeError("Input X must be a pandas DataFrame, Series, or NumPy array.")


def predict_naive_baseline(
    df: pd.DataFrame,
    strategy: Literal["day", "week"] = "day",
) -> np.ndarray:
    """Convenience function to generate naive baseline predictions from a DataFrame."""
    model = NaiveBaselineForecaster(strategy=strategy)
    return model.predict(df)


if __name__ == "__main__":
    test_df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=48, freq="h"),
        "demand_mw": np.linspace(4000, 6000, 48),
        "lag_24h": [np.nan] * 24 + list(np.linspace(4000, 6000, 24)),
    })
    preds = predict_naive_baseline(test_df, strategy="day")
    print(f"Naive baseline sample predictions (first 5 non-nan): {preds[24:29]}")
