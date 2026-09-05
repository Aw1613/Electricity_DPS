"""Inference and prediction engine for Delhi Electricity Demand Prediction System.

Exposes:
- load_demand_model: Loads trained model, feature names, and metadata from models/demand_model.joblib.
- predict_demand: Computes single or batch demand predictions (in MW) from feature inputs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

# Default model path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "demand_model.joblib"

# Module-level cache to avoid repeated disk reads
_CACHED_MODEL_BUNDLE: Optional[Dict[str, Any]] = None
_CACHED_MODEL_PATH: Optional[Path] = None


def load_demand_model(
    model_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """Load the trained demand forecasting model bundle.

    Args:
        model_path: Path to demand_model.joblib.
        force_reload: If True, reloads from disk bypassing memory cache.

    Returns:
        Tuple of (model, feature_names, metadata).
    """
    global _CACHED_MODEL_BUNDLE, _CACHED_MODEL_PATH
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not target_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {target_path}. Please run 'src/models/train.py' first."
        )

    if not force_reload and _CACHED_MODEL_BUNDLE is not None and _CACHED_MODEL_PATH == target_path:
        bundle = _CACHED_MODEL_BUNDLE
    else:
        # Cross-version numpy pickle compatibility shim (numpy 1.x vs 2.x)
        import sys
        try:
            import numpy as _np
            # Map numpy._core to numpy.core if running on numpy 1.x
            if hasattr(_np, "core") and not hasattr(_np, "_core"):
                sys.modules["numpy._core"] = _np.core
                sys.modules["numpy._core.multiarray"] = _np.core.multiarray
            # Map numpy.core to numpy._core if running on numpy 2.x
            elif hasattr(_np, "_core") and not hasattr(_np, "core"):
                sys.modules["numpy.core"] = _np._core
                sys.modules["numpy.core.multiarray"] = _np._core.multiarray
        except Exception:
            pass

        bundle = None
        if target_path.exists():
            try:
                bundle = joblib.load(target_path)
            except Exception as load_err:
                import logging
                logging.warning("Failed to unpickle %s (%s). Retraining model on the fly...", target_path, load_err)
                bundle = None

        if bundle is None:
            # Automatic fallback: train model on the fly
            from src.models.train import train_demand_model
            trained_payload = train_demand_model(output_model_path=target_path)
            bundle = {
                "model": trained_payload["model"],
                "feature_names": trained_payload["feature_names"],
                "metadata": trained_payload.get("metadata", {}),
            }

        _CACHED_MODEL_BUNDLE = bundle
        _CACHED_MODEL_PATH = target_path

    # Handle structured bundle vs raw estimator
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        feature_names = bundle.get("feature_names", [])
        metadata = bundle.get("metadata", {})
    else:
        model = bundle
        feature_names = getattr(model, "feature_names_in_", []).tolist() if hasattr(model, "feature_names_in_") else []
        metadata = {}

    return model, feature_names, metadata


def predict_demand(
    features: Union[pd.DataFrame, pd.Series, Dict[str, Any], np.ndarray, List[Dict[str, Any]]],
    model_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Compute single or batch demand predictions (in MW) from input feature vectors.

    Args:
        features: Input feature representations (DataFrame, Series, dict, or list of dicts).
        model_path: Optional custom model file path.

    Returns:
        1D NumPy array of predicted electricity demand in MW.
    """
    model, feature_names, _ = load_demand_model(model_path=model_path)

    # 1. Convert various input formats into a standardized DataFrame
    if isinstance(features, dict):
        df = pd.DataFrame([features])
    elif isinstance(features, pd.Series):
        df = pd.DataFrame([features.to_dict()])
    elif isinstance(features, list) and len(features) > 0 and isinstance(features[0], dict):
        df = pd.DataFrame(features)
    elif isinstance(features, np.ndarray):
        if features.ndim == 1:
            df = pd.DataFrame([features], columns=feature_names if feature_names else None)
        else:
            df = pd.DataFrame(features, columns=feature_names if feature_names else None)
    elif isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        raise TypeError(f"Unsupported features input type: {type(features)}")

    # 2. Align feature columns with model training order
    if feature_names:
        missing = [col for col in feature_names if col not in df.columns]
        if missing:
            raise ValueError(f"Input features missing required columns: {missing}")
        X = df[feature_names]
    else:
        X = df

    # 3. Predict
    predictions = model.predict(X)
    return np.asarray(predictions, dtype=float)


if __name__ == "__main__":
    try:
        loaded_model, feat_names, meta = load_demand_model()
        print(f"Model loaded: {meta.get('model_type', 'Estimator')}")
        print(f"Required features count: {len(feat_names)}")
    except FileNotFoundError as e:
        print(e)
