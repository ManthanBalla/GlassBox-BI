"""Validation Guards and Compatibility Matrix for Explainability Agent (Phase 6).

Validates model state, method compatibility, background sample integrity, and feature
presence, preventing silent failures and guaranteeing research integrity.
"""

from typing import Dict, List, Optional
import pandas as pd

from backend.app.forecasting.base import BaseForecastModel
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster
from backend.app.forecasting.prophet_model import ProphetForecaster
from backend.app.schemas.explainability import (
    ExplanationMethod,
    MethodCompatibilityInfo,
)


# Explicit Model Compatibility Matrix
MODEL_COMPATIBILITY: Dict[str, MethodCompatibilityInfo] = {
    "lightgbm": MethodCompatibilityInfo(
        model_name="lightgbm",
        supported_methods=["shap", "lime"],
        default_method="shap",
        native_tree_shap=True,
        sequence_attribution=False,
        component_decomposition=False,
        notes="Native TreeExplainer provides exact, efficient Shapley values across tabular calendar, lag, and rolling features.",
    ),
    "lstm": MethodCompatibilityInfo(
        model_name="lstm",
        supported_methods=["shap", "lime"],
        default_method="shap",
        native_tree_shap=False,
        sequence_attribution=True,
        component_decomposition=False,
        notes="Model-agnostic KernelExplainer / sequence attribution over lookback input window (lag_1 to lag_k).",
    ),
    "prophet": MethodCompatibilityInfo(
        model_name="prophet",
        supported_methods=["component_based"],
        default_method="component_based",
        native_tree_shap=False,
        sequence_attribution=False,
        component_decomposition=True,
        notes="Generalized Additive Model: mathematically decomposed into trend, weekly seasonality, yearly seasonality, and holiday effects.",
    ),
}


class ExplainabilityValidator:
    """Pre-explanation assertions and data integrity guards."""

    @staticmethod
    def validate_model_fitted(model: BaseForecastModel) -> None:
        """Asserts that the forecasting model is fully fitted."""
        if not getattr(model, "is_fitted", False):
            raise ValueError(f"Model '{getattr(model, 'model_name', 'unknown')}' is not fitted. Please fit model before explaining.")

    @staticmethod
    def validate_method_compatibility(model_name: str, method: ExplanationMethod) -> str:
        """Verifies that requested explanation method is supported for the model architecture.

        Returns:
            Resolved method name (e.g. 'shap', 'lime', 'component_based').
        """
        norm_name = model_name.lower().strip()
        if norm_name not in MODEL_COMPATIBILITY:
            raise ValueError(f"Unknown forecasting model: '{model_name}'. Supported models: {list(MODEL_COMPATIBILITY.keys())}")

        compat = MODEL_COMPATIBILITY[norm_name]

        if method == ExplanationMethod.AUTO:
            return compat.default_method

        req_method = method.value.lower().strip()
        if req_method not in compat.supported_methods:
            supported = ", ".join(compat.supported_methods)
            raise ValueError(
                f"Method '{method.value}' is not supported for model '{model_name}'. "
                f"Supported methods for {model_name}: [{supported}]. {compat.notes}"
            )

        return req_method

    @staticmethod
    def validate_background_data(background_df: Optional[pd.DataFrame], min_samples: int = 5) -> None:
        """Validates that sufficient reference observations exist for background sampling."""
        if background_df is None or len(background_df) < min_samples:
            count = len(background_df) if background_df is not None else 0
            raise ValueError(
                f"Insufficient reference data for background distribution: requires at least {min_samples} samples, got {count}."
            )

    @staticmethod
    def validate_feature_row(feature_row: pd.DataFrame, expected_features: Optional[List[str]] = None) -> None:
        """Validates the structure and columns of the feature vector to be explained."""
        if feature_row.empty:
            raise ValueError("Feature vector to explain is empty.")

        if expected_features:
            missing = [f for f in expected_features if f not in feature_row.columns]
            if missing:
                raise ValueError(f"Feature vector is missing required columns: {missing}")
