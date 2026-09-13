"""Base Abstract Interface for All GlassBox-BI Forecasting Models.

Enforces a strict common contract across statistical, machine learning,
and deep learning forecasters (Prophet, LightGBM, LSTM), enabling model-agnostic
competition, evaluation, and persistence without engine-specific branching.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from backend.app.schemas.forecasting import ForecastPoint, ModelMetadata


class BaseForecastModel(ABC):
    """Abstract base contract for time-series forecasting models."""

    def __init__(self, model_name: str, random_seed: int = 42) -> None:
        self.model_name = model_name
        self.random_seed = random_seed
        self.is_fitted = False
        self.target_col: str = "target"
        self.date_col: str = "date"
        self.training_duration: float = 0.0
        self.feature_names: List[str] = []
        self.training_rows: int = 0
        self.training_start: Optional[str] = None
        self.training_end: Optional[str] = None
        self.validation_residuals: np.ndarray = np.array([])
        self.lookback_window: Optional[int] = None
        self.uncertainty_method: str = "validation_residuals"

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "target",
        date_col: str = "date",
        **kwargs: Any,
    ) -> "BaseForecastModel":
        """Fits the model strictly on historical training observations.

        Zero Leakage Rule: Neither validation nor holdout test data may enter fit.
        """
        pass

    @abstractmethod
    def predict(
        self,
        horizon: int,
        future_dates: Optional[List[str]] = None,
        confidence_level: float = 0.80,
        **kwargs: Any,
    ) -> List[ForecastPoint]:
        """Generates future point predictions with prediction intervals."""
        pass

    @abstractmethod
    def save_model(self, path: Union[str, Path]) -> None:
        """Serializes model weights, architecture, and state to disk."""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path: Union[str, Path]) -> "BaseForecastModel":
        """Deserializes a fitted model from disk."""
        pass

    def evaluate_validation(
        self,
        val_df: pd.DataFrame,
        target_col: str = "target",
        date_col: str = "date",
    ) -> Dict[str, float]:
        """Evaluates model performance against validation partition.

        Computes MAE, RMSE, and MAPE, and preserves validation residuals
        to construct empirical prediction intervals for ML/DL models.
        """
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.model_name} must be fitted before evaluating validation.")

        val_dates = [str(pd.to_datetime(d).date()) for d in val_df[date_col]]
        val_horizon = len(val_df)
        if val_horizon == 0:
            return {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0}

        val_preds = self.predict(horizon=val_horizon, future_dates=val_dates)
        y_true = val_df[target_col].to_numpy(dtype=float)
        y_pred = np.array([p.prediction for p in val_preds], dtype=float)

        # Store residuals for prediction intervals
        self.validation_residuals = y_true - y_pred

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        
        # MAPE with division safety
        epsilon = 1e-5
        denom = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4),
        }

    def compute_residual_prediction_interval(
        self,
        point_prediction: float,
        confidence_level: float = 0.80,
    ) -> tuple[float, float]:
        """Calculates prediction intervals using empirical validation residuals.

        For models without native Bayesian interval estimation (e.g. LightGBM, LSTM),
        we use the empirical standard deviation or quantiles of validation residuals.
        """
        if len(self.validation_residuals) > 1:
            res_std = float(np.std(self.validation_residuals))
        else:
            res_std = max(1.0, abs(point_prediction) * 0.1)

        # Normal z-factor approximation for confidence level
        z_factors = {0.80: 1.282, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        # Closest match
        z = z_factors.get(confidence_level, 1.282)

        margin = z * res_std
        lower = max(0.0, point_prediction - margin)  # Retail demand non-negative bound
        upper = point_prediction + margin
        return round(lower, 4), round(upper, 4)

    def get_feature_requirements(self) -> List[str]:
        """Returns required feature column names for this model."""
        return self.feature_names

    def get_model_metadata(self) -> ModelMetadata:
        """Returns structured metadata documenting model configuration and training."""
        return ModelMetadata(
            model_name=self.model_name,
            model_type=self.__class__.__name__,
            hyperparameters=self._get_hyperparameters(),
            training_rows=self.training_rows,
            training_start=self.training_start,
            training_end=self.training_end,
            feature_names=self.feature_names,
            lookback_window=self.lookback_window,
            random_seed=self.random_seed,
            training_duration=round(self.training_duration, 4),
            uncertainty_method=self.uncertainty_method,
            software_versions=self._get_software_versions(),
        )

    def _get_hyperparameters(self) -> Dict[str, Any]:
        """Internal helper returning model hyperparameter dictionary."""
        return {}

    def _get_software_versions(self) -> Dict[str, str]:
        """Returns framework versions for reproducibility audit."""
        import numpy
        import pandas
        return {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
        }
