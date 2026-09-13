"""LightGBM Forecasting Adapter for GlassBox-BI.

Implements gradient boosted decision tree forecasting using engineered calendar,
lag, and rolling features, with recursive multi-step future forecasting and
validation-residual-based prediction intervals.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from backend.app.schemas.forecasting import ForecastPoint
from backend.app.forecasting.base import BaseForecastModel


class LightGBMForecaster(BaseForecastModel):
    """LightGBM adapter implementing the universal BaseForecastModel interface."""

    def __init__(
        self,
        model_name: str = "lightgbm",
        random_seed: int = 42,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        num_leaves: int = 31,
    ) -> None:
        super().__init__(model_name=model_name, random_seed=random_seed)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.model: Optional[lgb.LGBMRegressor] = None
        self._historical_targets: List[float] = []
        self._historical_dates: List[pd.Timestamp] = []
        self._last_known_features: Dict[str, Any] = {}

    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Identifies available feature columns excluding target, date, and IDs."""
        exclude = {self.target_col, self.date_col, "entity_id", "product_id", "target_original", "target_outlier"}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c not in exclude]
        return features

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "target",
        date_col: str = "date",
        **kwargs: Any,
    ) -> "LightGBMForecaster":
        """Fits LightGBM regressor on historical training observations and engineered features."""
        start_t = time.perf_counter()
        self.target_col = target_col
        self.date_col = date_col
        self.training_rows = len(train_df)

        if self.training_rows < 10:
            raise ValueError(f"LightGBM requires at least 10 observations, got {self.training_rows}.")

        sorted_train = train_df.sort_values(by=date_col).reset_index(drop=True)
        self.training_start = str(pd.to_datetime(sorted_train[date_col].iloc[0]).date())
        self.training_end = str(pd.to_datetime(sorted_train[date_col].iloc[-1]).date())

        # Cache historical timeline for recursive multi-step feature computation
        self._historical_dates = list(pd.to_datetime(sorted_train[date_col]))
        self._historical_targets = list(sorted_train[target_col].astype(float))

        # Select numerical feature regressors
        self.feature_names = self._select_feature_columns(sorted_train)
        if not self.feature_names:
            # Fallback: create basic calendar components if no pre-engineered features exist
            dates = pd.to_datetime(sorted_train[date_col])
            sorted_train["day_of_week"] = dates.dt.dayofweek
            sorted_train["month"] = dates.dt.month
            sorted_train["lag_1"] = sorted_train[target_col].shift(1).fillna(sorted_train[target_col].iloc[0])
            self.feature_names = ["day_of_week", "month", "lag_1"]

        X = sorted_train[self.feature_names].fillna(0.0)
        y = sorted_train[target_col].to_numpy(dtype=float)

        # Cache last row feature values for recursive forecasting fallback
        self._last_known_features = sorted_train[self.feature_names].iloc[-1].to_dict()

        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            random_state=self.random_seed,
            verbose=-1,
        )
        self.model.fit(X, y)

        self.is_fitted = True
        self.training_duration = time.perf_counter() - start_t
        return self

    def predict(
        self,
        horizon: int,
        future_dates: Optional[List[str]] = None,
        confidence_level: float = 0.80,
        **kwargs: Any,
    ) -> List[ForecastPoint]:
        """Generates future predictions using recursive multi-step forecasting."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"LightGBMForecaster {self.model_name} is not fitted.")

        if horizon <= 0:
            raise ValueError(f"Forecast horizon must be positive, got {horizon}")

        # Determine future date sequence
        if future_dates and len(future_dates) > 0:
            resolved_dates = [pd.to_datetime(d) for d in future_dates[:horizon]]
        else:
            last_date = self._historical_dates[-1]
            resolved_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(horizon)]

        # Recursive Multi-Step Forecast:
        # History buffer holds historical targets + newly predicted targets
        extended_targets = list(self._historical_targets)
        forecast_points: List[ForecastPoint] = []

        for step_idx, step_date in enumerate(resolved_dates):
            # Construct feature vector for current step
            row_features = dict(self._last_known_features)

            # 1. Update Calendar features
            row_features["year"] = step_date.year
            row_features["month"] = step_date.month
            row_features["quarter"] = step_date.quarter
            row_features["day_of_week"] = step_date.dayofweek
            row_features["day_of_month"] = step_date.day
            row_features["day_of_year"] = step_date.dayofyear
            row_features["week_of_year"] = step_date.isocalendar().week
            row_features["is_weekend"] = 1 if step_date.dayofweek >= 5 else 0

            # 2. Update Lag features from extended_targets (strictly historical + past predictions)
            for lag in [1, 7, 14, 28]:
                lag_col = f"lag_{lag}"
                if lag_col in self.feature_names:
                    if len(extended_targets) >= lag:
                        row_features[lag_col] = extended_targets[-lag]
                    else:
                        row_features[lag_col] = extended_targets[0]

            # 3. Update Rolling features from extended_targets (shift-1: uses history before this step)
            for w in [7, 14, 28]:
                mean_col = f"rolling_mean_{w}"
                std_col = f"rolling_std_{w}"
                if mean_col in self.feature_names:
                    win_vals = extended_targets[-w:] if len(extended_targets) >= w else extended_targets
                    row_features[mean_col] = float(np.mean(win_vals))
                if std_col in self.feature_names:
                    win_vals = extended_targets[-w:] if len(extended_targets) >= w else extended_targets
                    row_features[std_col] = float(np.std(win_vals)) if len(win_vals) > 1 else 0.0

            # Build single-row DataFrame matching trained feature columns
            x_step = pd.DataFrame([{col: row_features.get(col, 0.0) for col in self.feature_names}])
            
            # Predict step
            raw_pred = float(self.model.predict(x_step)[0])
            pred_val = max(0.0, round(raw_pred, 4))  # Clamp retail demand non-negative

            # Append to recursive buffer for subsequent lag/rolling calculations
            extended_targets.append(pred_val)

            # Compute prediction intervals via validation residuals
            low, high = self.compute_residual_prediction_interval(pred_val, confidence_level=confidence_level)

            forecast_points.append(
                ForecastPoint(
                    date=str(step_date.date()),
                    prediction=pred_val,
                    lower_bound=low,
                    upper_bound=high,
                )
            )

        return forecast_points

    def _get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
        }

    def _get_software_versions(self) -> Dict[str, str]:
        base_v = super()._get_software_versions()
        base_v["lightgbm"] = lgb.__version__
        return base_v

    def save_model(self, path: Union[str, Path]) -> None:
        """Serializes LightGBM model and metadata to disk."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted LightGBM model.")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_name": self.model_name,
            "model_type": "LightGBMForecaster",
            "hyperparameters": self._get_hyperparameters(),
            "target_col": self.target_col,
            "date_col": self.date_col,
            "training_rows": self.training_rows,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "feature_names": self.feature_names,
            "validation_residuals": self.validation_residuals.tolist(),
            "historical_targets": self._historical_targets,
            "historical_dates": [str(d) for d in self._historical_dates],
            "last_known_features": self._last_known_features,
            "model_object": self.model,
        }
        joblib.dump(payload, path_obj)

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> "LightGBMForecaster":
        """Deserializes LightGBM model from disk."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path_obj}")

        payload = joblib.load(path_obj)
        hp = payload.get("hyperparameters", {})
        forecaster = cls(
            model_name=payload.get("model_name", "lightgbm"),
            n_estimators=hp.get("n_estimators", 100),
            learning_rate=hp.get("learning_rate", 0.05),
            max_depth=hp.get("max_depth", 5),
            num_leaves=hp.get("num_leaves", 31),
        )
        forecaster.model = payload["model_object"]
        forecaster.target_col = payload.get("target_col", "target")
        forecaster.date_col = payload.get("date_col", "date")
        forecaster.training_rows = payload.get("training_rows", 0)
        forecaster.training_start = payload.get("training_start")
        forecaster.training_end = payload.get("training_end")
        forecaster.feature_names = payload.get("feature_names", [])
        forecaster.validation_residuals = np.array(payload.get("validation_residuals", []))
        forecaster._historical_targets = payload.get("historical_targets", [])
        forecaster._historical_dates = [pd.to_datetime(d) for d in payload.get("historical_dates", [])]
        forecaster._last_known_features = payload.get("last_known_features", {})
        forecaster.is_fitted = True
        return forecaster
