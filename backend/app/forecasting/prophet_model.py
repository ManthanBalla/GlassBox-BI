"""Prophet Forecasting Adapter for GlassBox-BI.

Wraps Facebook/Meta Prophet in the canonical BaseForecastModel interface,
providing trend and additive/multiplicative seasonal modeling with native Bayesian
prediction intervals.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from backend.app.schemas.forecasting import ForecastPoint
from backend.app.forecasting.base import BaseForecastModel


class ProphetForecaster(BaseForecastModel):
    """Prophet adapter implementing the universal BaseForecastModel interface."""

    def __init__(
        self,
        model_name: str = "prophet",
        random_seed: int = 42,
        yearly_seasonality: Union[bool, str] = "auto",
        weekly_seasonality: Union[bool, str] = "auto",
        daily_seasonality: Union[bool, str] = False,
        interval_width: float = 0.80,
    ) -> None:
        super().__init__(model_name=model_name, random_seed=random_seed)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.interval_width = interval_width
        self.uncertainty_method = "prophet_bayesian_interval"
        self.model: Optional[Prophet] = None
        self._last_training_date: Optional[pd.Timestamp] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "target",
        date_col: str = "date",
        **kwargs: Any,
    ) -> "ProphetForecaster":
        """Fits Prophet model strictly on historical training observations."""
        start_t = time.perf_counter()
        self.target_col = target_col
        self.date_col = date_col
        self.training_rows = len(train_df)

        if self.training_rows < 5:
            raise ValueError(f"Prophet requires at least 5 observations, but got {self.training_rows}.")

        # 1. Transform canonical schema to Prophet standard: ds, y
        p_df = pd.DataFrame({
            "ds": pd.to_datetime(train_df[date_col]),
            "y": train_df[target_col].to_numpy(dtype=float),
        }).sort_values("ds").reset_index(drop=True)

        self.training_start = str(p_df["ds"].iloc[0].date())
        self.training_end = str(p_df["ds"].iloc[-1].date())
        self._last_training_date = p_df["ds"].iloc[-1]
        self.feature_names = ["ds", "y"]

        # 2. Instantiate and fit Prophet model
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=self.interval_width,
        )
        # Suppress cmdstanpy verbose logs where possible
        self.model.fit(p_df)

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
        """Generates future forecast points using Prophet's native trend and seasonality."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError(f"ProphetForecaster {self.model_name} is not fitted.")

        if horizon <= 0:
            raise ValueError(f"Forecast horizon must be positive, got {horizon}")

        # Construct future dataframe
        if future_dates and len(future_dates) > 0:
            future_ds = pd.to_datetime(future_dates)
            df_future = pd.DataFrame({"ds": future_ds})
        else:
            # Generate daily future dates by default
            df_future = self.model.make_future_dataframe(periods=horizon, freq="D", include_history=False)

        # Generate predictions
        fcst = self.model.predict(df_future)
        points: List[ForecastPoint] = []

        for _, row in fcst.iterrows():
            d_str = str(pd.to_datetime(row["ds"]).date())
            # Clamp negative demand values to 0.0
            pred = max(0.0, float(row["yhat"]))
            low = max(0.0, float(row.get("yhat_lower", pred)))
            high = max(pred, float(row.get("yhat_upper", pred)))

            points.append(
                ForecastPoint(
                    date=d_str,
                    prediction=round(pred, 4),
                    lower_bound=round(low, 4),
                    upper_bound=round(high, 4),
                )
            )

        return points[:horizon]

    def _get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "interval_width": self.interval_width,
        }

    def _get_software_versions(self) -> Dict[str, str]:
        import prophet
        base_v = super()._get_software_versions()
        base_v["prophet"] = prophet.__version__
        return base_v

    def save_model(self, path: Union[str, Path]) -> None:
        """Serializes Prophet model to JSON on disk."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted Prophet model.")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        model_json = model_to_json(self.model)
        payload = {
            "model_name": self.model_name,
            "model_type": "ProphetForecaster",
            "hyperparameters": self._get_hyperparameters(),
            "target_col": self.target_col,
            "date_col": self.date_col,
            "training_rows": self.training_rows,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "validation_residuals": self.validation_residuals.tolist(),
            "prophet_serialized": model_json,
        }

        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> "ProphetForecaster":
        """Deserializes Prophet model from JSON on disk."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path_obj}")

        with open(path_obj, "r", encoding="utf-8") as f:
            payload = json.load(f)

        hp = payload.get("hyperparameters", {})
        forecaster = cls(
            model_name=payload.get("model_name", "prophet"),
            yearly_seasonality=hp.get("yearly_seasonality", "auto"),
            weekly_seasonality=hp.get("weekly_seasonality", "auto"),
            daily_seasonality=hp.get("daily_seasonality", False),
            interval_width=hp.get("interval_width", 0.80),
        )
        forecaster.model = model_from_json(payload["prophet_serialized"])
        forecaster.target_col = payload.get("target_col", "target")
        forecaster.date_col = payload.get("date_col", "date")
        forecaster.training_rows = payload.get("training_rows", 0)
        forecaster.training_start = payload.get("training_start")
        forecaster.training_end = payload.get("training_end")
        forecaster.validation_residuals = np.array(payload.get("validation_residuals", []))
        forecaster.is_fitted = True
        return forecaster
