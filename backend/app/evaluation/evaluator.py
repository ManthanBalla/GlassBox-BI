"""Model-Agnostic Test-Set Evaluator for GlassBox-BI (Phase 5).

Provides FormalForecastEvaluator to assess any BaseForecastModel against
the protected holdout test partition without lookahead contamination or target leakage.
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from backend.app.forecasting.base import BaseForecastModel
from backend.app.evaluation.metrics import compute_evaluation_metrics
from backend.app.schemas.evaluation import (
    EvaluationPoint,
    ModelTestEvaluation,
)


class FormalForecastEvaluator:
    """Evaluates time-series forecasting models strictly on holdout test partitions."""

    def __init__(self, zero_threshold: float = 1e-7) -> None:
        self.zero_threshold = zero_threshold

    def evaluate_model(
        self,
        model: BaseForecastModel,
        test_df: pd.DataFrame,
        horizon: int,
        target_col: str = "target",
        date_col: str = "date",
        confidence_level: float = 0.80,
    ) -> ModelTestEvaluation:
        """Evaluates a fitted forecasting model against a holdout test partition.

        Temporal Safety Invariants:
        1. Horizon Validation: Fails if requested horizon exceeds test observations.
        2. Model Isolation: Model must already be fitted on pre-test history; no fitting occurs here.
        3. No Future Target Peeking: Model.predict receives only calendar dates/horizon, never test targets.

        Args:
            model: Fitted BaseForecastModel (Prophet, LightGBM, LSTM, or future model).
            test_df: Untouched holdout test partition.
            horizon: Number of time steps to evaluate.
            target_col: Name of target column.
            date_col: Name of date column.
            confidence_level: Confidence level for prediction intervals.

        Returns:
            ModelTestEvaluation contract with metrics and actual vs predicted points.
        """
        start_time = time.perf_counter()

        if not model.is_fitted:
            raise RuntimeError(f"Cannot evaluate unfitted model: {model.model_name}")

        if horizon <= 0:
            raise ValueError(f"Evaluation horizon must be positive, got {horizon}")

        total_test_rows = len(test_df)
        if total_test_rows < horizon:
            raise ValueError(
                f"Requested evaluation horizon {horizon} exceeds available holdout test observations ({total_test_rows}). "
                f"Evaluation would require unobserved future data."
            )

        if target_col not in test_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in test dataframe.")
        if date_col not in test_df.columns:
            raise ValueError(f"Date column '{date_col}' not found in test dataframe.")

        # Sort test dataframe chronologically and take exact horizon slice
        sorted_test = test_df.sort_values(by=date_col).reset_index(drop=True)
        eval_slice = sorted_test.iloc[:horizon]

        test_dates = [str(pd.to_datetime(d).date()) for d in eval_slice[date_col]]
        test_start_date = test_dates[0]
        test_end_date = test_dates[-1]

        y_true = eval_slice[target_col].to_numpy(dtype=float)

        try:
            # Generate predictions for the test dates
            forecast_points = model.predict(
                horizon=horizon,
                future_dates=test_dates,
                confidence_level=confidence_level,
            )

            if len(forecast_points) != horizon:
                raise RuntimeError(
                    f"Model {model.model_name} generated {len(forecast_points)} predictions, expected {horizon}."
                )

            y_pred = np.array([pt.prediction for pt in forecast_points], dtype=float)

            # Compute formal metrics: MAE, RMSE, MAPE
            metrics_result = compute_evaluation_metrics(
                y_true=y_true,
                y_pred=y_pred,
                zero_threshold=self.zero_threshold,
            )

            # Build point-by-point actual vs prediction comparison
            comparison_points: List[EvaluationPoint] = []
            for i in range(horizon):
                act_val = float(y_true[i])
                pred_val = float(y_pred[i])
                pt = forecast_points[i]
                abs_err = round(abs(act_val - pred_val), 4)

                pct_err: Optional[float] = None
                if abs(act_val) > self.zero_threshold:
                    pct_err = round((abs(act_val - pred_val) / abs(act_val)) * 100.0, 4)

                comparison_points.append(
                    EvaluationPoint(
                        date=test_dates[i],
                        actual=round(act_val, 4),
                        prediction=round(pred_val, 4),
                        lower_bound=pt.lower_bound,
                        upper_bound=pt.upper_bound,
                        absolute_error=abs_err,
                        percentage_error=pct_err,
                    )
                )

            duration = time.perf_counter() - start_time
            meta = model.get_model_metadata()

            return ModelTestEvaluation(
                model_name=model.model_name,
                model_type=meta.model_type,
                status="SUCCESS",
                metrics=metrics_result,
                mae=metrics_result.mae,
                rmse=metrics_result.rmse,
                mape=metrics_result.mape,
                test_start_date=test_start_date,
                test_end_date=test_end_date,
                horizon=horizon,
                predictions=comparison_points,
                evaluation_duration_seconds=round(duration, 4),
                model_metadata=meta,
            )

        except Exception as exc:
            duration = time.perf_counter() - start_time
            return ModelTestEvaluation(
                model_name=model.model_name,
                model_type=getattr(model, "__class__", type(model)).__name__,
                status="FAILED",
                error_message=str(exc),
                test_start_date=test_start_date,
                test_end_date=test_end_date,
                horizon=horizon,
                evaluation_duration_seconds=round(duration, 4),
            )
