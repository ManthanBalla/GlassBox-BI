"""Mathematical Metric Definitions and Robust Zero-Target Calculations for Phase 5.

Implements formal forecasting accuracy metrics:
- Mean Absolute Error (MAE):
      MAE = (1/n) * sum(|y_i - y_hat_i|)
- Root Mean Squared Error (RMSE):
      RMSE = sqrt((1/n) * sum((y_i - y_hat_i)^2))
- Mean Absolute Percentage Error (MAPE):
      MAPE = (100/n_valid) * sum_{y_i != 0} (|y_i - y_hat_i| / |y_i|)

Zero-Target Strategy:
Division by zero is mathematically undefined. We implement a defensible,
transparent strategy: observations where actual y_i == 0 are excluded from
the MAPE calculation, and the exact count of excluded zero observations is
explicitly tracked and reported in zero_target_count. If all actual targets
are zero, MAPE evaluates to None with an explanatory audit record.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from backend.app.schemas.evaluation import EvaluationMetricResult


def calculate_mae(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
) -> float:
    """Calculates Mean Absolute Error (MAE).

    MAE = (1/n) * sum_{i=1}^n |y_i - y_hat_i|

    Args:
        y_true: Ground-truth target observations.
        y_pred: Model predicted values.

    Returns:
        Deterministic float MAE rounded to 4 decimal places.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) != len(y_p):
        raise ValueError(f"Array length mismatch: y_true has {len(y_t)}, y_pred has {len(y_p)}.")
    if len(y_t) == 0:
        raise ValueError("Cannot calculate MAE on empty arrays.")

    mae = float(np.mean(np.abs(y_t - y_p)))
    return round(mae, 4)


def calculate_rmse(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
) -> float:
    """Calculates Root Mean Squared Error (RMSE).

    RMSE = sqrt((1/n) * sum_{i=1}^n (y_i - y_hat_i)^2)

    Args:
        y_true: Ground-truth target observations.
        y_pred: Model predicted values.

    Returns:
        Deterministic float RMSE rounded to 4 decimal places.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) != len(y_p):
        raise ValueError(f"Array length mismatch: y_true has {len(y_t)}, y_pred has {len(y_p)}.")
    if len(y_t) == 0:
        raise ValueError("Cannot calculate RMSE on empty arrays.")

    mse = float(np.mean((y_t - y_p) ** 2))
    rmse = float(np.sqrt(mse))
    return round(rmse, 4)


def calculate_mape(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    zero_threshold: float = 1e-7,
) -> Tuple[Optional[float], int]:
    """Calculates Mean Absolute Percentage Error (MAPE) with zero-target protection.

    MAPE = (100 / n_valid) * sum_{|y_i| > threshold} (|y_i - y_hat_i| / |y_i|)

    Zero-Target Handling Policy:
    Observations where |y_true| <= zero_threshold are excluded from the summation
    to prevent division by zero or infinite distortion. The excluded count is
    returned alongside the metric for complete audit transparency.

    Args:
        y_true: Ground-truth target observations.
        y_pred: Model predicted values.
        zero_threshold: Threshold below which an actual value is treated as zero.

    Returns:
        Tuple of (mape_percentage, zero_target_count).
        If all observations are zero, mape_percentage is None.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) != len(y_p):
        raise ValueError(f"Array length mismatch: y_true has {len(y_t)}, y_pred has {len(y_p)}.")
    if len(y_t) == 0:
        raise ValueError("Cannot calculate MAPE on empty arrays.")

    non_zero_mask = np.abs(y_t) > zero_threshold
    zero_count = int(np.sum(~non_zero_mask))
    valid_count = int(np.sum(non_zero_mask))

    if valid_count == 0:
        # All observations are zero: cannot compute meaningful percentage
        return None, zero_count

    percentage_errors = np.abs((y_t[non_zero_mask] - y_p[non_zero_mask]) / y_t[non_zero_mask]) * 100.0
    mape = float(np.mean(percentage_errors))
    return round(mape, 4), zero_count


def compute_evaluation_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    zero_threshold: float = 1e-7,
) -> EvaluationMetricResult:
    """Computes complete suite of formal evaluation metrics (MAE, RMSE, MAPE).

    Args:
        y_true: Ground-truth target observations.
        y_pred: Model predicted values.
        zero_threshold: Threshold for zero-target exclusion in MAPE.

    Returns:
        Structured EvaluationMetricResult contract.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    mae = calculate_mae(y_t, y_p)
    rmse = calculate_rmse(y_t, y_p)
    mape, zero_count = calculate_mape(y_t, y_p, zero_threshold=zero_threshold)

    total_obs = len(y_t)
    eval_obs = total_obs - zero_count

    return EvaluationMetricResult(
        mae=mae,
        rmse=rmse,
        mape=mape,
        total_observations=total_obs,
        evaluated_observations=eval_obs,
        zero_target_count=zero_count,
        zero_handling_strategy="exclude_zeros_from_mape",
        metric_version="1.0.0",
    )
