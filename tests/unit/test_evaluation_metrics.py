"""Unit tests for Phase 5 Quantitative Evaluation Metrics.

Validates:
1. Exact mathematical accuracy of MAE, RMSE, and MAPE against hand-computed vectors.
2. Defensible, transparent handling of zero actual targets in MAPE without division by zero.
3. Deterministic repeatability of metric evaluations.
4. Input validation (length mismatches, empty arrays).
"""

import numpy as np
import pytest

from backend.app.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    compute_evaluation_metrics,
)
from backend.app.schemas.evaluation import EvaluationMetricResult


def test_mae_exact_calculation():
    """Verifies MAE matches exact hand-calculated formula: (1/n) * sum(|y - y_hat|)."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 37.0])
    # Absolute errors: [2, 2, 3, 3] -> sum = 10 -> mean = 2.5
    mae = calculate_mae(y_true, y_pred)
    assert np.isclose(mae, 2.5, atol=1e-4)


def test_rmse_exact_calculation():
    """Verifies RMSE matches exact formula: sqrt((1/n) * sum((y - y_hat)^2))."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 37.0])
    # Squared errors: [4, 4, 9, 9] -> sum = 26 -> mean = 6.5 -> sqrt(6.5) = 2.549509...
    rmse = calculate_rmse(y_true, y_pred)
    assert np.isclose(rmse, 2.5495, atol=1e-4)


def test_mape_exact_calculation():
    """Verifies MAPE matches exact formula: (100/n) * sum(|y - y_hat| / |y|)."""
    y_true = np.array([100.0, 200.0, 50.0])
    y_pred = np.array([110.0, 180.0, 45.0])
    # Percentage errors: [10/100 = 10%, 20/200 = 10%, 5/50 = 10%] -> mean = 10.0%
    mape, zero_count = calculate_mape(y_true, y_pred)
    assert mape is not None
    assert np.isclose(mape, 10.0, atol=1e-4)
    assert zero_count == 0


def test_mape_zero_target_handling():
    """Verifies defensible strategy: zeros are excluded from MAPE and counted transparently."""
    y_true = np.array([0.0, 100.0, 200.0, 0.0])
    y_pred = np.array([5.0, 110.0, 190.0, 2.0])

    mape, zero_count = calculate_mape(y_true, y_pred)
    # Zeros at indices 0 and 3 must be excluded: zero_count = 2
    assert zero_count == 2
    # Non-zero targets: [100, 200] with errors [10, 10] -> 10% and 5% -> mean = 7.5%
    assert mape is not None
    assert np.isclose(mape, 7.5, atol=1e-4)


def test_mape_all_zero_targets():
    """Verifies that if all actual targets are zero, MAPE evaluates to None without crashing."""
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    mape, zero_count = calculate_mape(y_true, y_pred)
    assert mape is None
    assert zero_count == 3


def test_compute_evaluation_metrics_suite():
    """Verifies comprehensive compute_evaluation_metrics returning structured EvaluationMetricResult."""
    y_true = np.array([10.0, 20.0, 0.0, 40.0])
    y_pred = np.array([12.0, 18.0, 1.0, 44.0])

    res = compute_evaluation_metrics(y_true, y_pred)
    assert isinstance(res, EvaluationMetricResult)
    assert res.total_observations == 4
    assert res.evaluated_observations == 3
    assert res.zero_target_count == 1
    assert res.zero_handling_strategy == "exclude_zeros_from_mape"
    assert res.mae > 0.0
    assert res.rmse > 0.0
    assert res.mape is not None and res.mape > 0.0


def test_metric_determinism():
    """Verifies identical results on multiple consecutive evaluations."""
    rng = np.random.RandomState(42)
    y_true = rng.uniform(10.0, 100.0, size=50)
    y_pred = y_true + rng.normal(0, 5.0, size=50)

    res1 = compute_evaluation_metrics(y_true, y_pred)
    res2 = compute_evaluation_metrics(y_true, y_pred)

    assert res1.mae == res2.mae
    assert res1.rmse == res2.rmse
    assert res1.mape == res2.mape
    assert res1.zero_target_count == res2.zero_target_count


def test_metric_validation_errors():
    """Verifies appropriate exceptions on length mismatch and empty arrays."""
    with pytest.raises(ValueError, match="mismatch"):
        calculate_mae([1.0, 2.0], [1.0])

    with pytest.raises(ValueError, match="empty"):
        calculate_rmse([], [])

    with pytest.raises(ValueError, match="mismatch"):
        calculate_mape([1.0], [1.0, 2.0])
