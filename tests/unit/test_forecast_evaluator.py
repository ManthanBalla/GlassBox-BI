"""Unit tests for FormalForecastEvaluator (Phase 5).

Validates:
1. Single model evaluation on holdout test partition.
2. Exact calculation of point-by-point ground truth comparison.
3. Horizon validation: rejecting horizon longer than available test set.
4. Unfitted model error handling.
5. Missing target/date column error handling.
6. Graceful failure capture if model prediction errors out.
"""

import numpy as np
import pandas as pd
import pytest

from backend.app.evaluation.evaluator import FormalForecastEvaluator
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.schemas.evaluation import ModelTestEvaluation


@pytest.fixture
def train_and_test_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates contiguous train and holdout test partitions."""
    train_dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    test_dates = pd.date_range(start="2024-03-01", periods=14, freq="D")

    train_df = pd.DataFrame({
        "date": train_dates.strftime("%Y-%m-%d"),
        "target": 50.0 + np.arange(60) * 0.5,
        "entity_id": "STORE_TEST",
        "product_id": "PROD_TEST",
    })

    test_df = pd.DataFrame({
        "date": test_dates.strftime("%Y-%m-%d"),
        "target": 80.0 + np.arange(14) * 0.5,
        "entity_id": "STORE_TEST",
        "product_id": "PROD_TEST",
    })

    return train_df, test_df


def test_evaluator_successful_model_evaluation(train_and_test_series):
    """Verifies that a fitted model is evaluated accurately on holdout test data."""
    train_df, test_df = train_and_test_series
    evaluator = FormalForecastEvaluator()

    model = LightGBMForecaster(model_name="lightgbm", random_seed=42, n_estimators=20)
    model.fit(train_df, target_col="target", date_col="date")

    eval_result = evaluator.evaluate_model(
        model=model,
        test_df=test_df,
        horizon=7,
        target_col="target",
        date_col="date",
    )

    assert isinstance(eval_result, ModelTestEvaluation)
    assert eval_result.status == "SUCCESS"
    assert eval_result.model_name == "lightgbm"
    assert eval_result.horizon == 7
    assert eval_result.mae is not None and eval_result.mae >= 0.0
    assert eval_result.rmse is not None and eval_result.rmse >= 0.0
    assert eval_result.mape is not None and eval_result.mape >= 0.0
    assert len(eval_result.predictions) == 7
    assert eval_result.test_start_date == "2024-03-01"
    assert eval_result.test_end_date == "2024-03-07"

    # Verify point-by-point comparison details
    for pt in eval_result.predictions:
        assert pt.actual > 0.0
        assert pt.prediction >= 0.0
        assert np.isclose(pt.absolute_error, abs(pt.actual - pt.prediction), atol=1e-4)


def test_evaluator_horizon_validation_rejection(train_and_test_series):
    """Verifies clear ValueError when requested horizon exceeds available holdout test observations."""
    train_df, test_df = train_and_test_series
    evaluator = FormalForecastEvaluator()

    model = LightGBMForecaster(model_name="lightgbm", random_seed=42, n_estimators=10)
    model.fit(train_df)

    # test_df has 14 rows; request 28 rows -> must raise ValueError
    with pytest.raises(ValueError, match="exceeds available holdout test observations"):
        evaluator.evaluate_model(model=model, test_df=test_df, horizon=28)


def test_evaluator_unfitted_model_rejection(train_and_test_series):
    """Verifies that attempting to evaluate an unfitted model raises RuntimeError."""
    _, test_df = train_and_test_series
    evaluator = FormalForecastEvaluator()
    unfitted_model = LightGBMForecaster(model_name="lightgbm")

    with pytest.raises(RuntimeError, match="unfitted"):
        evaluator.evaluate_model(model=unfitted_model, test_df=test_df, horizon=7)


def test_evaluator_missing_columns(train_and_test_series):
    """Verifies that missing date or target columns raise clear ValueError."""
    train_df, test_df = train_and_test_series
    evaluator = FormalForecastEvaluator()

    model = LightGBMForecaster(model_name="lightgbm", n_estimators=10)
    model.fit(train_df)

    bad_test = test_df.drop(columns=["target"])
    with pytest.raises(ValueError, match="Target column"):
        evaluator.evaluate_model(model=model, test_df=bad_test, horizon=7)


def test_evaluator_graceful_model_failure_capture(train_and_test_series, monkeypatch):
    """Verifies that if model.predict fails, evaluator captures FAILED status without crashing."""
    train_df, test_df = train_and_test_series
    evaluator = FormalForecastEvaluator()

    model = LightGBMForecaster(model_name="lightgbm", n_estimators=10)
    model.fit(train_df)

    def crashing_predict(*args, **kwargs):
        raise RuntimeError("Simulated test prediction crash")

    monkeypatch.setattr(model, "predict", crashing_predict)

    result = evaluator.evaluate_model(model=model, test_df=test_df, horizon=7)
    assert result.status == "FAILED"
    assert "Simulated test prediction crash" in (result.error_message or "")
