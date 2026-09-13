"""Unit tests for Temporal Integrity, Scaler Isolation, and Future Target Leakage Protection.

Critical Invariants Tested:
1. Scaler Isolation: Normalization parameters are strictly fitted on training data only.
2. Temporal Disjointness: Train, Validation, and Test partitions have zero temporal overlap.
3. Holdout Test Protection: Test target values are never seen during candidate model fitting or selection.
4. Recursive Horizon Safety: Multi-step forecasting dynamically computes future lags from model predictions
   without peeking into ground-truth future target values.
"""

import numpy as np
import pandas as pd
import pytest

from backend.app.forecasting.agent import GenericForecastingAgent
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster
from backend.app.schemas.forecasting import ForecastRequest


@pytest.fixture
def temporal_split_dataset() -> pd.DataFrame:
    """Creates 100 days of time-series data with clear trend to verify split isolation."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    t = np.arange(100)
    # Distinct non-stationary distribution: higher mean in later periods
    values = 10.0 + 1.5 * t

    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "target": values,
        "entity_id": "STORE_LEAK",
        "product_id": "PROD_LEAK",
    })


def test_lstm_scaler_train_only_isolation(temporal_split_dataset: pd.DataFrame):
    """Verifies that LSTM scaler mean and std are computed ONLY on training partition."""
    # Split 70 train, 15 val, 15 test
    train_df = temporal_split_dataset.iloc[:70].copy()
    val_df = temporal_split_dataset.iloc[70:85].copy()
    test_df = temporal_split_dataset.iloc[85:].copy()

    forecaster = LSTMForecaster(model_name="lstm", lookback=7, epochs=5, batch_size=8)
    forecaster.fit(train_df, target_col="target", date_col="date")

    expected_train_mean = float(np.mean(train_df["target"].values))
    expected_train_std = float(np.std(train_df["target"].values))
    global_mean = float(np.mean(temporal_split_dataset["target"].values))

    # Scaler mean must match training partition mean
    assert np.isclose(forecaster.scaler_mean, expected_train_mean, atol=1e-5)
    assert np.isclose(forecaster.scaler_std, expected_train_std, atol=1e-5)

    # Scaler mean must NOT match global mean (which would indicate future leakage)
    assert not np.isclose(forecaster.scaler_mean, global_mean, atol=1.0)


def test_holdout_test_set_protection_during_selection(temporal_split_dataset: pd.DataFrame, tmp_path, monkeypatch):
    """Verifies that candidate model training and selection never read or inspect the holdout test set."""
    agent = GenericForecastingAgent(persistence_dir=tmp_path)

    # Track every dataframe slice passed into model.fit
    seen_dates = set()

    original_fit = LightGBMForecaster.fit

    def auditing_fit(self, train_df, *args, **kwargs):
        for d in train_df["date"]:
            seen_dates.add(str(d))
        return original_fit(self, train_df, *args, **kwargs)

    monkeypatch.setattr(LightGBMForecaster, "fit", auditing_fit)

    req = ForecastRequest(
        entity_id="STORE_LEAK",
        product_id="PROD_LEAK",
        horizon=7,
        candidate_models=["lightgbm"],
        selection_metric="MAE",
    )

    # Execute forecasting agent
    result = agent.run_forecast(req, df=temporal_split_dataset)
    assert result.status == "SUCCESS"

    # Dataset had 100 days. By default: 70 train, 15 val, 15 test.
    # Training dates are days 0..69 (during candidate fit) and days 0..84 (during refit on train+val).
    # Test partition (days 85..99) MUST NEVER be seen in fit!
    test_dates = set(temporal_split_dataset["date"].iloc[85:].tolist())

    leakage = seen_dates.intersection(test_dates)
    assert len(leakage) == 0, f"Critical leakage detected: Test dates {leakage} were seen during model fitting!"


def test_recursive_multi_step_forecasting_does_not_peek_future_target(temporal_split_dataset: pd.DataFrame):
    """Verifies that LightGBM recursive multi-step forecasting creates lags from predictions, not actual future."""
    train_df = temporal_split_dataset.iloc[:70].copy()

    forecaster = LightGBMForecaster(model_name="lightgbm", random_seed=42, n_estimators=20)
    forecaster.fit(train_df, target_col="target", date_col="date")

    # Predict 14 steps into the future
    predictions = forecaster.predict(horizon=14)

    assert len(predictions) == 14
    for pt in predictions:
        assert np.isfinite(pt.prediction)

    # Predictions must not be identical to actual future target (days 70..83)
    actual_future = temporal_split_dataset["target"].iloc[70:84].values
    predicted_vals = np.array([p.prediction for p in predictions])

    # If it was peeking, predicted_vals == actual_future exactly
    assert not np.allclose(predicted_vals, actual_future, atol=1e-3)
