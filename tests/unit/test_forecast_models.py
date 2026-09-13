"""Unit tests for Forecasting Model Adapters (Prophet, LightGBM, LSTM).

Validates common BaseForecastModel interface, model training, horizon prediction,
prediction interval ordering, metadata recording, and serialization.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.app.forecasting.base import BaseForecastModel
from backend.app.forecasting.prophet_model import ProphetForecaster
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster
from backend.app.schemas.forecasting import ForecastPoint


@pytest.fixture
def synthetic_series_data() -> pd.DataFrame:
    """Generates a clean synthetic single-series DataFrame with trend and seasonality."""
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    np.random.seed(42)
    # Target with linear trend + weekly cycle + small noise
    t = np.arange(60)
    target = 50.0 + 0.5 * t + 10.0 * np.sin(2 * np.pi * t / 7.0) + np.random.normal(0, 1.0, 60)
    target = np.clip(target, 5.0, 200.0)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "target": target,
        "entity_id": "STORE_001",
        "product_id": "PROD_001",
    })
    return df


def test_base_forecast_model_contract():
    """Verifies that BaseForecastModel cannot be instantiated directly without abstract methods."""
    with pytest.raises(TypeError):
        BaseForecastModel(model_name="abstract_test")  # type: ignore


def test_prophet_forecaster_lifecycle(synthetic_series_data: pd.DataFrame):
    """Tests Prophet forecaster fitting, future forecasting, uncertainty bounds, and metadata."""
    forecaster = ProphetForecaster(model_name="prophet", random_seed=42)
    assert forecaster.model_name == "prophet"
    assert not forecaster.is_fitted

    forecaster.fit(synthetic_series_data, target_col="target", date_col="date")
    assert forecaster.is_fitted
    assert forecaster.training_rows == len(synthetic_series_data)

    horizon = 7
    future_dates = [f"2024-03-{i:02d}" for i in range(2, 9)]
    predictions = forecaster.predict(horizon=horizon, future_dates=future_dates, confidence_level=0.80)

    assert len(predictions) == horizon
    for pt in predictions:
        assert isinstance(pt, ForecastPoint)
        assert np.isfinite(pt.prediction)
        assert pt.prediction >= 0.0
        if pt.lower_bound is not None and pt.upper_bound is not None:
            # Interval ordering invariant: lower <= prediction <= upper
            assert pt.lower_bound <= pt.prediction + 1e-5
            assert pt.prediction <= pt.upper_bound + 1e-5

    meta = forecaster.get_model_metadata()
    assert meta.model_name == "prophet"
    assert meta.training_rows == 60
    assert meta.uncertainty_method == "prophet_bayesian_interval"

    # Persistence verification
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "prophet_model.json"
        forecaster.save_model(save_path)
        assert save_path.exists()

        loaded = ProphetForecaster.load_model(save_path)
        assert loaded.is_fitted
        assert loaded.model_name == "prophet"


def test_lightgbm_forecaster_lifecycle(synthetic_series_data: pd.DataFrame):
    """Tests LightGBM forecaster fitting, recursive multi-step forecasting, uncertainty bounds, and save/load."""
    forecaster = LightGBMForecaster(model_name="lightgbm", random_seed=42, n_estimators=20)
    assert not forecaster.is_fitted

    forecaster.fit(synthetic_series_data, target_col="target", date_col="date")
    assert forecaster.is_fitted
    assert len(forecaster.feature_names) > 0

    horizon = 10
    predictions = forecaster.predict(horizon=horizon, confidence_level=0.80)
    assert len(predictions) == horizon

    for pt in predictions:
        assert np.isfinite(pt.prediction)
        assert pt.prediction >= 0.0
        if pt.lower_bound is not None and pt.upper_bound is not None:
            # Interval ordering invariant
            assert pt.lower_bound <= pt.prediction + 1e-5
            assert pt.prediction <= pt.upper_bound + 1e-5

    meta = forecaster.get_model_metadata()
    assert meta.model_name == "lightgbm"
    assert meta.uncertainty_method == "validation_residuals"

    # Persistence verification
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "lightgbm_model.joblib"
        forecaster.save_model(save_path)
        assert save_path.exists()

        loaded = LightGBMForecaster.load_model(save_path)
        assert loaded.is_fitted
        assert loaded.model_name == "lightgbm"


def test_lstm_forecaster_lifecycle(synthetic_series_data: pd.DataFrame):
    """Tests PyTorch LSTM forecaster fitting, scaling isolation, sequence forecasting, and persistence."""
    forecaster = LSTMForecaster(
        model_name="lstm",
        random_seed=42,
        lookback=7,
        hidden_dim=16,
        epochs=10,
        batch_size=8,
    )
    assert not forecaster.is_fitted

    forecaster.fit(synthetic_series_data, target_col="target", date_col="date")
    assert forecaster.is_fitted
    assert forecaster.scaler_mean is not None
    assert forecaster.scaler_std is not None

    horizon = 5
    predictions = forecaster.predict(horizon=horizon, confidence_level=0.80)
    assert len(predictions) == horizon

    for pt in predictions:
        assert np.isfinite(pt.prediction)
        assert pt.prediction >= 0.0
        if pt.lower_bound is not None and pt.upper_bound is not None:
            assert pt.lower_bound <= pt.prediction + 1e-5
            assert pt.prediction <= pt.upper_bound + 1e-5

    meta = forecaster.get_model_metadata()
    assert meta.model_name == "lstm"
    assert meta.lookback_window == 7
    assert meta.uncertainty_method == "validation_residuals"

    # Persistence verification
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "lstm_model.pt"
        forecaster.save_model(save_path)
        assert save_path.exists()

        loaded = LSTMForecaster.load_model(save_path)
        assert loaded.is_fitted
        assert loaded.model_name == "lstm"
        assert loaded.lookback == 7
