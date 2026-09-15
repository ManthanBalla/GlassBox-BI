"""Unit Tests for LIME Explainer (Phase 6)."""

import pytest
import numpy as np
import pandas as pd

from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster
from backend.app.forecasting.prophet_model import ProphetForecaster
from backend.app.explainability.lime_explainer import LIMEExplainer
from backend.app.explainability.feature_adapter import ExplainabilityFeatureAdapter


@pytest.fixture
def synthetic_series():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(42)
    targets = 20.0 + 5.0 * np.sin(np.arange(60) * 2 * np.pi / 7) + rng.normal(0, 1, 60)
    df = pd.DataFrame({
        "date": dates,
        "target": targets,
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "lag_1": np.roll(targets, 1),
        "lag_7": np.roll(targets, 7),
        "price": 2.50 + rng.uniform(-0.2, 0.2, 60),
    })
    return df


def test_lightgbm_lime_local_explanation(synthetic_series):
    model = LightGBMForecaster(random_seed=42)
    model.fit(synthetic_series.iloc[:45])

    explainer = LIMEExplainer(random_seed=42, num_samples=200)
    bg = ExplainabilityFeatureAdapter.extract_background_sample(model, synthetic_series.iloc[:45], 20, 42)
    feat_row, pred, _ = ExplainabilityFeatureAdapter.extract_prediction_features(model, synthetic_series.iloc[:45], 0)

    res = explainer.explain_local(model, feat_row, pred, bg)

    assert "features" in res
    assert "fidelity" in res
    assert len(res["features"]) == len(model.feature_names)
    assert res["fidelity"].surrogate_r2 is not None


def test_lime_reproducible_seed(synthetic_series):
    model = LightGBMForecaster(random_seed=42)
    model.fit(synthetic_series.iloc[:45])

    expl1 = LIMEExplainer(random_seed=42, num_samples=200)
    expl2 = LIMEExplainer(random_seed=42, num_samples=200)

    bg = ExplainabilityFeatureAdapter.extract_background_sample(model, synthetic_series.iloc[:45], 20, 42)
    feat_row, pred, _ = ExplainabilityFeatureAdapter.extract_prediction_features(model, synthetic_series.iloc[:45], 0)

    res1 = expl1.explain_local(model, feat_row, pred, bg)
    res2 = expl2.explain_local(model, feat_row, pred, bg)

    # Identical seed produces identical feature ordering and weights
    for f1, f2 in zip(res1["features"], res2["features"]):
        assert f1.feature == f2.feature
        assert pytest.approx(f1.contribution, abs=1e-3) == f2.contribution


def test_lime_requires_background_data(synthetic_series):
    model = LightGBMForecaster(random_seed=42)
    model.fit(synthetic_series.iloc[:45])

    explainer = LIMEExplainer(random_seed=42)
    feat_row, pred, _ = ExplainabilityFeatureAdapter.extract_prediction_features(model, synthetic_series.iloc[:45], 0)

    with pytest.raises(ValueError, match="LIME requires a background"):
        explainer.explain_local(model, feat_row, pred, background_df=None)


def test_lstm_lime_explanation(synthetic_series):
    model = LSTMForecaster(random_seed=42, lookback=7, epochs=10)
    model.fit(synthetic_series.iloc[:45])

    explainer = LIMEExplainer(random_seed=42, num_samples=150)
    bg = ExplainabilityFeatureAdapter.extract_background_sample(model, synthetic_series.iloc[:45], 15, 42)
    feat_row, pred, _ = ExplainabilityFeatureAdapter.extract_prediction_features(model, synthetic_series.iloc[:45], 0)

    res = explainer.explain_local(model, feat_row, pred, bg)
    assert len(res["features"]) == 7
    assert {f.feature for f in res["features"]} == {f"lag_{i+1}" for i in range(7)}


def test_lime_rejects_prophet(synthetic_series):
    p_model = ProphetForecaster(random_seed=42)
    p_model.fit(synthetic_series.iloc[:30])

    explainer = LIMEExplainer(random_seed=42)
    feat_row = pd.DataFrame({"ds": [pd.to_datetime("2024-02-01")]})

    with pytest.raises(ValueError, match="Prophet does not consume tabular features"):
        explainer.explain_local(p_model, feat_row, 20.0)
