"""Unit Tests for SHAP Explainer (Phase 6)."""

import pytest
import numpy as np
import pandas as pd

from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster
from backend.app.forecasting.prophet_model import ProphetForecaster
from backend.app.explainability.shap_explainer import SHAPExplainer
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


def test_lightgbm_shap_local_explanation(synthetic_series):
    model = LightGBMForecaster(random_seed=42)
    model.fit(synthetic_series.iloc[:45])

    explainer = SHAPExplainer(random_seed=42)
    bg_sample = ExplainabilityFeatureAdapter.extract_background_sample(
        model=model,
        train_df=synthetic_series.iloc[:45],
        sample_size=20,
        random_seed=42,
    )

    feat_row, pred, _ = ExplainabilityFeatureAdapter.extract_prediction_features(
        model=model,
        history_df=synthetic_series.iloc[:45],
        step_index=0,
    )

    result = explainer.explain_local(
        model=model,
        feature_row=feat_row,
        prediction=pred,
        background_df=bg_sample,
    )

    assert "features" in result
    assert "fidelity" in result
    assert len(result["features"]) == len(model.feature_names)
    assert result["fidelity"].fidelity_score > 0.90

    # Test additive reconstruction property: prediction ≈ base_value + sum(shap_values)
    sum_contrib = sum(f.contribution for f in result["features"])
    reconstructed = result["base_value"] + sum_contrib
    assert abs(pred - reconstructed) < 0.1


def test_lightgbm_shap_global_explanation(synthetic_series):
    model = LightGBMForecaster(random_seed=42)
    model.fit(synthetic_series.iloc[:45])

    explainer = SHAPExplainer(random_seed=42)
    ref_sample = ExplainabilityFeatureAdapter.extract_background_sample(
        model=model,
        train_df=synthetic_series.iloc[:45],
        sample_size=25,
        random_seed=42,
    )

    result = explainer.explain_global(
        model=model,
        reference_df=ref_sample,
        top_k=5,
    )

    assert "global_importance" in result
    assert len(result["global_importance"]) <= 5
    assert result["global_importance"][0].rank == 1
    assert result["global_importance"][0].importance_score >= result["global_importance"][1].importance_score


def test_lstm_shap_sequence_explanation(synthetic_series):
    model = LSTMForecaster(random_seed=42, lookback=7, epochs=10)
    model.fit(synthetic_series.iloc[:45])

    explainer = SHAPExplainer(random_seed=42)
    bg_sample = ExplainabilityFeatureAdapter.extract_background_sample(
        model=model,
        train_df=synthetic_series.iloc[:45],
        sample_size=15,
        random_seed=42,
    )

    feat_row, pred, _ = ExplainabilityFeatureAdapter.extract_prediction_features(
        model=model,
        history_df=synthetic_series.iloc[:45],
        step_index=0,
    )

    result = explainer.explain_local(
        model=model,
        feature_row=feat_row,
        prediction=pred,
        background_df=bg_sample,
    )

    assert "features" in result
    # Features must be lag_1 to lag_7 (the historical target sequence)
    feature_names = [f.feature for f in result["features"]]
    assert set(feature_names) == {f"lag_{i+1}" for i in range(7)}
    assert result["fidelity"].fidelity_score > 0.85


def test_shap_explainer_rejects_prophet(synthetic_series):
    p_model = ProphetForecaster(random_seed=42)
    p_model.fit(synthetic_series.iloc[:30])

    explainer = SHAPExplainer(random_seed=42)
    feat_row = pd.DataFrame({"ds": [pd.to_datetime("2024-02-01")]})

    with pytest.raises(ValueError, match="Prophet does not consume tabular features"):
        explainer.explain_local(p_model, feat_row, 20.0)


def test_shap_determinism(synthetic_series):
    model = LightGBMForecaster(random_seed=42)
    model.fit(synthetic_series.iloc[:45])

    expl1 = SHAPExplainer(random_seed=42)
    expl2 = SHAPExplainer(random_seed=42)

    bg = ExplainabilityFeatureAdapter.extract_background_sample(model, synthetic_series.iloc[:45], 20, 42)
    feat_row, pred, _ = ExplainabilityFeatureAdapter.extract_prediction_features(model, synthetic_series.iloc[:45], 0)

    res1 = expl1.explain_local(model, feat_row, pred, bg)
    res2 = expl2.explain_local(model, feat_row, pred, bg)

    for f1, f2 in zip(res1["features"], res2["features"]):
        assert f1.feature == f2.feature
        assert pytest.approx(f1.contribution, abs=1e-4) == f2.contribution
