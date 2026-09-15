"""Unit Tests for Prophet Component-Based Explainer (Phase 6)."""

import pytest
import numpy as np
import pandas as pd

from backend.app.forecasting.prophet_model import ProphetForecaster
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.explainability.prophet_explainer import ProphetComponentExplainer
from backend.app.explainability.feature_adapter import ExplainabilityFeatureAdapter


@pytest.fixture
def prophet_data():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(42)
    targets = 25.0 + 3.0 * np.sin(np.arange(60) * 2 * np.pi / 7) + rng.normal(0, 0.5, 60)
    df = pd.DataFrame({
        "date": dates,
        "target": targets,
    })
    return df


def test_prophet_local_component_decomposition(prophet_data):
    model = ProphetForecaster(random_seed=42)
    model.fit(prophet_data.iloc[:45])

    explainer = ProphetComponentExplainer(random_seed=42)
    feat_row, pred, _ = ExplainabilityFeatureAdapter.extract_prediction_features(
        model=model,
        history_df=prophet_data.iloc[:45],
        step_index=0,
    )

    res = explainer.explain_local(model, feat_row, pred)

    assert "features" in res
    assert "fidelity" in res
    assert res["fidelity"].explanation_status == "DECOMPOSED"
    assert res["fidelity"].fidelity_score == 1.0

    # Components must include trend and seasonality
    feature_names = [f.feature for f in res["features"]]
    assert "trend" in feature_names
    assert "weekly_seasonality" in feature_names

    # Ensure no fabricated tabular features exist
    for f in feature_names:
        assert not f.startswith("lag_")
        assert not f.startswith("rolling_")


def test_prophet_global_component_importance(prophet_data):
    model = ProphetForecaster(random_seed=42)
    model.fit(prophet_data.iloc[:45])

    explainer = ProphetComponentExplainer(random_seed=42)
    ref_df = pd.DataFrame({"ds": pd.date_range("2024-02-15", periods=14, freq="D")})

    res = explainer.explain_global(model, ref_df)

    assert "global_importance" in res
    assert len(res["global_importance"]) >= 2
    comp_names = [gf.feature for gf in res["global_importance"]]
    assert "trend" in comp_names
    assert "weekly_seasonality" in comp_names


def test_prophet_explainer_rejects_other_models():
    lgb = LightGBMForecaster()
    explainer = ProphetComponentExplainer()
    with pytest.raises(ValueError, match="requires a ProphetForecaster"):
        explainer.explain_local(lgb, pd.DataFrame(), 10.0)
