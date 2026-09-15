"""Unit Tests for Explainability Agent and Model Immutability Regression (Phase 6)."""

import pytest
import numpy as np
import pandas as pd
import torch

from backend.app.explainability.agent import ExplainabilityAgent
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster
from backend.app.forecasting.prophet_model import ProphetForecaster
from backend.app.schemas.explainability import (
    ExplanationMethod,
    LocalExplanationRequest,
    GlobalExplanationRequest,
)


@pytest.fixture
def test_dataset():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    rng = np.random.default_rng(42)
    targets = 25.0 + 5.0 * np.sin(np.arange(80) * 2 * np.pi / 7) + rng.normal(0, 1, 80)
    df = pd.DataFrame({
        "date": dates,
        "target": targets,
        "entity_id": ["STORE_001"] * 80,
        "product_id": ["PROD_001"] * 80,
        "price": 3.0 + rng.uniform(-0.1, 0.1, 80),
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "lag_1": np.roll(targets, 1),
        "lag_7": np.roll(targets, 7),
    })
    return df


def test_agent_explain_local_lightgbm(test_dataset):
    agent = ExplainabilityAgent()
    req = LocalExplanationRequest(
        entity_id="STORE_001",
        product_id="PROD_001",
        model_name="lightgbm",
        method=ExplanationMethod.SHAP,
        background_samples=20,
    )

    history_df = test_dataset.iloc[:60]
    train_df = test_dataset.iloc[:45]

    res = agent.explain_local(req, history_df=history_df, train_df=train_df)

    assert res.explanation_id.startswith("expl_")
    assert res.model_name == "lightgbm"
    assert res.method == "shap"
    assert res.explanation_type == "local"
    assert len(res.features) > 0
    assert res.audit_trail.random_seed == 42
    assert res.fidelity.fidelity_score > 0.85


def test_agent_explain_global_lightgbm(test_dataset):
    agent = ExplainabilityAgent()
    req = GlobalExplanationRequest(
        entity_id="STORE_001",
        product_id="PROD_001",
        model_name="lightgbm",
        method=ExplanationMethod.SHAP,
        sample_size=20,
        top_k=5,
    )

    history_df = test_dataset.iloc[:60]
    train_df = test_dataset.iloc[:45]

    res = agent.explain_global(req, history_df=history_df, train_df=train_df)

    assert res.explanation_type == "global"
    assert len(res.global_importance) <= 5
    assert res.global_importance[0].rank == 1


def test_model_immutability_regression_lightgbm(test_dataset):
    """REGRESSION TEST: Verify Explainability Agent never modifies LightGBM state or trees."""
    agent = ExplainabilityAgent()
    history_df = test_dataset.iloc[:60]
    train_df = test_dataset.iloc[:45]

    model = LightGBMForecaster(random_seed=42)
    model.fit(history_df)

    # 1. Capture snapshot before explanation
    trees_before = model.model.booster_.dump_model()
    num_trees_before = model.model.booster_.num_trees()
    history_targets_before = list(model._historical_targets)
    features_before = list(model.feature_names)

    # 2. Run multiple explanations (SHAP and LIME)
    req_shap = LocalExplanationRequest(model_name="lightgbm", method="shap", background_samples=20)
    agent.explain_local(req_shap, model=model, history_df=history_df, train_df=train_df)

    req_lime = LocalExplanationRequest(model_name="lightgbm", method="lime", background_samples=20, num_lime_samples=150)
    agent.explain_local(req_lime, model=model, history_df=history_df, train_df=train_df)

    # 3. Verify state after explanations
    trees_after = model.model.booster_.dump_model()
    num_trees_after = model.model.booster_.num_trees()
    history_targets_after = list(model._historical_targets)
    features_after = list(model.feature_names)

    assert num_trees_before == num_trees_after
    assert trees_before == trees_after
    assert history_targets_before == history_targets_after
    assert features_before == features_after


def test_model_immutability_regression_lstm(test_dataset):
    """REGRESSION TEST: Verify Explainability Agent never modifies LSTM weights or scalers."""
    agent = ExplainabilityAgent()
    history_df = test_dataset.iloc[:60]
    train_df = test_dataset.iloc[:45]

    model = LSTMForecaster(random_seed=42, lookback=7, epochs=10)
    model.fit(history_df)

    # Capture state before
    scaler_mean_before = model.scaler_mean
    scaler_std_before = model.scaler_std
    weights_before = {name: param.clone() for name, param in model.net.named_parameters()}

    req = LocalExplanationRequest(model_name="lstm", method="shap", background_samples=15)
    agent.explain_local(req, model=model, history_df=history_df, train_df=train_df)

    # Verify state after
    assert model.scaler_mean == scaler_mean_before
    assert model.scaler_std == scaler_std_before
    for name, param in model.net.named_parameters():
        assert torch.equal(weights_before[name], param)


def test_agent_unsupported_method_error(test_dataset):
    agent = ExplainabilityAgent()
    history_df = test_dataset.iloc[:60]
    train_df = test_dataset.iloc[:45]

    req = LocalExplanationRequest(
        model_name="prophet",
        method=ExplanationMethod.LIME,  # LIME not supported for Prophet
    )

    with pytest.raises(ValueError, match="not supported for model 'prophet'"):
        agent.explain_local(req, history_df=history_df, train_df=train_df)
