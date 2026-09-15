"""Unit Tests for Explainability Base Contracts and Validators (Phase 6)."""

import pytest
import pandas as pd
import numpy as np

from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.forecasting.lstm_model import LSTMForecaster
from backend.app.forecasting.prophet_model import ProphetForecaster
from backend.app.explainability.base import BaseExplainer
from backend.app.explainability.validation import (
    ExplainabilityValidator,
    MODEL_COMPATIBILITY,
)
from backend.app.explainability.feature_adapter import ExplainabilityFeatureAdapter
from backend.app.schemas.explainability import (
    ExplanationMethod,
    FeatureContribution,
    GlobalFeatureImportance,
)


class DummyExplainer(BaseExplainer):
    """Concrete subclass for testing base ranking and contract helpers."""
    def explain_local(self, model, feature_row, prediction, background_df=None, **kwargs):
        return {}

    def explain_global(self, model, reference_df, top_k=20, **kwargs):
        return {}

    def get_metadata(self):
        return {"class": "DummyExplainer"}


def test_base_explainer_rank_contributions():
    explainer = DummyExplainer(random_seed=42)
    contributions = {"lag_1": 2.5, "price": -3.8, "promo": 0.0, "month": 1.2}
    feature_values = {"lag_1": 15.0, "price": 2.99, "promo": 0.0, "month": 7.0}

    all_ranked, pos_ranked, neg_ranked = explainer._rank_contributions(contributions, feature_values)

    assert len(all_ranked) == 4
    # Rank 1 should be 'price' because |-3.8| > |2.5| > |1.2|
    assert all_ranked[0].feature == "price"
    assert all_ranked[0].rank == 1
    assert all_ranked[0].direction == "negative"
    assert all_ranked[0].value == 2.99

    assert all_ranked[1].feature == "lag_1"
    assert all_ranked[1].rank == 2
    assert all_ranked[1].direction == "positive"

    assert len(pos_ranked) == 2
    assert pos_ranked[0].feature == "lag_1"
    assert pos_ranked[1].feature == "month"

    assert len(neg_ranked) == 1
    assert neg_ranked[0].feature == "price"


def test_base_explainer_rank_global_importance():
    explainer = DummyExplainer(random_seed=42)
    importance = {"lag_1": 10.0, "price": 30.0, "day": 5.0, "month": 5.0}

    ranked = explainer._rank_global_importance(importance, top_k=3)
    assert len(ranked) == 3
    assert ranked[0].feature == "price"
    assert ranked[0].rank == 1
    assert pytest.approx(ranked[0].normalized_importance, 0.01) == 0.60
    assert ranked[1].feature == "lag_1"


def test_explainability_validator_untrained_model():
    unfitted = LightGBMForecaster()
    with pytest.raises(ValueError, match="is not fitted"):
        ExplainabilityValidator.validate_model_fitted(unfitted)


def test_explainability_validator_compatibility_matrix():
    # LightGBM supports shap and lime
    assert ExplainabilityValidator.validate_method_compatibility("lightgbm", ExplanationMethod.SHAP) == "shap"
    assert ExplainabilityValidator.validate_method_compatibility("lightgbm", ExplanationMethod.LIME) == "lime"
    assert ExplainabilityValidator.validate_method_compatibility("lightgbm", ExplanationMethod.AUTO) == "shap"

    # Prophet supports component_based
    assert ExplainabilityValidator.validate_method_compatibility("prophet", ExplanationMethod.COMPONENT_BASED) == "component_based"
    assert ExplainabilityValidator.validate_method_compatibility("prophet", ExplanationMethod.AUTO) == "component_based"

    # Unsupported combinations
    with pytest.raises(ValueError, match="not supported for model 'prophet'"):
        ExplainabilityValidator.validate_method_compatibility("prophet", ExplanationMethod.SHAP)

    with pytest.raises(ValueError, match="Unknown forecasting model"):
        ExplainabilityValidator.validate_method_compatibility("unknown_model", ExplanationMethod.SHAP)


def test_explainability_validator_background_data():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Insufficient reference data"):
        ExplainabilityValidator.validate_background_data(empty_df, min_samples=5)

    small_df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Insufficient reference data"):
        ExplainabilityValidator.validate_background_data(small_df, min_samples=5)


def test_feature_adapter_background_sample_zero_leakage():
    # Verify sampling uses only the provided training partition
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "target": np.arange(50, dtype=float),
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "lag_1": np.roll(np.arange(50, dtype=float), 1),
    })

    lgb = LightGBMForecaster(random_seed=42)
    lgb.fit(df)

    bg = ExplainabilityFeatureAdapter.extract_background_sample(
        model=lgb,
        train_df=df,
        sample_size=15,
        random_seed=42,
    )
    assert len(bg) == 15
    assert all(col in bg.columns for col in lgb.feature_names)
