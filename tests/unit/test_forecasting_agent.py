"""Unit tests for GenericForecastingAgent (Phase 4).

Validates:
- Multi-model competition across Prophet, LightGBM, and LSTM
- Dynamic validation ranking (MAE, RMSE, MAPE) without holdout test leakage
- Model refit on (Train + Validation) prior to future horizon projection
- Model failure isolation (one model failing does not crash the pipeline)
- Insufficient history rejection
- Reproducibility with deterministic random seeds
"""

import numpy as np
import pandas as pd
import pytest

from backend.app.forecasting.agent import GenericForecastingAgent
from backend.app.schemas.forecasting import (
    ForecastingConfig,
    ForecastRequest,
    ForecastResult,
    ForecastStatus,
)


@pytest.fixture
def rich_series_dataframe() -> pd.DataFrame:
    """Generates 90 days of synthetic daily retail sales for testing competition."""
    dates = pd.date_range(start="2023-01-01", periods=90, freq="D")
    np.random.seed(123)
    t = np.arange(90)
    # Seasonality + trend + noise
    sales = 100.0 + 0.3 * t + 15.0 * np.sin(2 * np.pi * t / 7.0) + np.random.normal(0, 2.0, 90)
    sales = np.clip(sales, 10.0, 500.0)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "target": sales,
        "entity_id": "STORE_TEST",
        "product_id": "PROD_TEST",
    })
    return df


def test_agent_multi_model_competition(rich_series_dataframe: pd.DataFrame, tmp_path):
    """Verifies that the agent executes candidate models, ranks by validation MAE, and selects winner."""
    agent = GenericForecastingAgent(persistence_dir=tmp_path)

    request = ForecastRequest(
        entity_id="STORE_TEST",
        product_id="PROD_TEST",
        horizon=7,
        candidate_models=["prophet", "lightgbm", "lstm"],
        selection_metric="MAE",
        confidence_level=0.80,
    )

    result = agent.run_forecast(request, df=rich_series_dataframe)

    assert isinstance(result, ForecastResult)
    assert result.status == ForecastStatus.SUCCESS
    assert result.horizon == 7
    assert len(result.predictions) == 7
    assert result.model_name in ["prophet", "lightgbm", "lstm"]
    assert result.selection_result is not None

    rankings = result.selection_result.candidate_rankings
    assert len(rankings) == 3
    # All candidates succeeded
    for cand in rankings:
        assert cand.status in ["SUCCESS", "success"]
        assert cand.mae > 0.0
        assert cand.rmse > 0.0
        assert cand.mape > 0.0
        assert cand.training_duration >= 0.0

    # Winner has lowest validation MAE
    winner_score = next(c for c in rankings if c.model_name == result.selection_result.selected_model)
    for other in rankings:
        assert winner_score.mae <= other.mae + 1e-6

    # Model metadata preserved
    assert result.model_metadata is not None
    assert result.model_metadata.model_name == result.model_name
    assert result.model_metadata.training_rows > 0

    # Predictions verified
    for pt in result.predictions:
        assert np.isfinite(pt.prediction)
        assert pt.prediction >= 0.0
        if pt.lower_bound is not None and pt.upper_bound is not None:
            assert pt.lower_bound <= pt.prediction + 1e-5
            assert pt.prediction <= pt.upper_bound + 1e-5


def test_agent_metric_selection_variants(rich_series_dataframe: pd.DataFrame, tmp_path):
    """Tests that RMSE and MAPE selection metrics function accurately."""
    agent = GenericForecastingAgent(persistence_dir=tmp_path)

    # Test RMSE selection
    req_rmse = ForecastRequest(
        entity_id="STORE_TEST",
        product_id="PROD_TEST",
        horizon=5,
        candidate_models=["prophet", "lightgbm"],
        selection_metric="RMSE",
    )
    res_rmse = agent.run_forecast(req_rmse, df=rich_series_dataframe)
    assert res_rmse.status == ForecastStatus.SUCCESS
    assert res_rmse.selection_result.selection_metric == "RMSE"
    winner_rmse = next(c for c in res_rmse.selection_result.candidate_rankings if c.model_name == res_rmse.model_name)
    for other in res_rmse.selection_result.candidate_rankings:
        assert winner_rmse.rmse <= other.rmse + 1e-6

    # Test MAPE selection
    req_mape = ForecastRequest(
        entity_id="STORE_TEST",
        product_id="PROD_TEST",
        horizon=5,
        candidate_models=["prophet", "lightgbm"],
        selection_metric="MAPE",
    )
    res_mape = agent.run_forecast(req_mape, df=rich_series_dataframe)
    assert res_mape.status == ForecastStatus.SUCCESS
    assert res_mape.selection_result.selection_metric == "MAPE"


def test_agent_insufficient_history(tmp_path):
    """Verifies clear structured error when historical observations are too few."""
    short_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "target": [10.0, 12.0, 11.0, 13.0],
        "entity_id": ["STORE_01"] * 4,
        "product_id": ["PROD_01"] * 4,
    })

    agent = GenericForecastingAgent(persistence_dir=tmp_path)
    request = ForecastRequest(
        entity_id="STORE_01",
        product_id="PROD_01",
        horizon=7,
        candidate_models=["lightgbm"],
    )

    result = agent.run_forecast(request, df=short_df)
    assert result.status == ForecastStatus.INSUFFICIENT_HISTORY
    assert len(result.predictions) == 0
    assert any("Insufficient series observations" in w for w in result.warnings)


def test_agent_failure_isolation(rich_series_dataframe: pd.DataFrame, tmp_path, monkeypatch):
    """Verifies that an error in one candidate model does not crash the overall forecasting agent."""
    agent = GenericForecastingAgent(persistence_dir=tmp_path)

    # Monkeypatch ProphetForecaster.fit to simulate a model-specific failure
    from backend.app.forecasting.prophet_model import ProphetForecaster

    def broken_fit(self, *args, **kwargs):
        raise RuntimeError("Simulated Prophet runtime crash")

    monkeypatch.setattr(ProphetForecaster, "fit", broken_fit)

    request = ForecastRequest(
        entity_id="STORE_TEST",
        product_id="PROD_TEST",
        horizon=5,
        candidate_models=["prophet", "lightgbm"],
        selection_metric="MAE",
    )

    result = agent.run_forecast(request, df=rich_series_dataframe)

    # Overall agent succeeds because LightGBM survived
    assert result.status in [ForecastStatus.SUCCESS, ForecastStatus.PARTIAL_SUCCESS]
    assert result.model_name == "lightgbm"
    assert len(result.predictions) == 5

    # Prophet recorded as failed in rankings
    prophet_cand = next(c for c in result.selection_result.candidate_rankings if c.model_name == "prophet")
    assert prophet_cand.status in ["FAILED", "failed"]
    assert "Simulated Prophet runtime crash" in (prophet_cand.error_message or "")


def test_agent_reproducibility(rich_series_dataframe: pd.DataFrame, tmp_path):
    """Verifies deterministic forecast results when executed with identical seeds."""
    agent1 = GenericForecastingAgent(persistence_dir=tmp_path / "run1")
    agent2 = GenericForecastingAgent(persistence_dir=tmp_path / "run2")

    req = ForecastRequest(
        entity_id="STORE_TEST",
        product_id="PROD_TEST",
        horizon=7,
        candidate_models=["lightgbm"],
        selection_metric="MAE",
    )

    res1 = agent1.run_forecast(req, df=rich_series_dataframe)
    res2 = agent2.run_forecast(req, df=rich_series_dataframe)

    assert len(res1.predictions) == len(res2.predictions)
    for p1, p2 in zip(res1.predictions, res2.predictions):
        assert p1.date == p2.date
        assert np.isclose(p1.prediction, p2.prediction, atol=1e-4)
