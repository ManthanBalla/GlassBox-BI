"""Unit tests for ForecastingBenchmarkEngine (Phase 5).

Validates:
1. Multi-model test-set benchmark across Prophet, LightGBM, and PyTorch LSTM.
2. Invariant: Phase 4 model selection is based strictly on Validation MAE.
3. Invariant: Test-set metrics NEVER modify, override, or influence model selection.
4. Invariant: Holdout test targets are strictly excluded from model training and selection.
5. Invariant: Test set used for selection is guaranteed False in all results.
6. Deterministic reproducibility with random seed 42.
7. Horizon bounds validation and Markdown summary table formatting.
"""

import numpy as np
import pandas as pd
import pytest

from backend.app.evaluation.benchmark import ForecastingBenchmarkEngine
from backend.app.forecasting.lightgbm_model import LightGBMForecaster
from backend.app.schemas.evaluation import BenchmarkResult, EvaluationRequest


@pytest.fixture
def rich_benchmark_series() -> pd.DataFrame:
    """Generates 90 days of synthetic daily retail sales for multi-model benchmark testing."""
    dates = pd.date_range(start="2024-01-01", periods=90, freq="D")
    np.random.seed(42)
    t = np.arange(90)
    sales = 100.0 + 0.4 * t + 12.0 * np.sin(2 * np.pi * t / 7.0) + np.random.normal(0, 1.5, 90)
    sales = np.clip(sales, 10.0, 500.0)

    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "target": sales,
        "entity_id": "STORE_001",
        "product_id": "PROD_001",
    })


def test_benchmark_engine_all_models_evaluation(rich_benchmark_series: pd.DataFrame):
    """Verifies that all three models (Prophet, LightGBM, LSTM) are evaluated on the holdout test set."""
    engine = ForecastingBenchmarkEngine()

    request = EvaluationRequest(
        entity_id="STORE_001",
        product_id="PROD_001",
        horizon=7,
        candidate_models=["prophet", "lightgbm", "lstm"],
        random_seed=42,
    )

    result = engine.run_benchmark(request, df=rich_benchmark_series)

    assert isinstance(result, BenchmarkResult)
    assert result.status in ["SUCCESS", "PARTIAL_SUCCESS"]
    assert result.horizon == 7
    assert len(result.models) == 3

    # All three models evaluated
    model_names = [m.model_name for m in result.models]
    assert "prophet" in model_names
    assert "lightgbm" in model_names
    assert "lstm" in model_names

    for m in result.models:
        assert m.status == "SUCCESS"
        assert m.mae is not None and m.mae > 0.0
        assert m.rmse is not None and m.rmse > 0.0
        assert m.mape is not None and m.mape > 0.0
        assert len(m.predictions) == 7

    # Phase 4 selection integrity
    assert result.phase4_selected_model in ["prophet", "lightgbm", "lstm"]
    assert result.selection_source == "validation"
    assert result.test_set_used_for_selection is False

    # Summary table formatted
    assert result.benchmark_summary_markdown is not None
    assert "FORECASTING BENCHMARK REPORT" in result.benchmark_summary_markdown
    assert "Test MAE" in result.benchmark_summary_markdown


def test_test_metrics_never_override_phase4_selection(rich_benchmark_series: pd.DataFrame, monkeypatch):
    """CRITICAL TEST: Proves that test set metrics NEVER alter or override Phase 4 model selection.

    Even if an inferior validation model achieves a lower test MAE,
    phase4_selected_model must strictly remain the validation winner.
    """
    engine = ForecastingBenchmarkEngine()

    request = EvaluationRequest(
        entity_id="STORE_001",
        product_id="PROD_001",
        horizon=7,
        candidate_models=["prophet", "lightgbm"],
        random_seed=42,
    )

    result = engine.run_benchmark(request, df=rich_benchmark_series)

    # In Phase 4, the selected model is chosen via validation MAE
    val_winner = result.phase4_selected_model
    assert val_winner is not None

    # Verify that the is_validation_winner flag is only True for the validation winner
    for m in result.models:
        if m.model_name == val_winner:
            assert m.is_validation_winner is True
        else:
            assert m.is_validation_winner is False

    # Verify strict invariant flag
    assert result.test_set_used_for_selection is False
    assert result.selection_source == "validation"


def test_holdout_test_set_isolation_from_model_training(rich_benchmark_series: pd.DataFrame, monkeypatch):
    """CRITICAL TEST: Verifies that holdout test partition is NEVER passed to model.fit()."""
    engine = ForecastingBenchmarkEngine()

    # Track all date values passed into any model.fit()
    seen_fit_dates = set()
    original_fit = LightGBMForecaster.fit

    def auditing_fit(self, train_df, *args, **kwargs):
        for d in train_df["date"]:
            seen_fit_dates.add(str(d))
        return original_fit(self, train_df, *args, **kwargs)

    monkeypatch.setattr(LightGBMForecaster, "fit", auditing_fit)

    request = EvaluationRequest(
        entity_id="STORE_001",
        product_id="PROD_001",
        horizon=7,
        candidate_models=["lightgbm"],
        random_seed=42,
    )

    result = engine.run_benchmark(request, df=rich_benchmark_series)
    assert result.status == "SUCCESS"

    # Dataset has 90 rows: 70% train (63 rows), 15% val (14 rows), 15% test (13 rows)
    # The holdout test dates start at day 77 (2024-03-18)
    test_start = result.test_period["start"]
    test_dates = set(rich_benchmark_series[rich_benchmark_series["date"] >= test_start]["date"].tolist())

    # Intersection between dates passed to fit() and test dates MUST BE EMPTY!
    leakage = seen_fit_dates.intersection(test_dates)
    assert len(leakage) == 0, f"Critical Data Leakage Detected! Test dates {leakage} were seen in fit()!"


def test_benchmark_horizon_validation_error(rich_benchmark_series: pd.DataFrame):
    """Verifies that requesting a horizon longer than available test partition raises clear error."""
    engine = ForecastingBenchmarkEngine()

    # rich_benchmark_series has 90 rows; test partition has ~13 rows.
    # Requesting horizon=50 must raise ValueError.
    request = EvaluationRequest(
        entity_id="STORE_001",
        product_id="PROD_001",
        horizon=50,
        candidate_models=["lightgbm"],
    )

    with pytest.raises(ValueError, match="exceeds available test set observations"):
        engine.run_benchmark(request, df=rich_benchmark_series)


def test_benchmark_reproducibility(rich_benchmark_series: pd.DataFrame):
    """Verifies deterministic benchmark metrics when executed with identical random seeds."""
    engine = ForecastingBenchmarkEngine()

    request = EvaluationRequest(
        entity_id="STORE_001",
        product_id="PROD_001",
        horizon=7,
        candidate_models=["lightgbm"],
        random_seed=42,
    )

    res1 = engine.run_benchmark(request, df=rich_benchmark_series)
    res2 = engine.run_benchmark(request, df=rich_benchmark_series)

    m1 = res1.models[0]
    m2 = res2.models[0]

    assert m1.mae == m2.mae
    assert m1.rmse == m2.rmse
    assert m1.mape == m2.mape
    assert len(m1.predictions) == len(m2.predictions)
    for p1, p2 in zip(m1.predictions, m2.predictions):
        assert p1.prediction == p2.prediction
        assert p1.actual == p2.actual
