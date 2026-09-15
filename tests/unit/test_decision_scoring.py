"""Unit tests for transparent Decision Scoring and Confidence Calibration."""

import pytest
from backend.app.schemas.decisions import BusinessContext, DecisionPriority
from backend.app.schemas.forecasting import ForecastPoint, ForecastResult
from backend.app.decision_intelligence.scoring import DecisionScorer


def test_forecast_dynamics_calculation():
    points = [
        ForecastPoint(date=f"2024-08-{i:02d}", prediction=10.0 + i, lower_bound=8.0 + i, upper_bound=12.0 + i)
        for i in range(1, 15)
    ]
    fcst = ForecastResult(
        forecast_id="fcst_test",
        dataset_id="test",
        horizon=14,
        frequency="D",
        model_name="lightgbm",
        status="SUCCESS",
        predictions=points,
    )
    dyn = DecisionScorer.calculate_forecast_dynamics(fcst)
    assert dyn["count"] == 14
    assert dyn["trend_direction"] == "increasing"
    assert dyn["has_intervals"] is True
    assert dyn["relative_uncertainty"] > 0.0


def test_confidence_penalizes_missing_inventory():
    points = [
        ForecastPoint(date=f"2024-08-{i:02d}", prediction=15.0)
        for i in range(1, 8)
    ]
    fcst = ForecastResult(
        forecast_id="fcst_test",
        dataset_id="test",
        horizon=7,
        frequency="D",
        model_name="lightgbm",
        status="SUCCESS",
        predictions=points,
    )
    # Context without current inventory
    ctx_missing = BusinessContext(current_inventory=None)
    conf_missing, assumptions, warnings = DecisionScorer.compute_decision_confidence(
        forecast=fcst,
        explanation=None,
        context=ctx_missing,
    )

    ctx_present = BusinessContext(current_inventory=100.0, reorder_point=50.0, lead_time_days=7)
    conf_present, _, _ = DecisionScorer.compute_decision_confidence(
        forecast=fcst,
        explanation=None,
        context=ctx_present,
    )

    assert conf_missing < conf_present
    assert any("inventory" in a.lower() for a in assumptions)
    assert any("current_inventory" in w for w in warnings)


def test_recommendation_score_bounded_and_prioritized():
    score_crit = DecisionScorer.compute_recommendation_score(
        priority=DecisionPriority.CRITICAL,
        rule_severity=0.9,
        demand_gap_factor=1.0,
        explanation_support_score=0.8,
        confidence=0.85,
        has_missing_context=False,
    )
    score_low = DecisionScorer.compute_recommendation_score(
        priority=DecisionPriority.LOW,
        rule_severity=0.3,
        demand_gap_factor=0.0,
        explanation_support_score=0.5,
        confidence=0.85,
        has_missing_context=False,
    )

    assert 0.0 <= score_crit <= 100.0
    assert 0.0 <= score_low <= 100.0
    assert score_crit > score_low
