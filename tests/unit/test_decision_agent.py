"""Unit tests for DecisionIntelligenceAgent execution, scenario simulations, and audit trails."""

import pytest
from backend.app.schemas.decisions import (
    BusinessContext,
    DecisionRequest,
    ScenarioRequest,
)
from backend.app.schemas.forecasting import ForecastPoint, ForecastResult
from backend.app.decision_intelligence.agent import DecisionIntelligenceAgent


def test_agent_sample_decision():
    agent = DecisionIntelligenceAgent()
    sample = agent.get_sample_decision()
    assert sample.decision_id.startswith("dec_")
    assert sample.primary_recommendation is not None
    assert len(sample.recommendations) > 0
    assert sample.audit_trail is not None
    assert sample.audit_trail.execution_duration_ms > 0.0


def test_agent_scenario_inventory_cut():
    agent = DecisionIntelligenceAgent()
    sample_res = agent.get_sample_decision()

    base_fcst = ForecastResult(
        forecast_id="fcst_scen",
        dataset_id="test",
        horizon=7,
        frequency="D",
        model_name="lightgbm",
        status="SUCCESS",
        predictions=[
            ForecastPoint(date=f"2024-08-{i:02d}", prediction=20.0, lower_bound=16.0, upper_bound=24.0)
            for i in range(1, 8)
        ],
    )
    base_ctx = BusinessContext(
        current_inventory=140.0,
        reorder_point=80.0,
        lead_time_days=7,
    )
    base_req = DecisionRequest(
        forecast_result=base_fcst,
        business_context=base_ctx,
    )

    scen_req = ScenarioRequest(
        scenario_name="Simulated Stock Cut 80 Units",
        base_request=base_req,
        inventory_delta=-80.0,  # Drop inventory to 60 units (below reorder point)
    )

    scen_res = agent.run_scenario(scen_req)
    assert scen_res.scenario_name == "Simulated Stock Cut 80 Units"
    assert scen_res.baseline_decision is not None
    assert scen_res.scenario_decision is not None

    # Before cut: inventory was 140
    # After cut: inventory is 60 -> trigger replenishment
    assert "modifications" in scen_res.delta_summary
    assert scen_res.delta_summary["modifications"]["inventory"]["after"] == 60.0
    assert scen_res.scenario_decision.primary_recommendation is not None


def test_agent_scenario_price_elasticity():
    agent = DecisionIntelligenceAgent()
    base_fcst = ForecastResult(
        forecast_id="fcst_scen_price",
        dataset_id="test",
        horizon=5,
        frequency="D",
        model_name="lightgbm",
        status="SUCCESS",
        predictions=[
            ForecastPoint(date=f"2024-08-{i:02d}", prediction=10.0)
            for i in range(1, 6)
        ],
    )
    base_ctx = BusinessContext(
        current_inventory=100.0,
        current_price=20.0,
    )
    base_req = DecisionRequest(
        forecast_result=base_fcst,
        business_context=base_ctx,
    )

    # 10% price discount (-10) -> elasticity -1.5 -> demand multiplier 1.0 + (-1.5 * -0.1) = 1.15 (+15% demand)
    scen_req = ScenarioRequest(
        scenario_name="10% Discount Campaign",
        base_request=base_req,
        price_delta_percent=-10.0,
    )
    scen_res = agent.run_scenario(scen_req)
    assert "price_elasticity" in scen_res.delta_summary["modifications"]
    mult = scen_res.delta_summary["modifications"]["price_elasticity"]["demand_multiplier"]
    assert mult > 1.0
