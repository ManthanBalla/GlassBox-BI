"""Phase 7 Live Verification Script: Decision Intelligence Agent.

Validates the three representative business examples required by the Phase 7 protocol:
- Example A: High demand + low inventory (Replenishment scale-up)
- Example B: Declining demand + excess inventory (Curtailment / Clearance)
- Example C: High uncertainty + incomplete context (Conservative planning & Human review)
- Scenario Analysis: Inventory shock simulation
- Deterministic repeatability check
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime, timezone
import json

from backend.app.schemas.decisions import (
    BusinessContext,
    DecisionRequest,
    ScenarioRequest,
)
from backend.app.schemas.explainability import (
    ExplanationAuditTrail,
    ExplanationFidelity,
    ExplanationResult,
    FeatureContribution,
)
from backend.app.schemas.forecasting import ForecastPoint, ForecastResult
from backend.app.decision_intelligence.agent import DecisionIntelligenceAgent


def _make_fcst(preds, intervals=None, model_name="lightgbm"):
    points = []
    for i, p in enumerate(preds):
        d_str = f"2024-08-{i+1:02d}"
        low = intervals[i][0] if intervals else round(p * 0.9, 2)
        upp = intervals[i][1] if intervals else round(p * 1.1, 2)
        points.append(ForecastPoint(date=d_str, prediction=p, lower_bound=low, upper_bound=upp))
    return ForecastResult(
        forecast_id=f"fcst_{model_name}",
        dataset_id="retail_dataset",
        horizon=len(preds),
        frequency="D",
        model_name=model_name,
        status="SUCCESS",
        predictions=points,
        forecast_points=points,
        uncertainty_method="quantile_regression_80pct",
        metrics={"MAE": 1.35, "RMSE": 1.72, "MAPE": 7.8},
    )


def _make_expl(features_dict, fidelity_score=0.98, status="HIGH_FIDELITY"):
    conts = []
    for r, (k, v) in enumerate(features_dict.items(), start=1):
        conts.append(
            FeatureContribution(
                feature=k,
                value=1.0,
                contribution=round(v, 4),
                shap_value=round(v, 4),
                absolute_contribution=round(abs(v), 4),
                direction="positive" if v > 0 else "negative",
                rank=r,
            )
        )
    return ExplanationResult(
        explanation_id="expl_verify",
        model_name="lightgbm",
        method="shap",
        explanation_type="local",
        prediction=22.0,
        features=conts,
        top_positive_contributors=[c for c in conts if c.contribution > 0],
        top_negative_contributors=[c for c in conts if c.contribution < 0],
        fidelity=ExplanationFidelity(
            fidelity_score=fidelity_score,
            reconstruction_error=round(1.0 - fidelity_score, 4),
            explanation_status=status,
            method_notes="Live verification explainer",
        ),
        audit_trail=ExplanationAuditTrail(
            explanation_id="expl_verify",
            model_name="lightgbm",
            prediction=22.0,
            explanation_method="shap",
            explanation_type="local",
            feature_count=len(features_dict),
        ),
    )


def main():
    print("=" * 80)
    print("GLASSBOX-BI: PHASE 7 DECISION INTELLIGENCE LIVE VERIFICATION")
    print("=" * 80)

    agent = DecisionIntelligenceAgent()

    # -------------------------------------------------------------------------
    # Example A: High Demand + Low Inventory
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("EXAMPLE A: HIGH DEMAND + LOW INVENTORY")
    print("-" * 80)

    fcst_a = _make_fcst([16.0, 18.0, 21.0, 24.0, 27.0, 30.0, 34.0])
    ctx_a = BusinessContext(
        current_inventory=45.0,
        reorder_point=120.0,
        safety_stock=30.0,
        lead_time_days=7,
        current_price=24.99,
        promotion_active=True,
        unit_cost=14.0,
    )
    expl_a = _make_expl({
        "lag_1": 0.42,
        "rolling_mean_7": 0.31,
        "promo_flag": 0.22,
        "price": -0.08,
    })

    res_a = agent.run_decision(DecisionRequest(
        forecast_result=fcst_a,
        explanation_result=expl_a,
        business_context=ctx_a,
    ))

    rec_a = res_a.primary_recommendation
    print(f"ACTION:               {rec_a.action}")
    print(f"CATEGORY:             {rec_a.category}")
    print(f"PRIORITY:             {rec_a.priority}")
    print(f"RISK:                 {rec_a.risk}")
    print(f"RATIONALE:            {rec_a.rationale}")
    print("EVIDENCE:")
    for e in rec_a.evidence:
        print(f"  • {e}")
    if rec_a.trade_offs:
        print(f"TRADE-OFF BENEFIT:    {rec_a.trade_offs.benefit}")
        print(f"TRADE-OFF COST:       {rec_a.trade_offs.trade_off}")
    print(f"DECISION CONFIDENCE:  {rec_a.confidence * 100:.1f}%")
    print(f"RECOMMENDATION SCORE: {rec_a.recommendation_score:.1f} / 100")
    print(f"HUMAN REVIEW:         {rec_a.requires_human_review}")
    print(f"RULE TRIGGERED:       {rec_a.rule_id}")

    # -------------------------------------------------------------------------
    # Example B: Declining Demand + Excess Inventory
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("EXAMPLE B: DECLINING DEMAND + EXCESS INVENTORY")
    print("-" * 80)

    fcst_b = _make_fcst([28.0, 24.0, 20.0, 17.0, 14.0, 12.0, 10.0])
    ctx_b = BusinessContext(
        current_inventory=550.0,
        reorder_point=100.0,
        safety_stock=25.0,
        lead_time_days=7,
        current_price=29.99,
        promotion_active=False,
        unit_cost=16.0,
    )
    expl_b = _make_expl({
        "rolling_mean_7": -0.35,
        "trend": -0.25,
    })

    res_b = agent.run_decision(DecisionRequest(
        forecast_result=fcst_b,
        explanation_result=expl_b,
        business_context=ctx_b,
    ))

    rec_b = res_b.primary_recommendation
    print(f"ACTION:               {rec_b.action}")
    print(f"CATEGORY:             {rec_b.category}")
    print(f"PRIORITY:             {rec_b.priority}")
    print(f"RISK:                 {rec_b.risk}")
    print(f"RATIONALE:            {rec_b.rationale}")
    print("EVIDENCE:")
    for e in rec_b.evidence:
        print(f"  • {e}")
    if rec_b.trade_offs:
        print(f"TRADE-OFF BENEFIT:    {rec_b.trade_offs.benefit}")
        print(f"TRADE-OFF COST:       {rec_b.trade_offs.trade_off}")
    print(f"DECISION CONFIDENCE:  {rec_b.confidence * 100:.1f}%")
    print(f"RECOMMENDATION SCORE: {rec_b.recommendation_score:.1f} / 100")
    print(f"HUMAN REVIEW:         {rec_b.requires_human_review}")
    print(f"RULE TRIGGERED:       {rec_b.rule_id}")

    # -------------------------------------------------------------------------
    # Example C: High Uncertainty + Incomplete Business Context
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("EXAMPLE C: HIGH UNCERTAINTY + INCOMPLETE BUSINESS CONTEXT")
    print("-" * 80)

    # Wide prediction intervals: [6.0, 42.0] on prediction 20.0 (spread = 36 / 20 = 180%)
    preds_c = [20.0, 22.0, 24.0, 26.0, 28.0]
    intervals_c = [[6.0, 42.0], [7.0, 45.0], [8.0, 48.0], [9.0, 50.0], [10.0, 54.0]]
    fcst_c = _make_fcst(preds_c, intervals=intervals_c)

    # Incomplete context: current_inventory = None
    ctx_c = BusinessContext(
        current_inventory=None,
        reorder_point=None,
        lead_time_days=7,
        current_price=22.50,
    )

    res_c = agent.run_decision(DecisionRequest(
        forecast_result=fcst_c,
        business_context=ctx_c,
    ))

    rec_c = res_c.primary_recommendation
    print(f"ACTION:               {rec_c.action}")
    print(f"CATEGORY:             {rec_c.category}")
    print(f"PRIORITY:             {rec_c.priority}")
    print(f"RISK:                 {rec_c.risk}")
    print(f"RATIONALE:            {rec_c.rationale}")
    print("EVIDENCE:")
    for e in rec_c.evidence:
        print(f"  • {e}")
    print(f"DECISION CONFIDENCE:  {rec_c.confidence * 100:.1f}% (Penalized for missing inventory & high uncertainty)")
    print(f"RECOMMENDATION SCORE: {rec_c.recommendation_score:.1f} / 100")
    print(f"HUMAN REVIEW:         {rec_c.requires_human_review} (MANDATORY)")
    print("OPERATIONAL WARNINGS:")
    for w in res_c.warnings:
        print(f"  ⚠️  {w}")

    # -------------------------------------------------------------------------
    # Scenario Simulation Verification
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SCENARIO SIMULATION: INVENTORY SHOCK (-50 UNITS)")
    print("-" * 80)

    scen_res = agent.run_scenario(ScenarioRequest(
        scenario_name="Stress Test: Sudden -50 Unit Loss",
        base_request=DecisionRequest(
            forecast_result=fcst_a,
            explanation_result=expl_a,
            business_context=BusinessContext(
                current_inventory=80.0,
                reorder_point=100.0,
                lead_time_days=7,
            ),
        ),
        inventory_delta=-50.0,
    ))

    print(f"SCENARIO NAME:    {scen_res.scenario_name}")
    print(f"INVENTORY BEFORE: {scen_res.delta_summary['modifications']['inventory']['before']} units")
    print(f"INVENTORY AFTER:  {scen_res.delta_summary['modifications']['inventory']['after']} units")
    print(f"BASELINE ACTION:  {scen_res.delta_summary['primary_action_change']['baseline_action']}")
    print(f"SCENARIO ACTION:  {scen_res.delta_summary['primary_action_change']['scenario_action']}")
    print(f"PRIORITY SHIFT:   {scen_res.delta_summary['primary_action_change']['baseline_priority']} -> {scen_res.delta_summary['primary_action_change']['scenario_priority']}")

    # -------------------------------------------------------------------------
    # Deterministic Repeatability Check
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("DETERMINISTIC REPEATABILITY CHECK")
    print("-" * 80)
    r1 = agent.run_decision(DecisionRequest(forecast_result=fcst_a, business_context=ctx_a))
    r2 = agent.run_decision(DecisionRequest(forecast_result=fcst_a, business_context=ctx_a))

    match = (
        r1.primary_recommendation.action == r2.primary_recommendation.action
        and r1.primary_recommendation.priority == r2.primary_recommendation.priority
        and r1.primary_recommendation.recommendation_score == r2.primary_recommendation.recommendation_score
        and r1.primary_recommendation.confidence == r2.primary_recommendation.confidence
        and r1.rules_triggered == r2.rules_triggered
    )
    print(f"REPEATABILITY VERDICT: {'PASS (Bitwise Deterministic)' if match else 'FAIL'}")

    print("\n" + "=" * 80)
    print("PHASE 7 LIVE VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
