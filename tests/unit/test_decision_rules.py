"""Unit tests verifying the 8 mandatory Phase 7 retail decision test cases."""

from datetime import datetime, timezone
import pytest

from backend.app.schemas.decisions import (
    BusinessActionCategory,
    BusinessContext,
    DecisionPriority,
    DecisionRequest,
)
from backend.app.schemas.explainability import (
    ExplanationAuditTrail,
    ExplanationFidelity,
    ExplanationResult,
    FeatureContribution,
)
from backend.app.schemas.forecasting import ForecastPoint, ForecastResult
from backend.app.decision_intelligence.agent import DecisionIntelligenceAgent
from backend.app.decision_intelligence.rules import RetailDecisionRuleEngine


def _make_forecast(
    preds_list,
    intervals=None,
    model_name="lightgbm",
    metrics=None,
    uncertainty_method="quantile_regression",
) -> ForecastResult:
    """Helper to construct a ForecastResult."""
    points = []
    for i, val in enumerate(preds_list):
        date_str = f"2024-08-{i+1:02d}"
        lower = intervals[i][0] if intervals else round(val * 0.9, 2)
        upper = intervals[i][1] if intervals else round(val * 1.1, 2)
        points.append(
            ForecastPoint(
                date=date_str,
                prediction=float(val),
                lower_bound=float(lower),
                upper_bound=float(upper),
            )
        )
    return ForecastResult(
        forecast_id="fcst_test",
        dataset_id="retail_dataset",
        horizon=len(preds_list),
        frequency="D",
        model_name=model_name,
        status="SUCCESS",
        predictions=points,
        forecast_points=points,
        uncertainty_method=uncertainty_method,
        metrics=metrics or {"MAE": 1.2, "RMSE": 1.6, "MAPE": 7.5},
    )


def _make_explanation(
    features_dict,
    fidelity_score=0.95,
    fidelity_status="HIGH_FIDELITY",
    method="shap",
) -> ExplanationResult:
    """Helper to construct an ExplanationResult."""
    contributions = []
    for rank, (feat, val) in enumerate(features_dict.items(), start=1):
        v = float(val)
        contributions.append(
            FeatureContribution(
                feature=feat,
                value=1.0,
                contribution=round(v, 4),
                shap_value=round(v, 4),
                absolute_contribution=round(abs(v), 4),
                direction="positive" if v > 1e-5 else ("negative" if v < -1e-5 else "neutral"),
                rank=rank,
            )
        )
    fidelity = ExplanationFidelity(
        fidelity_score=fidelity_score,
        reconstruction_error=round(1.0 - fidelity_score, 4),
        explanation_status=fidelity_status,
        method_notes="Unit test synthetic explanation",
    )
    audit = ExplanationAuditTrail(
        explanation_id="expl_test",
        model_name="lightgbm",
        model_type="LightGBMForecaster",
        entity_id="STORE_001",
        product_id="PROD_001",
        prediction=20.0,
        explanation_method=method,
        explanation_type="local",
        feature_count=len(features_dict),
    )
    return ExplanationResult(
        explanation_id="expl_test",
        model_name="lightgbm",
        method=method,
        explanation_type="local",
        prediction=20.0,
        features=contributions,
        top_positive_contributors=[c for c in contributions if c.contribution > 0],
        top_negative_contributors=[c for c in contributions if c.contribution < 0],
        fidelity=fidelity,
        audit_trail=audit,
    )


# =============================================================================
# TEST CASE 1: High Demand + Low Inventory
# =============================================================================
def test_case_1_high_demand_low_inventory():
    """Forecast increasing strongly, inventory below reorder point, XAI supports recent demand.
    
    Expected: HIGH or CRITICAL replenishment recommendation.
    """
    agent = DecisionIntelligenceAgent()

    # Increasing demand: 15, 17, 19, 22, 25, 28, 32
    forecast = _make_forecast([15.0, 17.0, 19.0, 22.0, 25.0, 28.0, 32.0])
    context = BusinessContext(
        current_inventory=40.0,
        reorder_point=100.0,
        safety_stock=25.0,
        lead_time_days=7,
        unit_cost=10.0,
    )
    explanation = _make_explanation({
        "lag_1": 0.35,
        "rolling_mean_7": 0.28,
        "promo_flag": 0.15,
        "price": -0.05,
    })

    req = DecisionRequest(
        forecast_result=forecast,
        explanation_result=explanation,
        business_context=context,
    )
    result = agent.run_decision(req)

    assert result.primary_recommendation is not None
    rec = result.primary_recommendation
    assert rec.category == BusinessActionCategory.INVENTORY
    assert rec.priority in [DecisionPriority.HIGH, DecisionPriority.CRITICAL]
    assert "replenish" in rec.action.lower()
    assert rec.rule_id in ["RULE_1_HIGH_DEMAND_LOW_INVENTORY", "RULE_2_HIGH_STOCKOUT_CRITICAL"]
    assert rec.confidence >= 0.70
    assert any("inventory" in e.lower() for e in rec.evidence)
    assert any("demand" in e.lower() for e in rec.evidence)


# =============================================================================
# TEST CASE 2: Low Demand + High Inventory
# =============================================================================
def test_case_2_low_demand_high_inventory():
    """Forecast decreasing, inventory significantly high.
    
    Expected: Reduce replenishment / clearance planning recommendation.
    """
    agent = DecisionIntelligenceAgent()

    # Decreasing demand: 25, 22, 19, 16, 14, 12, 10
    forecast = _make_forecast([25.0, 22.0, 19.0, 16.0, 14.0, 12.0, 10.0])
    context = BusinessContext(
        current_inventory=650.0,  # Far above total horizon demand (approx 118)
        reorder_point=100.0,
        safety_stock=30.0,
        lead_time_days=7,
        unit_cost=12.0,
    )
    explanation = _make_explanation({
        "rolling_mean_7": -0.22,
        "trend": -0.18,
    })

    req = DecisionRequest(
        forecast_result=forecast,
        explanation_result=explanation,
        business_context=context,
    )
    result = agent.run_decision(req)

    assert result.primary_recommendation is not None
    rec = result.primary_recommendation
    assert rec.category == BusinessActionCategory.INVENTORY
    assert rec.rule_id == "RULE_3_LOW_DEMAND_HIGH_INVENTORY"
    assert "reduce replenishment" in rec.action.lower()
    assert rec.trade_offs is not None
    assert "carrying cost" in rec.trade_offs.benefit.lower() or "holding" in rec.trade_offs.benefit.lower()


# =============================================================================
# TEST CASE 3: High Forecast Uncertainty
# =============================================================================
def test_case_3_high_forecast_uncertainty():
    """Forecast increasing, prediction intervals very wide (> 40% relative width).
    
    Expected: Risk warning, lower decision confidence, human review required.
    """
    agent = DecisionIntelligenceAgent()

    # Point forecast: 20, intervals [5, 45] -> relative width = 40 / 20 = 2.0 (200%)
    preds = [20.0] * 7
    intervals = [[5.0, 45.0]] * 7
    forecast = _make_forecast(preds, intervals=intervals)
    context = BusinessContext(
        current_inventory=100.0,
        reorder_point=80.0,
        lead_time_days=7,
    )

    req = DecisionRequest(
        forecast_result=forecast,
        business_context=context,
        uncertainty_threshold=0.40,
    )
    result = agent.run_decision(req)

    # Triggered high uncertainty rule
    assert "RULE_6_HIGH_FORECAST_UNCERTAINTY" in result.rules_triggered
    # Check that human review is flagged
    assert result.requires_human_review is True
    # Check warning is present
    assert any("uncertainty" in w.lower() for w in result.warnings)


# =============================================================================
# TEST CASE 4: Low Explanation Fidelity
# =============================================================================
def test_case_4_low_explanation_fidelity():
    """Explanation fidelity is below threshold (< 0.65, status APPROXIMATE).
    
    Expected: Recommendation confidence downgraded, warning generated, human review required.
    """
    agent = DecisionIntelligenceAgent()

    forecast = _make_forecast([18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0])
    context = BusinessContext(
        current_inventory=150.0,
        reorder_point=100.0,
        lead_time_days=7,
    )
    # Low fidelity explanation (e.g. LIME R² = 0.204)
    explanation = _make_explanation(
        {"promo_flag": 0.45, "price": -0.30},
        fidelity_score=0.204,
        fidelity_status="APPROXIMATE",
        method="lime",
    )

    req = DecisionRequest(
        forecast_result=forecast,
        explanation_result=explanation,
        business_context=context,
        fidelity_threshold=0.65,
    )
    result = agent.run_decision(req)

    assert "RULE_7_LOW_EXPLANATION_FIDELITY" in result.rules_triggered
    assert result.requires_human_review is True
    assert any("fidelity" in w.lower() for w in result.warnings)
    # Decision confidence penalized
    assert result.audit_trail.decision_confidence < 0.80


# =============================================================================
# TEST CASE 5: Missing Inventory Context
# =============================================================================
def test_case_5_missing_inventory_context():
    """Missing inventory in business context (None).
    
    Expected: Zero value fabrication, planning recommendation explicitly mentioning missing inventory.
    """
    agent = DecisionIntelligenceAgent()

    forecast = _make_forecast([20.0, 22.0, 25.0, 27.0, 30.0])
    # BusinessContext with current_inventory = None
    context = BusinessContext(
        current_inventory=None,
        reorder_point=None,
        current_price=29.99,
        promotion_active=False,
    )

    req = DecisionRequest(
        forecast_result=forecast,
        business_context=context,
    )
    result = agent.run_decision(req)

    assert result.primary_recommendation is not None
    rec = result.primary_recommendation
    assert rec.category == BusinessActionCategory.PLANNING
    assert rec.rule_id == "RULE_MISSING_INVENTORY_PLANNING"
    assert "inventory planning review" in rec.action.lower()
    # Evidence must explicitly state zero fabrication
    assert any("not specified" in e.lower() or "no inventory values were fabricated" in e.lower() for e in rec.evidence)
    assert rec.requires_human_review is True


# =============================================================================
# TEST CASE 6: Price is Dominant Explanation Driver
# =============================================================================
def test_case_6_price_dominant_driver():
    """Price feature has dominant attribution driving demand.
    
    Expected: Pricing review recommendation, human review required, no auto price change.
    """
    agent = DecisionIntelligenceAgent()

    forecast = _make_forecast([16.0, 15.0, 14.0, 13.0, 12.0])
    context = BusinessContext(
        current_inventory=120.0,
        current_price=34.99,
        lead_time_days=7,
    )
    # Price is top negative driver with large magnitude
    explanation = _make_explanation({
        "price": -0.65,
        "lag_1": 0.10,
        "rolling_mean_7": 0.05,
    })

    req = DecisionRequest(
        forecast_result=forecast,
        explanation_result=explanation,
        business_context=context,
    )
    result = agent.run_decision(req)

    assert "RULE_5_PRICE_DRIVEN_DEMAND_CHANGE" in result.rules_triggered
    price_rec = next((r for r in result.recommendations if r.rule_id == "RULE_5_PRICE_DRIVEN_DEMAND_CHANGE"), None)
    assert price_rec is not None
    assert price_rec.category == BusinessActionCategory.PRICING
    assert "review pricing strategy" in price_rec.action.lower()
    assert price_rec.requires_human_review is True


# =============================================================================
# TEST CASE 7: Promotion is Dominant Positive Driver
# =============================================================================
def test_case_7_promotion_dominant_driver():
    """Promotion is active and dominant positive explanation driver.
    
    Expected: Promotion-aware recommendation with traceable evidence.
    """
    agent = DecisionIntelligenceAgent()

    forecast = _make_forecast([22.0, 25.0, 28.0, 30.0, 33.0])
    context = BusinessContext(
        current_inventory=180.0,
        reorder_point=100.0,
        lead_time_days=7,
        promotion_active=True,
    )
    explanation = _make_explanation({
        "promo_flag": 0.52,
        "rolling_mean_7": 0.20,
        "price": -0.10,
    })

    req = DecisionRequest(
        forecast_result=forecast,
        explanation_result=explanation,
        business_context=context,
    )
    result = agent.run_decision(req)

    assert "RULE_4_DEMAND_INCREASE_PROMOTION" in result.rules_triggered
    promo_rec = next((r for r in result.recommendations if r.rule_id == "RULE_4_DEMAND_INCREASE_PROMOTION"), None)
    assert promo_rec is not None
    assert promo_rec.category == BusinessActionCategory.PROMOTION
    assert "promotional strategy" in promo_rec.action.lower()
    assert any("promotion" in e.lower() for e in promo_rec.evidence)


# =============================================================================
# TEST CASE 8: Deterministic Repeatability
# =============================================================================
def test_case_8_deterministic_repeatability():
    """Same input executed twice must produce identical recommendations, scores, priorities, and rules.
    
    Expected: Bitwise deterministic decision logic (except timestamp/decision_id).
    """
    agent = DecisionIntelligenceAgent()

    forecast = _make_forecast([18.0, 20.0, 22.0, 24.0, 26.0])
    context = BusinessContext(
        current_inventory=60.0,
        reorder_point=90.0,
        lead_time_days=7,
        current_price=19.99,
        promotion_active=True,
    )
    explanation = _make_explanation({
        "promo_flag": 0.30,
        "lag_1": 0.20,
    })

    req1 = DecisionRequest(
        forecast_result=forecast,
        explanation_result=explanation,
        business_context=context,
    )
    req2 = DecisionRequest(
        forecast_result=forecast,
        explanation_result=explanation,
        business_context=context,
    )

    res1 = agent.run_decision(req1)
    res2 = agent.run_decision(req2)

    # Identical rules evaluated and triggered
    assert res1.rules_evaluated == res2.rules_evaluated
    assert res1.rules_triggered == res2.rules_triggered

    # Identical recommendation count
    assert len(res1.recommendations) == len(res2.recommendations)

    # Identical recommendation details
    for r1, r2 in zip(res1.recommendations, res2.recommendations):
        assert r1.action == r2.action
        assert r1.priority == r2.priority
        assert r1.category == r2.category
        assert r1.rule_id == r2.rule_id
        assert r1.recommendation_score == r2.recommendation_score
        assert r1.confidence == r2.confidence
        assert r1.requires_human_review == r2.requires_human_review
        assert r1.evidence == r2.evidence


# =============================================================================
# TEST CASE 9: RULE_7 Method-Aware Fidelity Audit (SHAP vs LIME vs Prophet)
# =============================================================================
def test_rule_7_method_distinction_shap_vs_lime():
    """Verify RULE_7 distinguishes SHAP mathematical reconstruction vs LIME surrogate R²."""
    agent = DecisionIntelligenceAgent()
    forecast = _make_forecast([50.0, 52.0, 54.0, 56.0, 58.0])
    context = BusinessContext(current_inventory=100.0, reorder_point=40.0)

    # 1. LightGBM with SHAP (99.99% fidelity)
    shap_expl = _make_explanation(
        {"promo_flag": 0.40, "price": -0.25},
        fidelity_score=0.9999,
        fidelity_status="HIGH_FIDELITY",
        method="shap",
    )
    res_shap = agent.run_decision(DecisionRequest(
        forecast_result=forecast,
        explanation_result=shap_expl,
        business_context=context,
    ))
    assert "RULE_7_LOW_EXPLANATION_FIDELITY" not in res_shap.rules_triggered

    # 2. LightGBM with LIME (surrogate R² = 0.204)
    lime_expl = _make_explanation(
        {"promo_flag": 0.40, "price": -0.25},
        fidelity_score=0.2040,
        fidelity_status="APPROXIMATE",
        method="lime",
    )
    res_lime = agent.run_decision(DecisionRequest(
        forecast_result=forecast,
        explanation_result=lime_expl,
        business_context=context,
        fidelity_threshold=0.65,
    ))
    assert "RULE_7_LOW_EXPLANATION_FIDELITY" in res_lime.rules_triggered
    rec_lime_7 = next(r for r in res_lime.recommendations if r.rule_id == "RULE_7_LOW_EXPLANATION_FIDELITY")
    assert "surrogate" in rec_lime_7.rationale.lower()
    assert "R²" in rec_lime_7.rationale or "r²" in rec_lime_7.rationale.lower()
    assert rec_lime_7.requires_human_review is True

    # 3. LIME with relaxed fidelity threshold (threshold = 0.15) - confirms APPROXIMATE is not blanket failure
    res_lime_relaxed = agent.run_decision(DecisionRequest(
        forecast_result=forecast,
        explanation_result=lime_expl,
        business_context=context,
        fidelity_threshold=0.15,
    ))
    assert "RULE_7_LOW_EXPLANATION_FIDELITY" not in res_lime_relaxed.rules_triggered

    # 4. SHAP with low fidelity (breakdown of efficiency axiom)
    shap_low = _make_explanation(
        {"promo_flag": 0.40, "price": -0.25},
        fidelity_score=0.72,
        fidelity_status="APPROXIMATE",
        method="shap",
    )
    res_shap_low = agent.run_decision(DecisionRequest(
        forecast_result=forecast,
        explanation_result=shap_low,
        business_context=context,
        fidelity_threshold=0.65,
    ))
    assert "RULE_7_LOW_EXPLANATION_FIDELITY" in res_shap_low.rules_triggered
    rec_shap_7 = next(r for r in res_shap_low.recommendations if r.rule_id == "RULE_7_LOW_EXPLANATION_FIDELITY")
    assert "additive reconstruction" in rec_shap_7.rationale.lower()
    assert "efficiency axiom" in rec_shap_7.rationale.lower()

    # 5. Prophet exact component decomposition
    prophet_expl = _make_explanation(
        {"trend": 45.0, "weekly": 5.0},
        fidelity_score=1.0,
        fidelity_status="DECOMPOSED",
        method="component_based",
    )
    res_prophet = agent.run_decision(DecisionRequest(
        forecast_result=forecast,
        explanation_result=prophet_expl,
        business_context=context,
    ))
    assert "RULE_7_LOW_EXPLANATION_FIDELITY" not in res_prophet.rules_triggered

