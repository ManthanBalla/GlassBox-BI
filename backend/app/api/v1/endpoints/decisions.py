"""Decision Intelligence API Endpoints for GlassBox-BI (Phase 7).

Exposes REST endpoints for deterministic business recommendations,
what-if scenario simulations, rule catalogs, and operational health checks.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status

from backend.app.decision_intelligence.agent import DecisionIntelligenceAgent
from backend.app.schemas.decisions import (
    DecisionConfigSchema,
    DecisionRequest,
    DecisionResult,
    ScenarioRequest,
    ScenarioResult,
)

router = APIRouter()
_agent = DecisionIntelligenceAgent()


@router.get(
    "/health",
    summary="Decision Intelligence Subsystem Health",
    response_model=Dict[str, Any],
)
def get_decisions_health() -> Dict[str, Any]:
    """Returns the operational status of the Decision Intelligence Agent."""
    return {
        "status": "HEALTHY",
        "module": "Phase 7 — Decision Intelligence Agent",
        "engine": _agent.engine_name,
        "supported_domains": _agent.supported_domains,
        "mode": "deterministic_rule_engine",
        "llm_dependency": False,
        "autonomous_execution": False,
        "supported_categories": [
            "INVENTORY",
            "PRICING",
            "PROMOTION",
            "MONITORING",
            "RISK",
            "PLANNING",
        ],
        "supported_priorities": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    }


@router.get(
    "/rules",
    summary="Retail Decision Rule Catalog",
    response_model=List[Dict[str, Any]],
)
def get_decision_rules() -> List[Dict[str, Any]]:
    """Returns the deterministic catalog of business rules, conditions, and priorities."""
    return [
        {
            "rule_id": "RULE_1_HIGH_DEMAND_LOW_INVENTORY",
            "name": "High Demand + Low Inventory Replenishment",
            "category": "INVENTORY",
            "default_priority": "HIGH",
            "trigger_condition": "Forecast demand trending upward AND current inventory <= reorder threshold.",
            "recommended_action": "Increase replenishment quantity",
            "risk_addressed": "Stockout risk during replenishment cycle",
            "requires_human_review": False,
        },
        {
            "rule_id": "RULE_2_HIGH_STOCKOUT_CRITICAL",
            "name": "Critical Stockout Threat",
            "category": "INVENTORY",
            "default_priority": "CRITICAL",
            "trigger_condition": "Days of inventory supply < supplier fulfillment lead time OR inventory < safety stock.",
            "recommended_action": "Prioritize emergency inventory replenishment",
            "risk_addressed": "Imminent stockout and service level breach",
            "requires_human_review": True,
        },
        {
            "rule_id": "RULE_3_LOW_DEMAND_HIGH_INVENTORY",
            "name": "Low Demand + Excess Inventory Curtailment",
            "category": "INVENTORY",
            "default_priority": "MEDIUM",
            "trigger_condition": "Forecast demand declining AND inventory coverage > 2.5x lead time.",
            "recommended_action": "Reduce replenishment order quantity / consider clearance planning",
            "risk_addressed": "Excess inventory holding costs, capital lockup, and product obsolescence",
            "requires_human_review": False,
        },
        {
            "rule_id": "RULE_4_DEMAND_INCREASE_PROMOTION",
            "name": "Promotion-Driven Demand Monitoring",
            "category": "PROMOTION",
            "default_priority": "MEDIUM",
            "trigger_condition": "Forecast demand increasing AND promotion active (or supported by XAI promo attribution).",
            "recommended_action": "Maintain promotional strategy while monitoring inventory burn rate",
            "risk_addressed": "Promotion-induced inventory depletion and margin erosion",
            "requires_human_review": False,
        },
        {
            "rule_id": "RULE_5_PRICE_DRIVEN_DEMAND_CHANGE",
            "name": "Price-Driven Demand Review",
            "category": "PRICING",
            "default_priority": "HIGH",
            "trigger_condition": "Price is identified as a major Phase 6 explanation driver AND demand changes significantly.",
            "recommended_action": "Review pricing strategy and elasticity",
            "risk_addressed": "Suboptimal price elasticity impacting margins or customer demand",
            "requires_human_review": True,
        },
        {
            "rule_id": "RULE_6_HIGH_FORECAST_UNCERTAINTY",
            "name": "High Forecast Uncertainty Buffering",
            "category": "RISK",
            "default_priority": "HIGH",
            "trigger_condition": "Forecast prediction interval relative width > configured threshold (e.g. 40%).",
            "recommended_action": "Adopt conservative inventory planning with human supervisory review",
            "risk_addressed": "Forecast variance creates asymmetrical stockout or excess inventory exposure",
            "requires_human_review": True,
        },
        {
            "rule_id": "RULE_7_LOW_EXPLANATION_FIDELITY",
            "name": "Low Explanation Fidelity Warning",
            "category": "MONITORING",
            "default_priority": "MEDIUM",
            "trigger_condition": "Phase 6 explanation fidelity < 65% or explanation status is APPROXIMATE.",
            "recommended_action": "Flag decision for human supervisory review due to low explanation fidelity",
            "risk_addressed": "Unreliable feature attribution may mask true operational demand drivers",
            "requires_human_review": True,
        },
        {
            "rule_id": "RULE_MISSING_INVENTORY_PLANNING",
            "name": "Missing Inventory Context Safety Guard",
            "category": "PLANNING",
            "default_priority": "HIGH",
            "trigger_condition": "Current inventory is null/unprovided in BusinessContext (zero value fabrication).",
            "recommended_action": "Conduct comprehensive inventory planning review to audit stock levels",
            "risk_addressed": "Unverified stockout exposure due to lack of on-hand inventory visibility",
            "requires_human_review": True,
        },
    ]


@router.get(
    "/config",
    summary="Decision Intelligence Configuration Parameters",
    response_model=DecisionConfigSchema,
)
def get_decision_config() -> DecisionConfigSchema:
    """Returns default thresholds, holding rates, and coverage multipliers."""
    return _agent.config


@router.get(
    "/sample",
    summary="Sample Decision Intelligence Run",
    response_model=DecisionResult,
)
def get_sample_decision() -> DecisionResult:
    """Returns an instantaneous sample decision with realistic retail forecast and context for UI demonstration."""
    try:
        return _agent.get_sample_decision()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate sample decision: {str(exc)}",
        )


@router.post(
    "/run",
    summary="Execute Decision Intelligence Analysis",
    response_model=DecisionResult,
)
def run_decisions(request: DecisionRequest) -> DecisionResult:
    """Evaluates forecast, uncertainty, explanation evidence, and business context into actionable recommendations."""
    try:
        result = _agent.run_decision(request)
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decision Intelligence execution error: {str(exc)}",
        )


@router.post(
    "/scenario",
    summary="Simulate What-If Scenario",
    response_model=ScenarioResult,
)
def run_decision_scenario(request: ScenarioRequest) -> ScenarioResult:
    """Performs lightweight deterministic what-if simulation (inventory shift, price elasticity, promo toggle)."""
    try:
        result = _agent.run_scenario(request)
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scenario simulation error: {str(exc)}",
        )
