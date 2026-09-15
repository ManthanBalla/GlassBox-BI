"""Decision Intelligence Agent for GlassBox-BI (Phase 7).

Main orchestration agent for prescriptive business recommendations, evidence synthesis,
scenario simulations, and reproducibility audit logging.
Operates 100% deterministically with zero LLM dependency and zero autonomous execution.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.app.schemas.decisions import (
    BusinessContext,
    DecisionAuditRecord,
    DecisionConfigSchema,
    DecisionPriority,
    DecisionRequest,
    DecisionResult,
    RecommendationItem,
    ScenarioRequest,
    ScenarioResult,
)
from backend.app.schemas.explainability import ExplanationResult
from backend.app.schemas.forecasting import ForecastPoint, ForecastResult
from backend.app.decision_intelligence.base import BaseDecisionEngine
from backend.app.decision_intelligence.rules import RetailDecisionRuleEngine
from backend.app.decision_intelligence.scoring import DecisionScorer
from backend.app.decision_intelligence.validation import DecisionValidator


class DecisionIntelligenceAgent(BaseDecisionEngine):
    """Orchestrates deterministic decision rule evaluation, scoring, and scenario analysis."""

    def __init__(self, config: Optional[DecisionConfigSchema] = None) -> None:
        super().__init__(config=config)
        self.rule_engine = RetailDecisionRuleEngine(config=self.config)

    @property
    def engine_name(self) -> str:
        return "GlassBox_Deterministic_Retail_Decision_Engine_v1"

    @property
    def supported_domains(self) -> List[str]:
        return ["retail", "sme_finance"]

    def evaluate_business_context(
        self,
        context: Optional[BusinessContext],
        forecast_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarizes inventory coverage, stockout risk, and operational parameters."""
        if context is None:
            return {"status": "MISSING_CONTEXT", "has_context": False}

        mean_demand = max(0.1, forecast_summary.get("mean_demand", 1.0))
        inv = context.current_inventory
        lead_time = context.lead_time_days or self.config.default_lead_time_days

        days_coverage = round(inv / mean_demand, 1) if inv is not None else None
        coverage_status = "UNKNOWN"
        if days_coverage is not None:
            if days_coverage < lead_time:
                coverage_status = "CRITICAL_DEFICIT"
            elif days_coverage < (lead_time * 1.5):
                coverage_status = "REORDER_WINDOW"
            elif days_coverage > (lead_time * self.config.overstock_coverage_multiplier):
                coverage_status = "OVERSTOCK"
            else:
                coverage_status = "BALANCED"

        return {
            "has_context": True,
            "current_inventory": inv,
            "reorder_point": context.reorder_point,
            "safety_stock": context.safety_stock,
            "lead_time_days": lead_time,
            "days_of_supply": days_coverage,
            "coverage_status": coverage_status,
            "current_price": context.current_price,
            "promotion_active": context.promotion_active,
            "business_type": context.business_type,
        }

    def generate_recommendations(
        self,
        forecast: ForecastResult,
        explanation: Optional[ExplanationResult],
        context: Optional[BusinessContext],
        config: Optional[DecisionConfigSchema] = None,
    ) -> List[RecommendationItem]:
        """Generates ranked prescriptive recommendations from the retail rule engine."""
        recs, _, _, _, _ = self.rule_engine.evaluate_all_rules(
            forecast=forecast,
            explanation=explanation,
            context=context,
        )
        return recs

    def score_recommendation(
        self,
        recommendation: RecommendationItem,
        context: Optional[BusinessContext],
        forecast: ForecastResult,
    ) -> float:
        """Calculates rule-based recommendation strength index (0-100)."""
        return recommendation.recommendation_score

    def validate_decision(self, result: DecisionResult) -> List[str]:
        """Validates DecisionResult against structural and safety contracts."""
        return DecisionValidator.validate_decision_result(result)

    def run_decision(self, request: DecisionRequest) -> DecisionResult:
        """Executes full deterministic decision intelligence pipeline."""
        start_time = time.perf_counter()
        now = datetime.now(timezone.utc)
        decision_id = f"dec_{now.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

        # 1. Resolve or Fallback Forecast
        forecast = request.forecast_result
        if forecast is None:
            forecast = self._build_synthetic_forecast(
                entity_id=request.entity_id or "STORE_001",
                product_id=request.product_id or "PROD_001",
            )

        # 2. Validate Inputs
        input_issues = DecisionValidator.validate_inputs(
            forecast=forecast,
            context=request.business_context,
            explanation=request.explanation_result,
        )
        if input_issues:
            raise ValueError(f"Decision input validation failed: {'; '.join(input_issues)}")

        # 3. Forecast Dynamics & Context Summaries
        dynamics = DecisionScorer.calculate_forecast_dynamics(forecast)
        ctx_summary = self.evaluate_business_context(request.business_context, dynamics)

        # 4. Evaluate Deterministic Rules
        (
            recommendations,
            rules_eval,
            rules_trig,
            assumptions,
            rule_warnings,
        ) = self.rule_engine.evaluate_all_rules(
            forecast=forecast,
            explanation=request.explanation_result,
            context=request.business_context,
            confidence_threshold=request.confidence_threshold,
            uncertainty_threshold=request.uncertainty_threshold,
            fidelity_threshold=request.fidelity_threshold,
        )

        primary_rec = recommendations[0] if recommendations else None

        # 5. Aggregate Warnings & Human Review Status
        all_warnings = list(rule_warnings)
        if request.business_context is None:
            all_warnings.append("Operating without business context. Recommendations represent planning guidelines.")
        requires_review = any(r.requires_human_review for r in recommendations) or (
            primary_rec is not None and primary_rec.confidence < request.confidence_threshold
        )

        # 6. Explanation Summary
        expl_summary: Dict[str, Any] = {"available": False}
        if request.explanation_result is not None:
            expl = request.explanation_result
            expl_summary = {
                "available": True,
                "method": expl.method,
                "fidelity_score": expl.fidelity.fidelity_score if expl.fidelity else None,
                "explanation_status": expl.fidelity.explanation_status if expl.fidelity else None,
                "top_features": [f.feature for f in expl.features[:5]] if expl.features else [],
            }

        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # 7. Assemble Audit Record
        audit_trail = DecisionAuditRecord(
            decision_id=decision_id,
            timestamp=now,
            entity_id=request.entity_id,
            product_id=request.product_id,
            model_name=forecast.model_name,
            rules_evaluated=rules_eval,
            rules_triggered=rules_trig,
            evidence_used=primary_rec.evidence if primary_rec else [],
            primary_rule_id=primary_rec.rule_id if primary_rec else None,
            decision_confidence=primary_rec.confidence if primary_rec else 0.5,
            recommendation_score=primary_rec.recommendation_score if primary_rec else 50.0,
            warnings=all_warnings,
            requires_human_review=requires_review,
            execution_duration_ms=duration_ms,
        )

        # 8. Assemble Complete Decision Result
        result = DecisionResult(
            decision_id=decision_id,
            entity_id=request.entity_id,
            product_id=request.product_id,
            forecast_id=forecast.forecast_id,
            primary_recommendation=primary_rec,
            recommendations=recommendations,
            context_summary=ctx_summary,
            forecast_summary=dynamics,
            explanation_summary=expl_summary,
            rules_evaluated=rules_eval,
            rules_triggered=rules_trig,
            audit_trail=audit_trail,
            warnings=all_warnings,
            requires_human_review=requires_review,
            created_at=now,
        )

        # 9. Invariant Validation
        validation_issues = self.validate_decision(result)
        if validation_issues:
            raise RuntimeError(f"Decision output contract violation: {'; '.join(validation_issues)}")

        return result

    def run_scenario(self, request: ScenarioRequest) -> ScenarioResult:
        """Executes lightweight deterministic what-if scenario simulation without model retraining."""
        scenario_id = f"scen_{uuid4().hex[:8]}"
        base_decision = self.run_decision(request.base_request)

        # Clone and modify business context
        base_ctx = request.base_request.business_context
        simulated_ctx = BusinessContext() if base_ctx is None else base_ctx.model_copy(deep=True)

        delta_summary: Dict[str, Any] = {
            "scenario_name": request.scenario_name,
            "modifications": {},
        }

        # Apply Inventory Delta
        if request.inventory_delta is not None and simulated_ctx.current_inventory is not None:
            old_inv = simulated_ctx.current_inventory
            new_inv = max(0.0, old_inv + request.inventory_delta)
            simulated_ctx.current_inventory = new_inv
            delta_summary["modifications"]["inventory"] = {
                "before": old_inv,
                "after": new_inv,
                "delta": request.inventory_delta,
            }

        # Apply Promotion Override
        if request.promotion_override is not None:
            old_promo = simulated_ctx.promotion_active
            simulated_ctx.promotion_active = request.promotion_override
            delta_summary["modifications"]["promotion_active"] = {
                "before": old_promo,
                "after": request.promotion_override,
            }

        # Apply Lead Time Override
        if request.lead_time_override is not None:
            old_lt = simulated_ctx.lead_time_days
            simulated_ctx.lead_time_days = request.lead_time_override
            delta_summary["modifications"]["lead_time_days"] = {
                "before": old_lt,
                "after": request.lead_time_override,
            }

        # Simulate Demand Response if price changed (lightweight elasticity approximation: e = -1.5)
        simulated_forecast = request.base_request.forecast_result
        if request.price_delta_percent is not None and simulated_forecast is not None:
            elasticity = -1.5  # Standard retail price elasticity estimate
            pct_change = request.price_delta_percent / 100.0
            demand_multiplier = max(0.2, 1.0 + (elasticity * pct_change))

            mod_points = []
            for p in (simulated_forecast.predictions or simulated_forecast.forecast_points):
                adj_pred = round(p.prediction * demand_multiplier, 2)
                adj_lower = round(p.lower_bound * demand_multiplier, 2) if p.lower_bound else None
                adj_upper = round(p.upper_bound * demand_multiplier, 2) if p.upper_bound else None
                mod_points.append(
                    ForecastPoint(
                        date=p.date,
                        prediction=adj_pred,
                        lower_bound=adj_lower,
                        upper_bound=adj_upper,
                    )
                )

            simulated_forecast = simulated_forecast.model_copy(deep=True)
            simulated_forecast.predictions = mod_points
            simulated_forecast.forecast_points = mod_points
            delta_summary["modifications"]["price_elasticity"] = {
                "price_delta_percent": request.price_delta_percent,
                "assumed_elasticity": elasticity,
                "demand_multiplier": round(demand_multiplier, 3),
            }

        # Run Scenario Decision
        scenario_req = request.base_request.model_copy(deep=True)
        scenario_req.business_context = simulated_ctx
        scenario_req.forecast_result = simulated_forecast
        scenario_decision = self.run_decision(scenario_req)

        # Delta Summary between Baseline and Scenario
        b_rec = base_decision.primary_recommendation
        s_rec = scenario_decision.primary_recommendation

        delta_summary["primary_action_change"] = {
            "baseline_action": b_rec.action if b_rec else "None",
            "scenario_action": s_rec.action if s_rec else "None",
            "baseline_priority": b_rec.priority if b_rec else "None",
            "scenario_priority": s_rec.priority if s_rec else "None",
            "baseline_score": b_rec.recommendation_score if b_rec else 0.0,
            "scenario_score": s_rec.recommendation_score if s_rec else 0.0,
        }

        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_name=request.scenario_name,
            baseline_decision=base_decision,
            scenario_decision=scenario_decision,
            delta_summary=delta_summary,
            created_at=datetime.now(timezone.utc),
        )

    def get_sample_decision(self) -> DecisionResult:
        """Returns a fast sample decision result for UI testing and demonstration."""
        sample_forecast = self._build_synthetic_forecast(
            entity_id="STORE_001",
            product_id="PROD_001",
        )
        sample_context = BusinessContext(
            current_inventory=85.0,
            reorder_point=140.0,
            safety_stock=40.0,
            lead_time_days=7,
            current_price=24.99,
            promotion_active=True,
            unit_cost=14.50,
            holding_cost_rate=0.15,
            business_type="retail",
            entity_id="STORE_001",
            product_id="PROD_001",
        )
        sample_request = DecisionRequest(
            entity_id="STORE_001",
            product_id="PROD_001",
            forecast_result=sample_forecast,
            business_context=sample_context,
        )
        return self.run_decision(sample_request)

    # -------------------------------------------------------------------------
    # Helper: Synthetic Forecast Builder for Standalone Testing
    # -------------------------------------------------------------------------
    def _build_synthetic_forecast(self, entity_id: str, product_id: str) -> ForecastResult:
        """Constructs a deterministic 14-day sample forecast with prediction intervals."""
        from datetime import date, timedelta

        start = date(2024, 8, 1)
        base_demand = 18.0
        points = []
        for i in range(14):
            dt_str = (start + timedelta(days=i)).isoformat()
            # Increasing trend
            pred = round(base_demand + (i * 0.45) + (1.2 if i % 7 in [5, 6] else -0.5), 2)
            lower = max(0.0, round(pred - 2.8, 2))
            upper = round(pred + 3.2, 2)
            points.append(
                ForecastPoint(
                    date=dt_str,
                    prediction=pred,
                    lower_bound=lower,
                    upper_bound=upper,
                )
            )

        return ForecastResult(
            forecast_id=f"fcst_{uuid4().hex[:8]}",
            dataset_id="retail_dataset",
            entity_id=entity_id,
            product_id=product_id,
            horizon=14,
            frequency="D",
            model_name="lightgbm",
            status="SUCCESS",
            predictions=points,
            forecast_points=points,
            uncertainty_method="quantile_regression_80pct",
            confidence_level=0.80,
            metrics={"MAE": 1.45, "RMSE": 1.88, "MAPE": 8.5},
        )
