"""Deterministic Retail Decision Rule Engine for GlassBox-BI (Phase 7).

Implements transparent, auditable business rules connecting:
- Forecast dynamics (trend, total demand, peak)
- Forecast uncertainty (prediction interval width)
- Phase 6 Explainability evidence (SHAP contributions, LIME R², GAM decomposition)
- Operational business context (inventory, reorder point, safety stock, pricing, promo)

Rules are 100% deterministic, reproducible, and explainable with zero LLM dependency.
"""

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from backend.app.schemas.decisions import (
    BusinessActionCategory,
    BusinessContext,
    DecisionConfigSchema,
    DecisionPriority,
    RecommendationItem,
    TradeOff,
)
from backend.app.schemas.explainability import ExplanationResult
from backend.app.schemas.forecasting import ForecastResult
from backend.app.decision_intelligence.scoring import DecisionScorer


class RetailDecisionRuleEngine:
    """Evaluates deterministic business rules across forecast, uncertainty, XAI, and context."""

    def __init__(self, config: Optional[DecisionConfigSchema] = None) -> None:
        self.config = config or DecisionConfigSchema()

    def evaluate_all_rules(
        self,
        forecast: ForecastResult,
        explanation: Optional[ExplanationResult],
        context: Optional[BusinessContext],
        confidence_threshold: float = 0.65,
        uncertainty_threshold: float = 0.40,
        fidelity_threshold: float = 0.65,
    ) -> Tuple[List[RecommendationItem], List[str], List[str], List[str], List[str]]:
        """Evaluates all candidate rules and returns triggered recommendations.
        
        Returns:
            Tuple of (recommendations, rules_evaluated, rules_triggered, assumptions, warnings)
        """
        dynamics = DecisionScorer.calculate_forecast_dynamics(forecast)
        confidence, assumptions, warnings = DecisionScorer.compute_decision_confidence(
            forecast=forecast,
            explanation=explanation,
            context=context,
            uncertainty_threshold=uncertainty_threshold,
            fidelity_threshold=fidelity_threshold,
        )

        # Extract explanation insights
        xai_insights = self._extract_explanation_insights(explanation)

        rules_evaluated: List[str] = []
        rules_triggered: List[str] = []
        recommendations: List[RecommendationItem] = []

        # ---------------------------------------------------------------------
        # RULE 8 / GUARD: Missing Inventory Context (Evaluated First for Safety)
        # ---------------------------------------------------------------------
        rules_evaluated.append("RULE_MISSING_INVENTORY_PLANNING")
        if context is None or context.current_inventory is None:
            rules_triggered.append("RULE_MISSING_INVENTORY_PLANNING")
            rec = self._build_rule_missing_inventory(
                forecast=forecast,
                dynamics=dynamics,
                xai_insights=xai_insights,
                confidence=confidence,
                assumptions=assumptions,
                warnings=warnings,
            )
            recommendations.append(rec)
            # When inventory is completely missing, we also check pricing, uncertainty, and fidelity
            # but do not execute stock-level replenishment rules with fabricated numbers.
        else:
            # -----------------------------------------------------------------
            # RULE 2: Critical Stockout Risk
            # -----------------------------------------------------------------
            rules_evaluated.append("RULE_2_HIGH_STOCKOUT_CRITICAL")
            rule_2_met, rec_2 = self._check_rule_2_stockout(
                forecast=forecast,
                dynamics=dynamics,
                context=context,
                xai_insights=xai_insights,
                confidence=confidence,
                warnings=warnings,
            )
            if rule_2_met and rec_2 is not None:
                rules_triggered.append("RULE_2_HIGH_STOCKOUT_CRITICAL")
                recommendations.append(rec_2)

            # -----------------------------------------------------------------
            # RULE 1: High Demand + Low Inventory (Replenishment Scale-Up)
            # -----------------------------------------------------------------
            rules_evaluated.append("RULE_1_HIGH_DEMAND_LOW_INVENTORY")
            rule_1_met, rec_1 = self._check_rule_1_replenish(
                forecast=forecast,
                dynamics=dynamics,
                context=context,
                xai_insights=xai_insights,
                confidence=confidence,
                warnings=warnings,
            )
            if rule_1_met and rec_1 is not None:
                rules_triggered.append("RULE_1_HIGH_DEMAND_LOW_INVENTORY")
                recommendations.append(rec_1)

            # -----------------------------------------------------------------
            # RULE 3: Low Demand + High Inventory (Curtailment / Clearance)
            # -----------------------------------------------------------------
            rules_evaluated.append("RULE_3_LOW_DEMAND_HIGH_INVENTORY")
            rule_3_met, rec_3 = self._check_rule_3_overstock(
                forecast=forecast,
                dynamics=dynamics,
                context=context,
                xai_insights=xai_insights,
                confidence=confidence,
                warnings=warnings,
            )
            if rule_3_met and rec_3 is not None:
                rules_triggered.append("RULE_3_LOW_DEMAND_HIGH_INVENTORY")
                recommendations.append(rec_3)

        # ---------------------------------------------------------------------
        # RULE 4: Promotion-Driven Demand Monitoring
        # ---------------------------------------------------------------------
        rules_evaluated.append("RULE_4_DEMAND_INCREASE_PROMOTION")
        rule_4_met, rec_4 = self._check_rule_4_promotion(
            forecast=forecast,
            dynamics=dynamics,
            context=context,
            xai_insights=xai_insights,
            confidence=confidence,
            warnings=warnings,
        )
        if rule_4_met and rec_4 is not None:
            rules_triggered.append("RULE_4_DEMAND_INCREASE_PROMOTION")
            recommendations.append(rec_4)

        # ---------------------------------------------------------------------
        # RULE 5: Price-Driven Demand Review
        # ---------------------------------------------------------------------
        rules_evaluated.append("RULE_5_PRICE_DRIVEN_DEMAND_CHANGE")
        rule_5_met, rec_5 = self._check_rule_5_pricing(
            forecast=forecast,
            dynamics=dynamics,
            context=context,
            xai_insights=xai_insights,
            confidence=confidence,
            warnings=warnings,
        )
        if rule_5_met and rec_5 is not None:
            rules_triggered.append("RULE_5_PRICE_DRIVEN_DEMAND_CHANGE")
            recommendations.append(rec_5)

        # ---------------------------------------------------------------------
        # RULE 6: High Forecast Uncertainty Guard
        # ---------------------------------------------------------------------
        rules_evaluated.append("RULE_6_HIGH_FORECAST_UNCERTAINTY")
        rule_6_met, rec_6 = self._check_rule_6_uncertainty(
            forecast=forecast,
            dynamics=dynamics,
            confidence=confidence,
            uncertainty_threshold=uncertainty_threshold,
            warnings=warnings,
        )
        if rule_6_met and rec_6 is not None:
            rules_triggered.append("RULE_6_HIGH_FORECAST_UNCERTAINTY")
            recommendations.append(rec_6)

        # ---------------------------------------------------------------------
        # RULE 7: Low Explanation Fidelity Guard
        # ---------------------------------------------------------------------
        rules_evaluated.append("RULE_7_LOW_EXPLANATION_FIDELITY")
        rule_7_met, rec_7 = self._check_rule_7_fidelity(
            explanation=explanation,
            confidence=confidence,
            fidelity_threshold=fidelity_threshold,
            warnings=warnings,
        )
        if rule_7_met and rec_7 is not None:
            rules_triggered.append("RULE_7_LOW_EXPLANATION_FIDELITY")
            recommendations.append(rec_7)

        # Fallback if no specific rule triggered
        if not recommendations:
            rules_evaluated.append("RULE_STEADY_STATE_MONITORING")
            rules_triggered.append("RULE_STEADY_STATE_MONITORING")
            rec_fallback = self._build_steady_state_recommendation(
                forecast=forecast,
                dynamics=dynamics,
                context=context,
                confidence=confidence,
            )
            recommendations.append(rec_fallback)

        # Sort recommendations by priority and score
        priority_weights = {
            DecisionPriority.CRITICAL: 4,
            DecisionPriority.HIGH: 3,
            DecisionPriority.MEDIUM: 2,
            DecisionPriority.LOW: 1,
        }
        recommendations.sort(
            key=lambda r: (priority_weights.get(r.priority, 0), r.recommendation_score),
            reverse=True,
        )

        # Enforce safety review invariant: if confidence < confidence_threshold or CRITICAL, mark requires_human_review = True
        for r in recommendations:
            if confidence < confidence_threshold or r.priority == DecisionPriority.CRITICAL:
                r.requires_human_review = True

        return recommendations, rules_evaluated, rules_triggered, assumptions, warnings

    # =========================================================================
    # Rule Evaluators
    # =========================================================================

    def _check_rule_1_replenish(
        self,
        forecast: ForecastResult,
        dynamics: Dict[str, Any],
        context: BusinessContext,
        xai_insights: Dict[str, Any],
        confidence: float,
        warnings: List[str],
    ) -> Tuple[bool, Optional[RecommendationItem]]:
        """Rule 1: High Demand + Low Inventory -> Increase Replenishment."""
        inv = context.current_inventory
        if inv is None:
            return False, None

        total_demand = dynamics["total_demand"]
        reorder = context.reorder_point if context.reorder_point is not None else total_demand

        # Triggered if demand is increasing or total demand exceeds current inventory, and inventory <= reorder point
        is_increasing = dynamics["trend_direction"] == "increasing" or dynamics["trend_slope"] > 0.03
        is_low_stock = inv <= reorder or inv < total_demand

        if not (is_increasing and is_low_stock):
            return False, None

        # Determine priority: if inventory is below safety stock or coverage < 4 days, mark HIGH or CRITICAL
        lead_time = context.lead_time_days or self.config.default_lead_time_days
        mean_demand = max(1.0, dynamics["mean_demand"])
        days_coverage = inv / mean_demand
        priority = DecisionPriority.CRITICAL if days_coverage < 3.0 else DecisionPriority.HIGH

        # Evidence
        evidence = [
            f"Forecasted demand is trending upward (slope: +{round(dynamics['trend_slope'] * 100, 1)}%) with total expected demand of {total_demand} units.",
            f"Current physical inventory ({inv} units) is below the configured reorder threshold ({reorder} units).",
            f"Current stock coverage is {round(days_coverage, 1)} days against a fulfillment lead time of {lead_time} days.",
        ]
        if xai_insights["top_positive_features"]:
            feat_names = ", ".join(xai_insights["top_positive_features"][:3])
            evidence.append(f"Phase 6 feature attributions confirm demand support driven by: {feat_names}.")

        deficit = max(0.0, total_demand - inv)
        target_reorder_qty = round(deficit + (context.safety_stock or (mean_demand * 3.0)), 0)
        if context.minimum_order_quantity and target_reorder_qty < context.minimum_order_quantity:
            target_reorder_qty = context.minimum_order_quantity
            evidence.append(f"Order adjusted upward to satisfy supplier MOQ ({context.minimum_order_quantity} units).")

        rationale = (
            f"Projected demand is expanding while on-hand stock ({inv} units) is insufficient to cover the "
            f"forecast horizon without entering stockout. Accelerating replenishment by approximately "
            f"{int(target_reorder_qty)} units restores the required safety buffer."
        )

        trade_offs = TradeOff(
            benefit=f"Prevents an estimated stockout of up to {round(deficit, 0)} units during peak demand.",
            trade_off="Increases near-term inventory carrying cost and working capital commitment.",
            quantified_impact=f"Estimated inventory carrying cost: +${round(target_reorder_qty * (context.unit_cost or 10.0) * (context.holding_cost_rate or 0.15) / 12, 2)}/mo.",
        )

        score = DecisionScorer.compute_recommendation_score(
            priority=priority,
            rule_severity=0.85,
            demand_gap_factor=min(1.0, deficit / max(1.0, total_demand)),
            explanation_support_score=0.80 if xai_insights["demand_momentum_supported"] else 0.50,
            confidence=confidence,
            has_missing_context=False,
        )

        rec = RecommendationItem(
            recommendation_id=f"rec_{uuid4().hex[:8]}",
            action=f"Increase replenishment quantity by {int(target_reorder_qty)} units",
            category=BusinessActionCategory.INVENTORY,
            priority=priority,
            rationale=rationale,
            evidence=evidence,
            expected_impact=f"Restores days-of-supply to {round((inv + target_reorder_qty) / mean_demand, 1)} days and protects customer fulfillment.",
            risk="Stockout risk during replenishment cycle and lost sales revenue",
            confidence=confidence,
            recommendation_score=score,
            supporting_forecast=dynamics,
            supporting_features=xai_insights["top_positive_features"][:4],
            trade_offs=trade_offs,
            assumptions=[],
            warnings=[],
            requires_human_review=(priority == DecisionPriority.CRITICAL or confidence < 0.65),
            rule_id="RULE_1_HIGH_DEMAND_LOW_INVENTORY",
        )
        return True, rec

    def _check_rule_2_stockout(
        self,
        forecast: ForecastResult,
        dynamics: Dict[str, Any],
        context: BusinessContext,
        xai_insights: Dict[str, Any],
        confidence: float,
        warnings: List[str],
    ) -> Tuple[bool, Optional[RecommendationItem]]:
        """Rule 2: High Stockout Risk -> Prioritize Emergency Replenishment."""
        inv = context.current_inventory
        if inv is None:
            return False, None

        mean_demand = max(0.1, dynamics["mean_demand"])
        lead_time = context.lead_time_days or self.config.default_lead_time_days
        days_coverage = inv / mean_demand

        # Triggered if coverage is strictly below lead time OR below critical threshold
        is_critical_stockout = (
            days_coverage < lead_time
            or (context.safety_stock is not None and inv < context.safety_stock)
            or days_coverage <= self.config.stockout_critical_coverage_days
        )

        if not is_critical_stockout:
            return False, None

        evidence = [
            f"Critical inventory deficit: current on-hand stock ({inv} units) represents only {round(days_coverage, 1)} days of supply.",
            f"Fulfillment lead time is {lead_time} days; a stockout will occur before standard replenishment arrives.",
            f"Daily projected demand burn rate is {round(mean_demand, 1)} units.",
        ]
        if context.safety_stock is not None and inv < context.safety_stock:
            evidence.append(f"On-hand inventory has breached safety stock threshold ({context.safety_stock} units).")

        deficit = max(0.0, (lead_time * mean_demand) - inv)
        target_emergency_qty = int(deficit + (context.safety_stock or (mean_demand * 2.0)))

        rationale = (
            f"Imminent stockout warning: current inventory coverage ({round(days_coverage, 1)} days) is strictly "
            f"less than supplier fulfillment lead time ({lead_time} days). Emergency replenishment prioritization "
            f"or expedited shipment of {target_emergency_qty} units is required to maintain service level continuity."
        )

        trade_offs = TradeOff(
            benefit=f"Mitigates immediate stockout and preserves customer service level target of {round((context.service_level_target or 0.95) * 100, 0)}%.",
            trade_off="Expedited procurement or split shipments may incur premium freight surcharges.",
            quantified_impact=f"Prevents estimated lost revenue of ${round(deficit * (context.current_price or 25.0), 2)}.",
        )

        score = DecisionScorer.compute_recommendation_score(
            priority=DecisionPriority.CRITICAL,
            rule_severity=0.95,
            demand_gap_factor=1.0,
            explanation_support_score=0.70,
            confidence=confidence,
            has_missing_context=False,
        )

        rec = RecommendationItem(
            recommendation_id=f"rec_{uuid4().hex[:8]}",
            action=f"Prioritize emergency inventory replenishment ({target_emergency_qty} units)",
            category=BusinessActionCategory.INVENTORY,
            priority=DecisionPriority.CRITICAL,
            rationale=rationale,
            evidence=evidence,
            expected_impact="Eliminates imminent stockout window and secures critical customer fulfillment.",
            risk="Potential stockout and operational disruption",
            confidence=confidence,
            recommendation_score=score,
            supporting_forecast=dynamics,
            supporting_features=xai_insights["top_positive_features"][:3],
            trade_offs=trade_offs,
            assumptions=[],
            warnings=["Urgent inventory coverage breach: stockout projected before standard lead-time fulfillment."],
            requires_human_review=True,
            rule_id="RULE_2_HIGH_STOCKOUT_CRITICAL",
        )
        return True, rec

    def _check_rule_3_overstock(
        self,
        forecast: ForecastResult,
        dynamics: Dict[str, Any],
        context: BusinessContext,
        xai_insights: Dict[str, Any],
        confidence: float,
        warnings: List[str],
    ) -> Tuple[bool, Optional[RecommendationItem]]:
        """Rule 3: Low Demand + High Inventory -> Reduce Replenishment / Clearance."""
        inv = context.current_inventory
        if inv is None:
            return False, None

        total_demand = dynamics["total_demand"]
        mean_demand = max(0.1, dynamics["mean_demand"])
        lead_time = context.lead_time_days or self.config.default_lead_time_days
        days_coverage = inv / mean_demand

        # Triggered if demand is declining or inventory is > 2.5x lead time or > 2x total demand
        is_overstock = (
            days_coverage > (lead_time * self.config.overstock_coverage_multiplier)
            or (dynamics["trend_direction"] == "decreasing" and inv > total_demand * 1.5)
            or (inv > total_demand * 2.2 and total_demand > 0)
        )

        if not is_overstock:
            return False, None

        excess_units = max(0.0, inv - (total_demand + (context.safety_stock or mean_demand * 3.0)))
        evidence = [
            f"Forecasted demand is subdued/declining ({dynamics['trend_direction']}, slope: {round(dynamics['trend_slope'] * 100, 1)}%).",
            f"Current physical inventory ({inv} units) represents {round(days_coverage, 1)} days of supply, exceeding lead-time coverage by {round(days_coverage / lead_time, 1)}x.",
            f"Excess inventory volume is estimated at {int(excess_units)} units above the normal planning horizon.",
        ]

        rationale = (
            f"Demand is projected to remain subdued while on-hand stock ({inv} units) represents excessive "
            f"days-of-supply ({round(days_coverage, 1)} days). Reducing scheduled replenishment orders or "
            f"planning promotional clearance prevents working capital stagnation."
        )

        trade_offs = TradeOff(
            benefit=f"Reduces inventory carrying cost by approximately ${round(excess_units * (context.unit_cost or 10.0) * (context.holding_cost_rate or 0.15) / 12, 2)}/mo.",
            trade_off="Curtailing replenishment reduces buffer capacity if demand abruptly rebounds.",
            quantified_impact="Frees working capital and prevents aged inventory markdowns.",
        )

        score = DecisionScorer.compute_recommendation_score(
            priority=DecisionPriority.MEDIUM,
            rule_severity=0.60,
            demand_gap_factor=min(1.0, excess_units / max(1.0, inv)),
            explanation_support_score=0.60,
            confidence=confidence,
            has_missing_context=False,
        )

        action_msg = "Reduce replenishment order quantity"
        if excess_units > total_demand:
            action_msg += " and consider promotional clearance planning"

        rec = RecommendationItem(
            recommendation_id=f"rec_{uuid4().hex[:8]}",
            action=action_msg,
            category=BusinessActionCategory.INVENTORY,
            priority=DecisionPriority.MEDIUM,
            rationale=rationale,
            evidence=evidence,
            expected_impact="Normalizes inventory coverage toward target lead-time levels and curbs holding expense.",
            risk="Inventory obsolescence, capital lockup, and excessive holding costs",
            confidence=confidence,
            recommendation_score=score,
            supporting_forecast=dynamics,
            supporting_features=xai_insights["top_negative_features"][:3],
            trade_offs=trade_offs,
            assumptions=[],
            warnings=[],
            requires_human_review=(confidence < 0.65),
            rule_id="RULE_3_LOW_DEMAND_HIGH_INVENTORY",
        )
        return True, rec

    def _check_rule_4_promotion(
        self,
        forecast: ForecastResult,
        dynamics: Dict[str, Any],
        context: Optional[BusinessContext],
        xai_insights: Dict[str, Any],
        confidence: float,
        warnings: List[str],
    ) -> Tuple[bool, Optional[RecommendationItem]]:
        """Rule 4: Demand Increase + Active Promotion -> Maintain Strategy & Monitor Inventory."""
        promo_active = context.promotion_active if context is not None and context.promotion_active is not None else False
        promo_xai = xai_insights["promo_contribution"] > 0.05

        # Triggered if promotion is active in context OR strongly driving demand in XAI
        if not (promo_active or promo_xai):
            return False, None

        if dynamics["trend_direction"] != "increasing" and dynamics["trend_slope"] <= 0.0:
            return False, None

        evidence = [
            f"Forecasted demand is expanding (+{round(dynamics['trend_slope'] * 100, 1)}%) during promotional period.",
            f"Promotional campaign is verified active (Context: {promo_active}, XAI Attribution: +{round(xai_insights['promo_contribution'], 3)}).",
        ]
        if context is not None and context.current_inventory is not None:
            mean_demand = max(0.1, dynamics["mean_demand"])
            days_coverage = context.current_inventory / mean_demand
            evidence.append(f"Current inventory provides {round(days_coverage, 1)} days of coverage during promotional surge.")
            risk_level = "High inventory burn rate" if days_coverage < 7.0 else "Normal promotional velocity"
        else:
            evidence.append("Inventory level is unconfigured; promotional inventory burn cannot be verified.")
            risk_level = "Unverified promotional inventory exhaustion"

        rationale = (
            "Promotional campaign is actively stimulating demand as confirmed by forecast dynamics and "
            "Phase 6 feature attribution. Maintaining the promotion is recommended while establishing daily "
            "inventory burn rate tracking to prevent premature stockout."
        )

        trade_offs = TradeOff(
            benefit="Maximizes top-line promotional revenue lift and customer acquisition volume.",
            trade_off="Accelerates stock exhaustion; requires stringent supply monitoring to avoid stockout.",
            quantified_impact="Captures demand expansion supported by positive promotional coefficient.",
        )

        score = DecisionScorer.compute_recommendation_score(
            priority=DecisionPriority.MEDIUM,
            rule_severity=0.65,
            demand_gap_factor=0.5,
            explanation_support_score=0.90 if promo_xai else 0.60,
            confidence=confidence,
            has_missing_context=(context is None or context.current_inventory is None),
        )

        rec = RecommendationItem(
            recommendation_id=f"rec_{uuid4().hex[:8]}",
            action="Maintain promotional strategy while monitoring inventory burn rate",
            category=BusinessActionCategory.PROMOTION,
            priority=DecisionPriority.MEDIUM,
            rationale=rationale,
            evidence=evidence,
            expected_impact="Captures full promotional demand potential while preempting campaign stockout.",
            risk=risk_level,
            confidence=confidence,
            recommendation_score=score,
            supporting_forecast=dynamics,
            supporting_features=[f for f in ["promo_flag", "promotion", "promo"] if f in xai_insights["all_features"]] or xai_insights["top_positive_features"][:2],
            trade_offs=trade_offs,
            assumptions=[],
            warnings=[],
            requires_human_review=False,
            rule_id="RULE_4_DEMAND_INCREASE_PROMOTION",
        )
        return True, rec

    def _check_rule_5_pricing(
        self,
        forecast: ForecastResult,
        dynamics: Dict[str, Any],
        context: Optional[BusinessContext],
        xai_insights: Dict[str, Any],
        confidence: float,
        warnings: List[str],
    ) -> Tuple[bool, Optional[RecommendationItem]]:
        """Rule 5: Price-Driven Demand Change -> Review Pricing Strategy (Human Review Required)."""
        price_contrib = abs(xai_insights["price_contribution"])
        is_price_dominant = price_contrib > 0.15 or ("price" in xai_insights["top_features"][:2])

        if not is_price_dominant:
            return False, None

        evidence = [
            f"Price attribute is identified as a major driver of forecast variance (Attribution: {round(xai_insights['price_contribution'], 4)}).",
            f"Demand sensitivity to price is ranked in top {min(2, len(xai_insights['top_features']))} explanation features.",
        ]
        if context is not None and context.current_price is not None:
            evidence.append(f"Active base selling price is ${round(context.current_price, 2)}.")

        price_dir = "suppressing" if xai_insights["price_contribution"] < 0 else "stimulating"
        rationale = (
            f"Phase 6 feature attribution reveals that pricing is a dominant factor {price_dir} forecasted customer "
            f"demand. Conducting a structured pricing elasticity and margin review is recommended before adjusting orders."
        )

        trade_offs = TradeOff(
            benefit="Optimizes gross margin and price elasticity balance.",
            trade_off="Price changes carry customer demand sensitivity and competitive positioning risks.",
            quantified_impact="Prevents revenue leakage from uncalibrated price points.",
        )

        score = DecisionScorer.compute_recommendation_score(
            priority=DecisionPriority.HIGH,
            rule_severity=0.75,
            demand_gap_factor=0.5,
            explanation_support_score=0.95,
            confidence=confidence,
            has_missing_context=False,
        )

        rec = RecommendationItem(
            recommendation_id=f"rec_{uuid4().hex[:8]}",
            action="Review pricing strategy and elasticity",
            category=BusinessActionCategory.PRICING,
            priority=DecisionPriority.HIGH,
            rationale=rationale,
            evidence=evidence,
            expected_impact="Aligns retail price point with observed demand elasticity and margin targets.",
            risk="Suboptimal price elasticity impacting margins or customer demand",
            confidence=confidence,
            recommendation_score=score,
            supporting_forecast=dynamics,
            supporting_features=["price"] if "price" in xai_insights["all_features"] else xai_insights["top_features"][:2],
            trade_offs=trade_offs,
            assumptions=["Pricing modifications require managerial approval; autonomous pricing disabled."],
            warnings=["Price elasticity is heavily influencing forecast output."],
            requires_human_review=True,  # Explicit Requirement
            rule_id="RULE_5_PRICE_DRIVEN_DEMAND_CHANGE",
        )
        return True, rec

    def _check_rule_6_uncertainty(
        self,
        forecast: ForecastResult,
        dynamics: Dict[str, Any],
        confidence: float,
        uncertainty_threshold: float,
        warnings: List[str],
    ) -> Tuple[bool, Optional[RecommendationItem]]:
        """Rule 6: High Forecast Uncertainty -> Conservative Buffers & Human Review."""
        rel_unc = dynamics["relative_uncertainty"]
        if not dynamics["has_intervals"] or rel_unc <= uncertainty_threshold:
            return False, None

        evidence = [
            f"Forecast prediction interval width is {round(rel_unc * 100, 1)}% of point forecast, exceeding the uncertainty threshold ({round(uncertainty_threshold * 100, 1)}%).",
            f"Uncertainty calculation method: {forecast.uncertainty_method or 'prediction interval bounds'}.",
            f"Downside-to-upside variance creates high operational risk.",
        ]

        rationale = (
            f"The candidate model exhibits substantial forecast dispersion (average interval spread: "
            f"{round(rel_unc * 100, 1)}%). A conservative inventory buffering approach is recommended "
            f"with mandatory supervisory sign-off before committing substantial procurement capital."
        )

        trade_offs = TradeOff(
            benefit="Protects business solvency against extreme forecast variance and stockout/overstock tails.",
            trade_off="Conservative inventory buffers may result in minor lost sales if demand surges to upper bound.",
            quantified_impact=f"Bounds operational commitment within safe variance margin (spread: {round(rel_unc * 100, 1)}%).",
        )

        score = DecisionScorer.compute_recommendation_score(
            priority=DecisionPriority.HIGH,
            rule_severity=0.75,
            demand_gap_factor=0.6,
            explanation_support_score=0.50,
            confidence=confidence,
            has_missing_context=False,
        )

        rec = RecommendationItem(
            recommendation_id=f"rec_{uuid4().hex[:8]}",
            action="Adopt conservative inventory planning with human supervisory review",
            category=BusinessActionCategory.RISK,
            priority=DecisionPriority.HIGH,
            rationale=rationale,
            evidence=evidence,
            expected_impact="Minimizes financial downside exposure from statistical forecast variance.",
            risk="Forecast uncertainty creates asymmetrical stockout or excess inventory exposure",
            confidence=confidence,
            recommendation_score=score,
            supporting_forecast=dynamics,
            supporting_features=[],
            trade_offs=trade_offs,
            assumptions=["Forecast intervals reflect model dispersion rather than operational certainty."],
            warnings=[f"Elevated forecast dispersion: interval width is {round(rel_unc * 100, 1)}% of point forecast."],
            requires_human_review=True,  # Explicit Requirement
            rule_id="RULE_6_HIGH_FORECAST_UNCERTAINTY",
        )
        return True, rec

    def _check_rule_7_fidelity(
        self,
        explanation: Optional[ExplanationResult],
        confidence: float,
        fidelity_threshold: float,
        warnings: List[str],
    ) -> Tuple[bool, Optional[RecommendationItem]]:
        """Rule 7: Low Explanation Fidelity -> Flag Decision & Require Review."""
        if explanation is None or explanation.fidelity is None:
            return False, None

        fid = explanation.fidelity.fidelity_score
        status = explanation.fidelity.explanation_status
        method = (explanation.method or "").lower().strip()

        # Method-aware fidelity evaluation:
        # - LIME: fidelity represents local linear surrogate R² (goodness-of-fit).
        #         Evaluated against user-configured fidelity_threshold.
        # - SHAP: fidelity represents additive mathematical reconstruction (efficiency axiom).
        #         Fails if below threshold or status is APPROXIMATE (>20% reconstruction error).
        # - Prophet / Component: exact GAM decomposition; triggers only if decomposed components diverge.
        if method == "lime":
            is_low_fidelity = fid < fidelity_threshold
        elif method == "shap":
            is_low_fidelity = (fid < fidelity_threshold) or (status == "APPROXIMATE")
        elif method in ("component_based", "prophet"):
            is_low_fidelity = (fid < fidelity_threshold) or (status == "APPROXIMATE")
        else:
            is_low_fidelity = (fid < fidelity_threshold) or (status == "APPROXIMATE")

        if not is_low_fidelity:
            return False, None

        # Method-specific evidence, rationale, and operational risk
        if method == "lime":
            surr_r2 = explanation.fidelity.surrogate_r2 if explanation.fidelity.surrogate_r2 is not None else fid
            evidence = [
                f"LIME local linear surrogate goodness-of-fit R² is {round(surr_r2, 4)} ({round(surr_r2 * 100, 1)}%, status: {status}), falling below reliability threshold ({round(fidelity_threshold, 2)}).",
                f"Explanation methodology: LIME (Local Interpretable Model-agnostic Explanations).",
                f"Local linear surrogate residual error is {explanation.fidelity.reconstruction_error}.",
            ]
            rationale = (
                f"The LIME local linear surrogate exhibited low goodness-of-fit (R²={round(surr_r2 * 100, 1)}%, status: {status}) "
                f"due to the non-linear decision boundary of the underlying forecasting model in this perturbation neighborhood. "
                f"While the forecast itself remains valid, linear feature attribution slopes are approximate and should not be "
                f"relied upon for automated business interventions without human supervisory review."
            )
            risk = "Low LIME surrogate R² indicates linear attributions may not capture non-linear feature interactions"
        elif method == "shap":
            evidence = [
                f"SHAP additive reconstruction fidelity is {round(fid * 100, 1)}% ({status}), falling below reliability threshold ({round(fidelity_threshold * 100, 1)}%).",
                f"Explanation methodology: SHAP (Shapley Additive exPlanations).",
                f"Additive reconstruction error |y_pred - (base + sum(phi))| is {explanation.fidelity.reconstruction_error}.",
            ]
            rationale = (
                f"The SHAP feature attribution exhibited an additive reconstruction discrepancy (fidelity: {round(fid * 100, 1)}%, status: {status}). "
                f"Because the sum of Shapley attributions plus expected base value deviates from the model prediction by "
                f"{explanation.fidelity.reconstruction_error}, the efficiency axiom does not strictly hold, "
                f"and automated feature-level decisions require human supervisory verification."
            )
            risk = "SHAP additive reconstruction discrepancy indicates numerical or efficiency axiom deviation"
        else:
            evidence = [
                f"Explanation fidelity score is {round(fid * 100, 1)}% ({status}), falling below reliability threshold ({round(fidelity_threshold * 100, 1)}%).",
                f"Explanation methodology: {explanation.method.upper()}.",
                f"Component / reconstruction residual is {explanation.fidelity.reconstruction_error}.",
            ]
            rationale = (
                f"The feature attribution exhibited low fidelity ({round(fid * 100, 1)}%, status: {status}). "
                f"Automated feature-level decisions should not be relied upon without human expert review."
            )
            risk = "Unreliable feature attribution may mask true operational demand drivers"

        trade_offs = TradeOff(
            benefit="Prevents erroneous operational interventions based on low-fidelity or approximate explanations.",
            trade_off="Requires operational manager intervention and delays immediate automated dispatch.",
            quantified_impact="Preserves decision integrity and audit compliance.",
        )

        score = DecisionScorer.compute_recommendation_score(
            priority=DecisionPriority.MEDIUM,
            rule_severity=0.65,
            demand_gap_factor=0.3,
            explanation_support_score=0.20,
            confidence=confidence,
            has_missing_context=False,
        )

        rec = RecommendationItem(
            recommendation_id=f"rec_{uuid4().hex[:8]}",
            action="Flag decision for human supervisory review due to low explanation fidelity",
            category=BusinessActionCategory.MONITORING,
            priority=DecisionPriority.MEDIUM,
            rationale=rationale,
            evidence=evidence,
            expected_impact="Mandates manual operational sanity check prior to executing inventory/pricing decisions.",
            risk=risk,
            confidence=confidence,
            recommendation_score=score,
            supporting_forecast={},
            supporting_features=[],
            trade_offs=trade_offs,
            assumptions=["Explanation attributions are treated as approximate indicators rather than ground-truth causality."],
            warnings=[f"Phase 6 explanation fidelity is {round(fid * 100, 1)}% ({status}); feature explanations require caution."],
            requires_human_review=True,  # Explicit Requirement
            rule_id="RULE_7_LOW_EXPLANATION_FIDELITY",
        )
        return True, rec

    def _build_rule_missing_inventory(
        self,
        forecast: ForecastResult,
        dynamics: Dict[str, Any],
        xai_insights: Dict[str, Any],
        confidence: float,
        assumptions: List[str],
        warnings: List[str],
    ) -> RecommendationItem:
        """Rule 8: Missing inventory information -> Guideline recommendation without value fabrication."""
        trend = dynamics["trend_direction"]
        total_demand = dynamics["total_demand"]

        evidence = [
            f"Forecasted demand is {trend} with total projected demand of {total_demand} units over the planning horizon.",
            "Current physical inventory is not specified in the business context.",
            "No inventory values were fabricated or imputed; strict zero-fabrication contract enforced.",
        ]
        if xai_insights["top_positive_features"]:
            evidence.append(f"Demand is supported by features: {', '.join(xai_insights['top_positive_features'][:3])}.")

        rationale = (
            f"Demand is forecast to {trend} (total {total_demand} units), but physical inventory information "
            f"is unavailable in the business context. Inventory planning review is recommended to audit on-hand stock "
            f"before initiating replenishment."
        )

        trade_offs = TradeOff(
            benefit="Prevents unguided ordering and unnecessary capital expenditure.",
            trade_off="Requires operational inventory verification prior to replenishment action.",
            quantified_impact="Protects against overstock or stockout due to blind ordering.",
        )

        score = DecisionScorer.compute_recommendation_score(
            priority=DecisionPriority.HIGH,
            rule_severity=0.70,
            demand_gap_factor=0.5,
            explanation_support_score=0.50,
            confidence=confidence,
            has_missing_context=True,
        )

        return RecommendationItem(
            recommendation_id=f"rec_{uuid4().hex[:8]}",
            action="Conduct comprehensive inventory planning review to audit stock levels",
            category=BusinessActionCategory.PLANNING,
            priority=DecisionPriority.HIGH,
            rationale=rationale,
            evidence=evidence,
            expected_impact="Establishes ground-truth inventory levels before placing replenishment orders.",
            risk="Unverified stockout exposure due to lack of on-hand inventory visibility",
            confidence=confidence,
            recommendation_score=score,
            supporting_forecast=dynamics,
            supporting_features=xai_insights["top_positive_features"][:3],
            trade_offs=trade_offs,
            assumptions=["Current inventory assumed unknown; no values were fabricated."],
            warnings=["Missing current_inventory in business context."],
            requires_human_review=True,  # Explicit Requirement
            rule_id="RULE_MISSING_INVENTORY_PLANNING",
        )

    def _build_steady_state_recommendation(
        self,
        forecast: ForecastResult,
        dynamics: Dict[str, Any],
        context: Optional[BusinessContext],
        confidence: float,
    ) -> RecommendationItem:
        """Fallback rule for steady-state balanced inventory."""
        evidence = [
            f"Forecast demand is steady (slope: {round(dynamics['trend_slope'] * 100, 1)}%, total: {dynamics['total_demand']} units).",
            "Inventory and operational indicators are balanced within acceptable operating boundaries.",
        ]

        rationale = (
            "Current forecast and business context indicators reflect stable operating conditions. "
            "Maintain standard replenishment cadences while monitoring regular demand consumption."
        )

        trade_offs = TradeOff(
            benefit="Maintains operational equilibrium without unnecessary supply chain disruptions.",
            trade_off="Standard cadence assumes historical demand patterns continue without external shocks.",
            quantified_impact="Preserves baseline operating margins.",
        )

        score = DecisionScorer.compute_recommendation_score(
            priority=DecisionPriority.LOW,
            rule_severity=0.30,
            demand_gap_factor=0.0,
            explanation_support_score=0.50,
            confidence=confidence,
            has_missing_context=False,
        )

        return RecommendationItem(
            recommendation_id=f"rec_{uuid4().hex[:8]}",
            action="Maintain standard replenishment pacing and monitor demand velocity",
            category=BusinessActionCategory.MONITORING,
            priority=DecisionPriority.LOW,
            rationale=rationale,
            evidence=evidence,
            expected_impact="Sustains operational balance across inventory and fulfillment channels.",
            risk="Routine demand variance",
            confidence=confidence,
            recommendation_score=score,
            supporting_forecast=dynamics,
            supporting_features=[],
            trade_offs=trade_offs,
            assumptions=[],
            warnings=[],
            requires_human_review=False,
            rule_id="RULE_STEADY_STATE_MONITORING",
        )

    # =========================================================================
    # Explanation Feature Extraction Helper
    # =========================================================================

    def _extract_explanation_insights(self, explanation: Optional[ExplanationResult]) -> Dict[str, Any]:
        """Extracts key explanation drivers from Phase 6 ExplanationResult."""
        insights: Dict[str, Any] = {
            "all_features": [],
            "top_features": [],
            "top_positive_features": [],
            "top_negative_features": [],
            "promo_contribution": 0.0,
            "price_contribution": 0.0,
            "demand_momentum_supported": False,
        }

        if explanation is None:
            return insights

        # Local explanation features
        feats = explanation.features or []
        for f in feats:
            insights["all_features"].append(f.feature)
            val = getattr(f, "contribution", getattr(f, "attribution", 0.0))
            if val > 0:
                insights["top_positive_features"].append(f.feature)
            elif val < 0:
                insights["top_negative_features"].append(f.feature)

            if "promo" in f.feature.lower():
                insights["promo_contribution"] = val
            if "price" in f.feature.lower():
                insights["price_contribution"] = val
            if any(term in f.feature.lower() for term in ["lag_1", "lag_7", "rolling_mean", "momentum"]):
                if val > 0.1:
                    insights["demand_momentum_supported"] = True

        # Check global importance if local features empty
        if not feats and explanation.global_importance:
            for g in explanation.global_importance:
                insights["all_features"].append(g.feature)
                insights["top_features"].append(g.feature)

        if not insights["top_features"] and feats:
            insights["top_features"] = [f.feature for f in feats[:5]]

        return insights
