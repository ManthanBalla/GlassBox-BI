"""Transparent Decision Scoring and Confidence Calibration for GlassBox-BI (Phase 7).

Methodological Separation:
- Forecast Uncertainty: Prediction interval width relative to point forecast ((upper - lower) / prediction).
- Decision Confidence: Quantitative trust in the recommendation derived from forecast validation quality,
  prediction interval tightness, explanation fidelity, and business context completeness.
- Recommendation Score: Deterministic rule strength index (0-100), explicitly NOT a statistical probability.
"""

from typing import Any, Dict, List, Optional, Tuple
from backend.app.schemas.decisions import BusinessContext, DecisionPriority
from backend.app.schemas.explainability import ExplanationResult
from backend.app.schemas.forecasting import ForecastResult


class DecisionScorer:
    """Computes transparent decision confidence and rule-based recommendation scores."""

    @staticmethod
    def calculate_forecast_dynamics(forecast: ForecastResult) -> Dict[str, Any]:
        """Extracts key dynamics from forecast predictions: trend, totals, and uncertainty."""
        preds = forecast.predictions or forecast.forecast_points or []
        if not preds:
            return {
                "count": 0,
                "total_demand": 0.0,
                "mean_demand": 0.0,
                "trend_slope": 0.0,
                "trend_direction": "flat",
                "relative_uncertainty": 0.0,
                "has_intervals": False,
            }

        values = [p.prediction for p in preds]
        total_demand = sum(values)
        mean_demand = total_demand / len(values) if values else 0.0

        # Simple linear slope across horizon
        n = len(values)
        if n >= 2:
            first_half = sum(values[: n // 2]) / (n // 2)
            second_half = sum(values[n // 2 :]) / (n - n // 2)
            slope = (second_half - first_half) / max(1e-4, first_half)
            if slope > 0.05:
                direction = "increasing"
            elif slope < -0.05:
                direction = "decreasing"
            else:
                direction = "flat"
        else:
            slope = 0.0
            direction = "flat"

        # Calculate prediction interval relative uncertainty
        intervals = []
        for p in preds:
            if p.upper_bound is not None and p.lower_bound is not None and p.prediction > 1e-4:
                rel_width = (p.upper_bound - p.lower_bound) / p.prediction
                intervals.append(rel_width)

        has_intervals = len(intervals) > 0
        avg_uncertainty = sum(intervals) / len(intervals) if intervals else 0.0

        return {
            "count": n,
            "total_demand": round(total_demand, 2),
            "mean_demand": round(mean_demand, 2),
            "trend_slope": round(slope, 4),
            "trend_direction": direction,
            "relative_uncertainty": round(avg_uncertainty, 4),
            "has_intervals": has_intervals,
            "peak_demand": round(max(values), 2) if values else 0.0,
            "min_demand": round(min(values), 2) if values else 0.0,
        }

    @classmethod
    def compute_decision_confidence(
        cls,
        forecast: ForecastResult,
        explanation: Optional[ExplanationResult],
        context: Optional[BusinessContext],
        uncertainty_threshold: float = 0.40,
        fidelity_threshold: float = 0.65,
    ) -> Tuple[float, List[str], List[str]]:
        """Calibrates decision confidence score from [0.10, 0.99].
        
        Returns:
            Tuple of (confidence_score, assumptions_list, warnings_list)
        """
        assumptions: List[str] = []
        warnings: List[str] = []

        # 1. Base confidence from model validation metrics
        base_confidence = 0.90
        metrics = forecast.metrics or {}
        if "MAPE" in metrics and metrics["MAPE"] > 0:
            mape_fraction = metrics["MAPE"] / 100.0 if metrics["MAPE"] > 1.0 else metrics["MAPE"]
            penalty = min(0.30, mape_fraction * 0.5)
            base_confidence -= penalty
        elif "MAE" in metrics:
            base_confidence = 0.88

        # 2. Forecast Uncertainty Penalty
        dynamics = cls.calculate_forecast_dynamics(forecast)
        rel_unc = dynamics["relative_uncertainty"]
        if dynamics["has_intervals"] and rel_unc > uncertainty_threshold:
            excess = rel_unc - uncertainty_threshold
            unc_penalty = min(0.30, round(excess * 0.5, 4))
            base_confidence -= unc_penalty
            warnings.append(
                f"High forecast uncertainty: average interval width is {round(rel_unc * 100, 1)}% "
                f"of predicted demand (threshold: {round(uncertainty_threshold * 100, 1)}%)."
            )
        elif not dynamics["has_intervals"]:
            assumptions.append("Prediction intervals unavailable; point forecasts treated as deterministic baseline.")

        # 3. Explanation Fidelity Penalty
        if explanation is not None and explanation.fidelity is not None:
            fid_score = explanation.fidelity.fidelity_score
            status = explanation.fidelity.explanation_status
            method = (explanation.method or "").lower().strip()

            if method == "lime":
                is_low_fid = fid_score < fidelity_threshold
            else:
                is_low_fid = (fid_score < fidelity_threshold) or (status == "APPROXIMATE")

            if is_low_fid:
                # Differentiate penalty: SHAP additive breakdown is more severe (-0.25)
                # than LIME linear surrogate approximation on non-linear boundaries (-0.20)
                pen = 0.20 if method == "lime" else 0.25
                base_confidence -= pen
                metric_name = "surrogate R²" if method == "lime" else "reconstruction fidelity"
                warnings.append(
                    f"Low explanation fidelity: {explanation.method.upper()} {metric_name} is {round(fid_score * 100, 1)}% "
                    f"({status}). Feature attribution evidence may be approximate."
                )
            elif fid_score < 0.85:
                base_confidence -= 0.08
        elif explanation is None:
            assumptions.append("No Phase 6 explanation provided; decisions based purely on forecast and business parameters.")
            base_confidence -= 0.10

        # 4. Business Context Completeness Penalty (No Value Fabrication)
        if context is None:
            base_confidence -= 0.35
            assumptions.append("Business context is missing; operational inventory parameters assumed unknown.")
            warnings.append("Missing complete business context. Prescriptive recommendations limited to planning guidelines.")
        else:
            if context.current_inventory is None:
                base_confidence -= 0.20
                assumptions.append("Current inventory level not provided; inventory planning review recommended.")
                warnings.append("Missing current_inventory. Physical stockout risk cannot be definitively verified.")
            if context.reorder_point is None and context.current_inventory is not None:
                base_confidence -= 0.08
                assumptions.append("Reorder point not configured; evaluated against total horizon demand.")
            if context.lead_time_days is None:
                base_confidence -= 0.05
                assumptions.append("Supplier lead time not configured; default 7-day fulfillment assumed.")

        # Bound confidence between 0.10 and 0.99
        calibrated_confidence = round(max(0.10, min(0.99, base_confidence)), 4)
        return calibrated_confidence, assumptions, warnings

    @classmethod
    def compute_recommendation_score(
        cls,
        priority: DecisionPriority,
        rule_severity: float,
        demand_gap_factor: float,
        explanation_support_score: float,
        confidence: float,
        has_missing_context: bool,
    ) -> float:
        """Calculates transparent rule strength index (0-100).
        
        Args:
            priority: Priority level (CRITICAL, HIGH, MEDIUM, LOW)
            rule_severity: Baseline rule severity weight (0.0 to 1.0)
            demand_gap_factor: Normalized demand/inventory imbalance factor (-1.0 to 1.0)
            explanation_support_score: Alignment of XAI attributions with rule logic (0.0 to 1.0)
            confidence: Calibrated decision confidence (0.0 to 1.0)
            has_missing_context: Whether critical fields were absent
        """
        # Baseline priority anchor
        anchor_map = {
            DecisionPriority.CRITICAL: 88.0,
            DecisionPriority.HIGH: 74.0,
            DecisionPriority.MEDIUM: 58.0,
            DecisionPriority.LOW: 42.0,
        }
        score = anchor_map.get(priority, 50.0)

        # Severity adjustment (+/- 6 points)
        score += (rule_severity - 0.5) * 12.0

        # Demand imbalance gap adjustment (+/- 8 points)
        score += demand_gap_factor * 8.0

        # Explanation evidence support adjustment (+/- 6 points)
        score += (explanation_support_score - 0.5) * 12.0

        # Confidence modulation (if confidence is low, pull score down)
        if confidence < 0.65:
            score -= (0.65 - confidence) * 25.0

        # Missing context penalty
        if has_missing_context:
            score -= 10.0

        return round(max(5.0, min(99.0, score)), 1)
