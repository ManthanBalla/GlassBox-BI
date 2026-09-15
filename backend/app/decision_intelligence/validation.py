"""Decision Intelligence Validation Logic for GlassBox-BI (Phase 7).

Ensures all recommendations, inputs, scores, and audit trails satisfy strict methodological
and architectural invariants:
- Zero value fabrication for missing business context
- Valid priority, category, confidence, and recommendation score ranges
- Traceable evidence linkage and non-empty rationale
- Mandatory human review flagging under critical risk, low confidence, or pricing reviews
"""

from typing import List, Optional

from backend.app.schemas.decisions import (
    BusinessActionCategory,
    BusinessContext,
    DecisionPriority,
    DecisionResult,
    RecommendationItem,
)
from backend.app.schemas.explainability import ExplanationResult
from backend.app.schemas.forecasting import ForecastResult


class DecisionValidator:
    """Validates inputs, generated recommendations, and audit records."""

    @staticmethod
    def validate_inputs(
        forecast: Optional[ForecastResult],
        context: Optional[BusinessContext],
        explanation: Optional[ExplanationResult] = None,
    ) -> List[str]:
        """Validates input payload integrity prior to rule evaluation."""
        issues: List[str] = []

        if forecast is None:
            issues.append("ForecastResult cannot be None. Forecast data is required for decision intelligence.")
            return issues

        preds = forecast.predictions or forecast.forecast_points or []
        if not preds:
            issues.append("ForecastResult contains zero prediction points.")

        for idx, p in enumerate(preds):
            if p.prediction < 0.0:
                issues.append(f"Forecast prediction at index {idx} ({p.date}) is negative ({p.prediction}).")
            if p.lower_bound is not None and p.upper_bound is not None:
                if p.lower_bound > p.upper_bound:
                    issues.append(
                        f"Inverted prediction interval at index {idx}: lower ({p.lower_bound}) > upper ({p.upper_bound})."
                    )

        if context is not None:
            if context.current_inventory is not None and context.current_inventory < 0:
                issues.append("BusinessContext current_inventory cannot be negative.")
            if context.reorder_point is not None and context.reorder_point < 0:
                issues.append("BusinessContext reorder_point cannot be negative.")
            if context.lead_time_days is not None and context.lead_time_days <= 0:
                issues.append("BusinessContext lead_time_days must be positive.")
            if context.current_price is not None and context.current_price <= 0:
                issues.append("BusinessContext current_price must be positive.")

        return issues

    @staticmethod
    def validate_recommendation(rec: RecommendationItem) -> List[str]:
        """Validates that a single recommendation item satisfies all contract invariants."""
        issues: List[str] = []

        if not rec.recommendation_id:
            issues.append("RecommendationItem recommendation_id must be non-empty.")
        if not rec.action or not rec.action.strip():
            issues.append("RecommendationItem action description must be non-empty.")
        if not isinstance(rec.category, BusinessActionCategory):
            issues.append(f"Invalid recommendation category: {rec.category}.")
        if not isinstance(rec.priority, DecisionPriority):
            issues.append(f"Invalid recommendation priority: {rec.priority}.")
        if not rec.rationale or not rec.rationale.strip():
            issues.append("RecommendationItem rationale must be non-empty.")
        if not rec.evidence:
            issues.append("RecommendationItem evidence list cannot be empty.")
        if not (0.0 <= rec.confidence <= 1.0):
            issues.append(f"Decision confidence {rec.confidence} out of range [0.0, 1.0].")
        if not (0.0 <= rec.recommendation_score <= 100.0):
            issues.append(f"Recommendation score {rec.recommendation_score} out of range [0.0, 100.0].")
        if not rec.rule_id:
            issues.append("RecommendationItem rule_id is required.")

        # Human review invariants
        if rec.priority == DecisionPriority.CRITICAL and not rec.requires_human_review:
            issues.append("CRITICAL priority recommendations mandate requires_human_review=True.")
        if rec.category == BusinessActionCategory.PRICING and not rec.requires_human_review:
            issues.append("PRICING category recommendations mandate requires_human_review=True.")
        if rec.confidence < 0.65 and not rec.requires_human_review:
            issues.append("Low confidence (<0.65) recommendations mandate requires_human_review=True.")

        return issues

    @classmethod
    def validate_decision_result(cls, result: DecisionResult) -> List[str]:
        """Validates top-level DecisionResult structure and consistency."""
        issues: List[str] = []

        if not result.decision_id:
            issues.append("DecisionResult decision_id cannot be empty.")

        if result.recommendations:
            for idx, rec in enumerate(result.recommendations):
                rec_issues = cls.validate_recommendation(rec)
                for issue in rec_issues:
                    issues.append(f"Recommendation[{idx}]: {issue}")

            # Check primary recommendation consistency
            if result.primary_recommendation is None:
                issues.append("DecisionResult contains recommendations but primary_recommendation is None.")
            else:
                if result.primary_recommendation.recommendation_id != result.recommendations[0].recommendation_id:
                    issues.append("primary_recommendation does not match the first ranked recommendation in the list.")

        # If any recommendation requires review, overall must require review
        any_review = any(r.requires_human_review for r in result.recommendations)
        if any_review and not result.requires_human_review:
            issues.append("DecisionResult requires_human_review must be True when any recommendation mandates review.")

        return issues
