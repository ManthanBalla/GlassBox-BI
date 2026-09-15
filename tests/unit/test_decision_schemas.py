"""Unit tests for Phase 7 Decision Intelligence Schemas."""

from datetime import datetime
import pytest
from pydantic import ValidationError

from backend.app.schemas.decisions import (
    BusinessActionCategory,
    BusinessContext,
    DecisionAuditRecord,
    DecisionConfigSchema,
    DecisionPriority,
    DecisionRequest,
    DecisionResult,
    RecommendationItem,
    ScenarioRequest,
    ScenarioResult,
    TradeOff,
)


def test_decision_priority_and_category_enums():
    assert DecisionPriority.CRITICAL.value == "CRITICAL"
    assert DecisionPriority.HIGH.value == "HIGH"
    assert DecisionPriority.MEDIUM.value == "MEDIUM"
    assert DecisionPriority.LOW.value == "LOW"

    assert BusinessActionCategory.INVENTORY.value == "INVENTORY"
    assert BusinessActionCategory.PRICING.value == "PRICING"
    assert BusinessActionCategory.PROMOTION.value == "PROMOTION"
    assert BusinessActionCategory.MONITORING.value == "MONITORING"
    assert BusinessActionCategory.RISK.value == "RISK"
    assert BusinessActionCategory.PLANNING.value == "PLANNING"


def test_business_context_validation():
    # Valid context
    ctx = BusinessContext(
        current_inventory=120.0,
        reorder_point=50.0,
        safety_stock=20.0,
        lead_time_days=5,
        current_price=19.99,
        promotion_active=True,
    )
    assert ctx.current_inventory == 120.0
    assert ctx.lead_time_days == 5
    assert ctx.promotion_active is True

    # Negative inventory rejected
    with pytest.raises(ValidationError):
        BusinessContext(current_inventory=-10.0)

    # Lead time zero rejected
    with pytest.raises(ValidationError):
        BusinessContext(lead_time_days=0)


def test_recommendation_item_constraints():
    rec = RecommendationItem(
        recommendation_id="rec_001",
        action="Increase replenishment quantity by 50 units",
        category=BusinessActionCategory.INVENTORY,
        priority=DecisionPriority.HIGH,
        rationale="Expected demand exceeds current on-hand stock.",
        evidence=["Demand expands by 18%", "Inventory below reorder point"],
        risk="Stockout risk during replenishment cycle",
        confidence=0.88,
        recommendation_score=78.5,
        rule_id="RULE_1_HIGH_DEMAND_LOW_INVENTORY",
    )
    assert rec.confidence == 0.88
    assert rec.recommendation_score == 78.5
    assert rec.priority == DecisionPriority.HIGH

    # Confidence must be between 0 and 1
    with pytest.raises(ValidationError):
        RecommendationItem(
            recommendation_id="rec_bad",
            action="Action",
            category=BusinessActionCategory.INVENTORY,
            priority=DecisionPriority.HIGH,
            rationale="Rationale",
            evidence=["Evidence"],
            risk="Risk",
            confidence=1.5,  # Invalid
            recommendation_score=50.0,
            rule_id="RULE_TEST",
        )

    # Recommendation score must be between 0 and 100
    with pytest.raises(ValidationError):
        RecommendationItem(
            recommendation_id="rec_bad2",
            action="Action",
            category=BusinessActionCategory.INVENTORY,
            priority=DecisionPriority.HIGH,
            rationale="Rationale",
            evidence=["Evidence"],
            risk="Risk",
            confidence=0.5,
            recommendation_score=120.0,  # Invalid
            rule_id="RULE_TEST",
        )


def test_decision_result_serialization():
    audit = DecisionAuditRecord(
        decision_id="dec_test_001",
        rules_evaluated=["RULE_1", "RULE_2"],
        rules_triggered=["RULE_1"],
        evidence_used=["Evidence 1"],
        primary_rule_id="RULE_1",
        decision_confidence=0.85,
        recommendation_score=76.0,
        execution_duration_ms=12.5,
    )
    res = DecisionResult(
        decision_id="dec_test_001",
        audit_trail=audit,
    )
    dump = res.model_dump()
    assert dump["decision_id"] == "dec_test_001"
    assert dump["audit_trail"]["primary_rule_id"] == "RULE_1"
    assert dump["requires_human_review"] is False
