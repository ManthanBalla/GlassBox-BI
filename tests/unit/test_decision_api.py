"""Unit tests for Phase 7 Decision Intelligence REST API endpoints."""

from fastapi.testclient import TestClient
import pytest

from backend.app.main import app

client = TestClient(app)


def test_decisions_health_endpoint():
    response = client.get("/api/v1/decisions/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["llm_dependency"] is False
    assert data["autonomous_execution"] is False
    assert "INVENTORY" in data["supported_categories"]


def test_decisions_rules_endpoint():
    response = client.get("/api/v1/decisions/rules")
    assert response.status_code == 200
    rules = response.json()
    assert isinstance(rules, list)
    assert len(rules) >= 7
    rule_ids = [r["rule_id"] for r in rules]
    assert "RULE_1_HIGH_DEMAND_LOW_INVENTORY" in rule_ids
    assert "RULE_2_HIGH_STOCKOUT_CRITICAL" in rule_ids
    assert "RULE_5_PRICE_DRIVEN_DEMAND_CHANGE" in rule_ids


def test_decisions_config_endpoint():
    response = client.get("/api/v1/decisions/config")
    assert response.status_code == 200
    data = response.json()
    assert "uncertainty_relative_threshold" in data
    assert "stockout_critical_coverage_days" in data


def test_decisions_sample_endpoint():
    response = client.get("/api/v1/decisions/sample")
    assert response.status_code == 200
    data = response.json()
    assert data["decision_id"].startswith("dec_")
    assert data["primary_recommendation"] is not None
    assert "action" in data["primary_recommendation"]
    assert "priority" in data["primary_recommendation"]
    assert "recommendation_score" in data["primary_recommendation"]
    assert "confidence" in data["primary_recommendation"]


def test_decisions_run_endpoint():
    payload = {
        "entity_id": "STORE_001",
        "product_id": "PROD_001",
        "business_context": {
            "current_inventory": 30.0,
            "reorder_point": 90.0,
            "lead_time_days": 7,
            "current_price": 19.99,
            "promotion_active": False,
        },
    }
    response = client.post("/api/v1/decisions/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["primary_recommendation"] is not None
    assert data["primary_recommendation"]["priority"] in ["HIGH", "CRITICAL"]


def test_decisions_scenario_endpoint():
    payload = {
        "scenario_name": "Inventory Reduction Test",
        "base_request": {
            "entity_id": "STORE_001",
            "product_id": "PROD_001",
            "business_context": {
                "current_inventory": 150.0,
                "reorder_point": 80.0,
            },
        },
        "inventory_delta": -100.0,
    }
    response = client.post("/api/v1/decisions/scenario", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scenario_name"] == "Inventory Reduction Test"
    assert "delta_summary" in data
