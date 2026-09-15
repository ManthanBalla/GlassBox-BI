"""Unit and Integration Tests for Orchestration REST API Endpoints (Phase 8)."""

from fastapi.testclient import TestClient
import pytest

from backend.app.main import app

client = TestClient(app)


def test_api_orchestration_health():
    """Verify GET /api/v1/orchestration/health response schema and status."""
    res = client.get("/api/v1/orchestration/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["module"] == "Phase 8 — Multi-Agent Orchestration"
    assert len(data["pipeline_sequence"]) == 5
    assert data["self_correction_enabled"] is False
    assert data["automatic_retries_enabled"] is False


def test_api_orchestration_stages():
    """Verify GET /api/v1/orchestration/stages returns 5 stages and dependencies."""
    res = client.get("/api/v1/orchestration/stages")
    assert res.status_code == 200
    stages = res.json()
    assert len(stages) == 5
    stage_names = [s["stage"] for s in stages]
    assert stage_names == [
        "DATA_PROCESSING",
        "FORECASTING",
        "EVALUATION",
        "EXPLAINABILITY",
        "DECISION_INTELLIGENCE",
    ]


def test_api_orchestration_sample():
    """Verify GET /api/v1/orchestration/sample executes fast benchmark sample."""
    res = client.get("/api/v1/orchestration/sample")
    assert res.status_code == 200
    data = res.json()
    assert data["workflow_id"].startswith("wf_")
    assert data["status"] in ("COMPLETED", "PARTIAL")
    assert "DATA_PROCESSING" in data["stage_summary"]
    assert "FORECASTING" in data["stage_summary"]
    assert data["forecast"] is not None


def test_api_orchestration_run_and_get_by_id():
    """Verify POST /api/v1/orchestration/run followed by GET /api/v1/orchestration/{id}."""
    payload = {
        "dataset_path": "data/processed/synthetic/retail_processed.csv",
        "entity_id": "STORE_001",
        "product_id": "PROD_001",
        "horizon": 7,
        "selection_metric": "MAE",
        "candidate_models": ["lightgbm"],
        "run_evaluation": False,
        "run_explainability": False,
    }
    res = client.post("/api/v1/orchestration/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    wf_id = data["workflow_id"]
    assert data["status"] in ("COMPLETED", "PARTIAL")

    # Retrieve by ID
    res_get = client.get(f"/api/v1/orchestration/{wf_id}")
    assert res_get.status_code == 200
    state_data = res_get.json()
    assert state_data["workflow_id"] == wf_id


def test_api_orchestration_not_found():
    """Verify GET /api/v1/orchestration/{id} with unknown ID yields 404."""
    res = client.get("/api/v1/orchestration/wf_non_existent_999999")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_api_orchestration_bad_request():
    """Verify POST /api/v1/orchestration/run with invalid horizon yields 422 or 400."""
    payload = {
        "horizon": -5,  # Invalid
    }
    res = client.post("/api/v1/orchestration/run", json=payload)
    assert res.status_code in (400, 422)
