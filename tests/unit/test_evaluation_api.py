"""Unit tests for Phase 5 Evaluation API Endpoints.

Validates:
1. GET /api/v1/evaluation/health
2. GET /api/v1/evaluation/metrics
3. POST /api/v1/evaluation/run
4. GET /api/v1/evaluation/sample
5. Error responses (e.g. 400 Bad Request on invalid horizon).
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_evaluation_health_endpoint():
    """Verifies GET /api/v1/evaluation/health returns operational status."""
    response = client.get("/api/v1/evaluation/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "MAE" in data["supported_metrics"]
    assert "RMSE" in data["supported_metrics"]
    assert "MAPE" in data["supported_metrics"]
    assert data["zero_target_strategy"] == "exclude_zeros_from_mape"
    assert "Strictly Isolated" in data["test_set_protection"]


def test_evaluation_metrics_documentation_endpoint():
    """Verifies GET /api/v1/evaluation/metrics documents metric equations and zero-handling policy."""
    response = client.get("/api/v1/evaluation/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "MAE" in data["metrics"]
    assert "RMSE" in data["metrics"]
    assert "MAPE" in data["metrics"]
    assert "zero_target_handling" in data["metrics"]["MAPE"]
    assert data["protocols"]["test_set_isolation"] is not None
    assert data["protocols"]["selection_source"] is not None


def test_evaluation_run_api_endpoint():
    """Verifies POST /api/v1/evaluation/run executes benchmark on retail dataset."""
    sample_path = Path("data/processed/synthetic/retail_processed.csv")
    if not sample_path.exists():
        sample_path = Path("data/sample/retail_processed_sample.csv")
    if not sample_path.exists():
        sample_path = Path("data/sample/retail_sample.csv")
    if not sample_path.exists():
        pytest.skip("No sample retail dataset available for live API test.")

    payload = {
        "dataset_path": str(sample_path),
        "entity_id": "STORE_001",
        "product_id": "PROD_001",
        "horizon": 7,
        "candidate_models": ["lightgbm"],
        "random_seed": 42,
    }

    response = client.post("/api/v1/evaluation/run", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "SUCCESS"
    assert data["horizon"] == 7
    assert len(data["models"]) == 1
    assert data["models"][0]["model_name"] == "lightgbm"
    assert data["models"][0]["mae"] is not None
    assert data["models"][0]["rmse"] is not None
    assert data["models"][0]["mape"] is not None
    assert data["phase4_selected_model"] == "lightgbm"
    assert data["selection_source"] == "validation"
    assert data["test_set_used_for_selection"] is False


def test_evaluation_run_horizon_exceeded_returns_400():
    """Verifies POST /api/v1/evaluation/run returns 400 Bad Request when horizon exceeds test length."""
    sample_path = Path("data/processed/synthetic/retail_processed.csv")
    if not sample_path.exists():
        sample_path = Path("data/sample/retail_processed_sample.csv")
    if not sample_path.exists():
        sample_path = Path("data/sample/retail_sample.csv")
    if not sample_path.exists():
        pytest.skip("No sample retail dataset available for live API test.")

    payload = {
        "dataset_path": str(sample_path),
        "entity_id": "STORE_001",
        "product_id": "PROD_001",
        "horizon": 200,  # Far exceeds available 37 test days
        "candidate_models": ["lightgbm"],
    }

    response = client.post("/api/v1/evaluation/run", json=payload)
    assert response.status_code == 400
    assert "exceeds available test set observations" in response.json()["detail"]


def test_evaluation_run_invalid_horizon_pydantic_422():
    """Verifies POST /api/v1/evaluation/run returns 422 Unprocessable Entity when horizon <= 0."""
    payload = {
        "entity_id": "STORE_001",
        "product_id": "PROD_001",
        "horizon": 0,  # Invalid: ge=1 constraint
        "candidate_models": ["lightgbm"],
    }
    response = client.post("/api/v1/evaluation/run", json=payload)
    assert response.status_code == 422


def test_evaluation_sample_endpoint():
    """Verifies GET /api/v1/evaluation/sample executes fast demonstration benchmark."""
    sample_path = Path("data/processed/synthetic/retail_processed.csv")
    if not sample_path.exists():
        sample_path = Path("data/sample/retail_processed_sample.csv")
    if not sample_path.exists():
        pytest.skip("No sample dataset found.")

    response = client.get("/api/v1/evaluation/sample?horizon=5&models=lightgbm")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["horizon"] == 5
    assert len(data["models"]) == 1
    assert data["test_set_used_for_selection"] is False
