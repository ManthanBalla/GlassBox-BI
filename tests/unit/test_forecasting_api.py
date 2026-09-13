"""Unit tests for Forecasting API v1 Endpoints.

Validates:
- GET /api/v1/forecast/health: checks dependency health and persistence directories
- GET /api/v1/forecast/models: technical metadata for prophet, lightgbm, and lstm
- GET /api/v1/forecast/config: default forecasting configurations
- GET /api/v1/forecast/sample: quick sample forecast execution on committed data
- POST /api/v1/forecast/run: full forecasting competition and prediction endpoint
- Request validation: rejection of invalid horizons and payloads
"""

from fastapi.testclient import TestClient
import pytest

from backend.app.main import app

client = TestClient(app)


def test_get_forecasting_health():
    """Verifies that the forecasting health endpoint confirms library availability."""
    response = client.get("/api/v1/forecast/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["dependencies"]["torch"] is True
    assert data["dependencies"]["lightgbm"] is True
    assert data["dependencies"]["prophet"] is True
    assert data["dependencies"]["sklearn"] is True
    assert "persistence_directory" in data


def test_get_forecasting_models():
    """Verifies retrieval of supported candidate architectures and history requirements."""
    response = client.get("/api/v1/forecast/models")
    assert response.status_code == 200
    data = response.json()
    candidates = data["candidate_models"]
    names = [c["name"] for c in candidates]
    assert "prophet" in names
    assert "lightgbm" in names
    assert "lstm" in names

    for c in candidates:
        assert c["minimum_history_rows"] > 0
        assert "uncertainty_method" in c
        assert len(c["supported_features"]) > 0


def test_get_forecasting_config():
    """Verifies default forecasting configuration payload."""
    response = client.get("/api/v1/forecast/config")
    assert response.status_code == 200
    data = response.json()
    assert data["forecast_horizon"] == 14
    assert data["model_selection_metric"] == "MAE"
    assert "prophet" in data["candidate_models"]
    assert "lightgbm" in data["candidate_models"]
    assert "lstm" in data["candidate_models"]


def test_run_sample_forecast():
    """Verifies fast demonstration forecast execution on sample data."""
    response = client.get("/api/v1/forecast/sample?model=lightgbm&horizon=5")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["horizon"] == 5
    assert len(data["predictions"]) == 5
    assert data["model_name"] == "lightgbm"


def test_post_run_forecast_valid():
    """Verifies standard POST /api/v1/forecast/run endpoint execution."""
    payload = {
        "dataset_path": "data/processed/synthetic/retail_processed.csv",
        "entity_id": "STORE_001",
        "product_id": "PROD_001",
        "horizon": 7,
        "candidate_models": ["lightgbm"],
        "selection_metric": "MAE",
        "confidence_level": 0.80,
    }
    response = client.post("/api/v1/forecast/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["model_name"] == "lightgbm"
    assert len(data["predictions"]) == 7
    assert data["model_metadata"]["model_name"] == "lightgbm"


def test_post_run_forecast_invalid_horizon():
    """Verifies validation error when horizon is <= 0 or violates Pydantic constraints."""
    payload = {
        "dataset_path": "data/sample/retail_processed_sample.csv",
        "entity_id": "STORE_001",
        "horizon": 0,  # Invalid: ge=1 constraint
        "candidate_models": ["lightgbm"],
    }
    response = client.post("/api/v1/forecast/run", json=payload)
    assert response.status_code == 422
