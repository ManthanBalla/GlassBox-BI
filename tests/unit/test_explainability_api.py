"""Unit Tests for Explainability API Endpoints (Phase 6)."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_explainability_health_endpoint():
    response = client.get("/api/v1/explainability/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "shap" in data["libraries"]
    assert "lime" in data["libraries"]
    assert "lightgbm" in data["supported_models"]


def test_explainability_methods_endpoint():
    response = client.get("/api/v1/explainability/methods")
    assert response.status_code == 200
    data = response.json()
    assert "lightgbm" in data
    assert "prophet" in data
    assert "lstm" in data
    assert "shap" in data["lightgbm"]["supported_methods"]
    assert "component_based" in data["prophet"]["supported_methods"]


def test_explainability_config_endpoint():
    response = client.get("/api/v1/explainability/config")
    assert response.status_code == 200
    data = response.json()
    assert "default_background_samples" in data
    assert data["default_random_seed"] == 42


def test_explainability_local_endpoint():
    payload = {
        "dataset_path": "data/processed/synthetic/retail_processed.csv",
        "entity_id": "STORE_001",
        "product_id": "PROD_001",
        "model_name": "lightgbm",
        "method": "shap",
        "background_samples": 20,
    }
    response = client.post("/api/v1/explainability/local", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "lightgbm"
    assert data["method"] == "shap"
    assert data["explanation_type"] == "local"
    assert len(data["features"]) > 0
    assert "fidelity" in data
    assert "audit_trail" in data


def test_explainability_global_endpoint():
    payload = {
        "dataset_path": "data/processed/synthetic/retail_processed.csv",
        "entity_id": "STORE_001",
        "product_id": "PROD_001",
        "model_name": "lightgbm",
        "method": "shap",
        "sample_size": 20,
        "top_k": 5,
    }
    response = client.post("/api/v1/explainability/global", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["explanation_type"] == "global"
    assert len(data["global_importance"]) <= 5


def test_explainability_sample_endpoint():
    response = client.get("/api/v1/explainability/sample")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "lightgbm"
    assert data["explanation_type"] == "local"


def test_explainability_invalid_method_returns_400():
    payload = {
        "dataset_path": "data/processed/synthetic/retail_processed.csv",
        "entity_id": "STORE_001",
        "product_id": "PROD_001",
        "model_name": "prophet",
        "method": "lime",  # Prophet does not support LIME
    }
    response = client.post("/api/v1/explainability/local", json=payload)
    assert response.status_code == 400
    assert "not supported for model 'prophet'" in response.json()["detail"]
