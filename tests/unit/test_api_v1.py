"""Unit tests for FastAPI application and API v1 routes."""

import unittest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings


class TestApiV1(unittest.TestCase):
    """Test suite for API endpoints and CORS configuration."""

    def setUp(self):
        self.client = TestClient(app)

    def test_root_health_endpoint(self):
        """Verify GET /health returns 200 and valid schema."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["app_name"], settings.app_name)
        self.assertEqual(data["version"], settings.version)
        self.assertIn("timestamp", data)

    def test_v1_health_endpoint(self):
        """Verify GET /api/v1/health returns 200 and matches HealthResponse schema."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], settings.version)

    def test_v1_contracts_specs_endpoint(self):
        """Verify GET /api/v1/contracts/specs returns registered contract metadata."""
        response = self.client.get("/api/v1/contracts/specs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("contracts", data)
        contract_names = [c["name"] for c in data["contracts"]]
        self.assertIn("DatasetMetadata", contract_names)
        self.assertIn("ForecastRequest", contract_names)
        self.assertIn("ForecastResult", contract_names)
        self.assertIn("ExplanationResult", contract_names)
        self.assertIn("RecommendationResult", contract_names)

    def test_cors_headers(self):
        """Verify CORS headers are returned for allowed origins."""
        origin = "http://localhost:3000"
        response = self.client.options(
            "/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access-control-allow-origin", response.headers)


if __name__ == "__main__":
    unittest.main()
