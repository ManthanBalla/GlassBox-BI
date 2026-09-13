"""Unit Tests for Processing API Endpoints (Phase 3)."""

import unittest
from fastapi.testclient import TestClient

from backend.app.main import app


class TestProcessingApi(unittest.TestCase):
    """Test suite verifying REST API contracts for Data Processing Agent."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_get_processing_config(self) -> None:
        """Verify default processing config endpoint."""
        response = self.client.get("/api/v1/process/config")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["missing_numeric_strategy"], "median")
        self.assertEqual(data["outlier_strategy"], "flag")
        self.assertTrue(data["feature_engineering_enabled"])

    def test_get_sample_processing_summary(self) -> None:
        """Verify processing endpoint on committed retail sample."""
        response = self.client.get("/api/v1/process/sample-summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["input_row_count"], 150)
        self.assertGreater(data["output_column_count"], data["input_column_count"])
        self.assertIn("temporal_split", data)
        self.assertIn("quality_score_after", data)

    def test_post_process_local_file(self) -> None:
        """Verify processing local CSV file via API."""
        payload = {
            "file_path": "data/sample/retail_sample.csv",
            "dataset_name": "Test_Local_Process",
        }
        response = self.client.post("/api/v1/process/local", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["dataset_name"], "Test_Local_Process")
        self.assertIn("generated_features", data)

    def test_post_process_local_file_not_found(self) -> None:
        """Verify 404 response on missing local dataset."""
        payload = {
            "file_path": "data/non_existent_file.csv",
        }
        response = self.client.post("/api/v1/process/local", json=payload)
        self.assertEqual(response.status_code, 404)

    def test_post_process_file_upload(self) -> None:
        """Verify processing dataset via multipart file upload."""
        csv_content = (
            "date,entity_id,product_id,target,price,promotion\n"
            "2024-01-01,E1,P1,10.0,5.0,0\n"
            "2024-01-02,E1,P1,20.0,5.0,1\n"
            "2024-01-03,E1,P1,30.0,5.5,0\n"
        )
        files = {
            "file": ("test_upload.csv", csv_content.encode("utf-8"), "text/csv")
        }
        response = self.client.post("/api/v1/process/file", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["input_row_count"], 3)
        self.assertIn("lag_1", data["generated_features"])


if __name__ == "__main__":
    unittest.main()
