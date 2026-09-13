"""Unit tests for the Dataset Ingestion Service, Quality Scorer, Profiler, and Contracts."""

import io
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
import pandas as pd
from pydantic import ValidationError

from backend.app.main import app
from backend.app.schemas.data_contract import (
    BusinessTimeSeriesRecord,
    ColumnMapping,
    WALMART_MAPPING_TEMPLATE,
    ROSSMANN_MAPPING_TEMPLATE,
    auto_detect_column_mapping,
)
from backend.app.data_processing.ingestion import DatasetIngestionService
from backend.app.data_processing.quality import DataQualityScorer
from backend.app.data_processing.profiling import DataProfiler
from backend.app.data_processing.validation import DataValidator
from backend.app.data_processing.leakage import TemporalLeakageDetector


class TestDataContractAndMapping(unittest.TestCase):
    """Test suite for canonical contracts and dataset column mapping."""

    def test_canonical_record_valid(self):
        """Verify BusinessTimeSeriesRecord instantiation."""
        rec = BusinessTimeSeriesRecord(
            date="2024-01-15",
            entity_id="STORE_001",
            target=150.0,
            product_id="PROD_010",
            category="Snacks",
            price=3.99,
            promotion=1,
            holiday=0,
            inventory=250.0,
        )
        self.assertEqual(rec.entity_id, "STORE_001")
        self.assertEqual(rec.target, 150.0)
        self.assertEqual(str(rec.date), "2024-01-15")

    def test_canonical_record_rejects_negative_target(self):
        """Verify BusinessTimeSeriesRecord rejects negative target."""
        with self.assertRaises(ValidationError):
            BusinessTimeSeriesRecord(
                date="2024-01-15",
                entity_id="STORE_001",
                target=-25.0,
            )

    def test_walmart_and_rossmann_templates(self):
        """Verify pre-configured benchmark templates map expected columns."""
        self.assertEqual(WALMART_MAPPING_TEMPLATE.date, "Date")
        self.assertEqual(WALMART_MAPPING_TEMPLATE.entity_id, "Store")
        self.assertEqual(WALMART_MAPPING_TEMPLATE.target, "Weekly_Sales")
        self.assertEqual(WALMART_MAPPING_TEMPLATE.holiday, "IsHoliday")

        self.assertEqual(ROSSMANN_MAPPING_TEMPLATE.date, "Date")
        self.assertEqual(ROSSMANN_MAPPING_TEMPLATE.entity_id, "Store")
        self.assertEqual(ROSSMANN_MAPPING_TEMPLATE.target, "Sales")

    def test_auto_detect_column_mapping(self):
        """Verify heuristic synonym detection maps date, target, and entity."""
        cols = ["timestamp", "shop_id", "item_sku", "weekly_sales", "unit_price"]
        mapping = auto_detect_column_mapping(cols)
        self.assertEqual(mapping.date, "timestamp")
        self.assertEqual(mapping.entity_id, "shop_id")
        self.assertEqual(mapping.product_id, "item_sku")
        self.assertEqual(mapping.target, "weekly_sales")
        self.assertEqual(mapping.price, "unit_price")


class TestDataQualityAndProfiling(unittest.TestCase):
    """Test suite verifying explainable quality scores and profiling statistics."""

    def test_quality_scorer_on_clean_data(self):
        """Verify high quality score (>90, Grade A/A+) on clean data."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D").astype(str),
            "entity_id": ["STORE_001"] * 30,
            "target": [20.0 + i for i in range(30)],
            "price": [4.99] * 30,
        })
        mapping = ColumnMapping(date="date", entity_id="entity_id", target="target", price="price")
        scorer = DataQualityScorer()
        score = scorer.score(df, mapping, validation_issues=[], leakage_report=TemporalLeakageDetector().detect(df, mapping))
        self.assertGreaterEqual(score.overall_score, 90.0)
        self.assertIn(score.grade, ("A", "A+"))
        self.assertEqual(score.schema_score, 100.0)

    def test_profiler_statistics_and_frequency(self):
        """Verify profiler calculates min, max, mean, and infers daily frequency."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=20, freq="D").astype(str),
            "entity_id": ["STORE_001"] * 10 + ["STORE_002"] * 10,
            "target": [10.0] * 19 + [200.0],  # Single outlier
        })
        mapping = ColumnMapping(date="date", entity_id="entity_id", target="target")
        profiler = DataProfiler()
        profile = profiler.profile(df, mapping)

        self.assertEqual(profile.row_count, 20)
        self.assertEqual(profile.unique_entities, 2)
        self.assertEqual(profile.date_frequency, "Daily (D)")
        self.assertIsNotNone(profile.target_summary)
        self.assertEqual(profile.target_summary.min, 10.0)
        self.assertEqual(profile.target_summary.max, 200.0)


class TestDatasetIngestionService(unittest.TestCase):
    """Test suite for the end-to-end dataset ingestion service and endpoints."""

    def setUp(self):
        self.service = DatasetIngestionService()
        self.client = TestClient(app)

    def test_ingest_sample_csv_file(self):
        """Verify ingestion of committed retail sample file."""
        sample_path = Path("data/sample/retail_sample.csv")
        if sample_path.exists():
            result = self.service.ingest_csv(sample_path)
            self.assertIn(result.status, ("SUCCESS", "WARNING"))
            self.assertEqual(result.metadata.row_count, 150)
            self.assertGreater(result.quality_score.overall_score, 85.0)
            self.assertGreater(len(result.canonical_sample), 0)

    def test_ingest_from_bytes_buffer(self):
        """Verify ingestion from in-memory CSV buffer."""
        csv_data = (
            "date,entity_id,product_id,target,price\n"
            "2024-01-01,S01,P01,50.0,9.99\n"
            "2024-01-02,S01,P01,55.0,9.99\n"
            "2024-01-03,S01,P01,60.0,9.99\n"
        ).encode("utf-8")

        result = self.service.ingest_csv(csv_data, dataset_name="Buffer_Test")
        self.assertIn(result.status, ("SUCCESS", "WARNING"))
        self.assertEqual(result.metadata.row_count, 3)
        self.assertEqual(result.metadata.column_count, 5)

    def test_missing_file_fails_gracefully(self):
        """Verify non-existent filepath returns FAILED result without crash."""
        result = self.service.ingest_csv("non_existent_file.csv")
        self.assertEqual(result.status, "FAILED")
        self.assertGreater(len(result.validation_errors), 0)
        self.assertEqual(result.validation_errors[0].issue_type, "INGESTION_LOAD_FAILED")

    def test_api_sample_summary_endpoint(self):
        """Verify GET /api/v1/datasets/sample-summary."""
        response = self.client.get("/api/v1/datasets/sample-summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("metadata", data)
        self.assertEqual(data["metadata"]["row_count"], 150)
        self.assertIn("quality_score", data)

    def test_api_mapping_templates_endpoint(self):
        """Verify GET /api/v1/datasets/mapping-templates."""
        response = self.client.get("/api/v1/datasets/mapping-templates")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("templates", data)
        self.assertIn("walmart_benchmark", data["templates"])
        self.assertIn("synthetic_retail", data["templates"])


if __name__ == "__main__":
    unittest.main()
