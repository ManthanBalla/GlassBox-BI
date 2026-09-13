"""Unit tests for the Data Validation and Temporal Leakage engines."""

import unittest
import pandas as pd
from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.ingestion import ValidationSeverity
from backend.app.data_processing.validation import DataValidator
from backend.app.data_processing.leakage import TemporalLeakageDetector


class TestDataValidation(unittest.TestCase):
    """Test suite verifying multi-dimensional data validation logic."""

    def setUp(self):
        self.validator = DataValidator()
        self.mapping = ColumnMapping(
            date="date",
            entity_id="entity_id",
            target="target",
            product_id="product_id",
            price="price",
            inventory="inventory",
        )

    def test_valid_dataframe_passes(self):
        """Verify clean data produces zero critical errors."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 5,
            "entity_id": ["STORE_001"] * 15,
            "target": [10.0, 15.0, 20.0] * 5,
            "price": [5.0] * 15,
            "inventory": [100] * 15,
        })
        errors, warnings = self.validator.validate(df, self.mapping)
        self.assertEqual(len(errors), 0)

    def test_missing_required_column_raises_error(self):
        """Verify missing target or date raises ValidationSeverity.ERROR."""
        df = pd.DataFrame({
            "entity_id": ["STORE_001"],
            "price": [5.0],
        })
        errors, warnings = self.validator.validate(df, self.mapping)
        self.assertTrue(any(e.issue_type == "MISSING_REQUIRED_COLUMN" for e in errors))

    def test_nulls_in_required_column(self):
        """Verify nulls in required target column raise ERROR."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "entity_id": ["STORE_001", "STORE_001"],
            "target": [10.0, None],
        })
        errors, warnings = self.validator.validate(df, self.mapping)
        self.assertTrue(any(e.issue_type == "NULL_IN_REQUIRED_COLUMN" for e in errors))

    def test_negative_target_raises_error(self):
        """Verify negative demand values are flagged as ERROR."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "entity_id": ["STORE_001", "STORE_001"],
            "target": [25.0, -5.0],
        })
        errors, warnings = self.validator.validate(df, self.mapping)
        self.assertTrue(any(e.issue_type == "NEGATIVE_TARGET_VALUES" for e in errors))

    def test_duplicate_detection(self):
        """Verify exact duplicate rows produce WARNING."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01"],
            "entity_id": ["STORE_001", "STORE_001"],
            "target": [10.0, 10.0],
            "price": [5.0, 5.0],
        })
        errors, warnings = self.validator.validate(df, self.mapping)
        self.assertTrue(any(w.issue_type == "EXACT_DUPLICATE_ROWS" for w in warnings))

    def test_unparseable_dates_raises_error(self):
        """Verify invalid date strings raise ERROR."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "NOT_A_DATE"],
            "entity_id": ["STORE_001", "STORE_001"],
            "target": [10.0, 15.0],
        })
        errors, warnings = self.validator.validate(df, self.mapping)
        self.assertTrue(any(e.issue_type == "UNPARSEABLE_DATES" for e in errors))


class TestTemporalLeakageDetection(unittest.TestCase):
    """Test suite verifying temporal lookahead leakage detection."""

    def setUp(self):
        self.detector = TemporalLeakageDetector()
        self.mapping = ColumnMapping(date="date", entity_id="entity_id", target="target")

    def test_future_token_in_column_name_flagged(self):
        """Verify columns containing future lookahead keywords are flagged."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "entity_id": ["S1", "S1"],
            "target": [100.0, 120.0],
            "future_sales_t+1": [120.0, 115.0],
        })
        report = self.detector.detect(df, self.mapping)
        self.assertTrue(report.has_leakage)
        self.assertIn("future_sales_t+1", report.flagged_columns)
        self.assertIn("chronological walk-forward", report.splitting_recommendation)

    def test_identical_target_under_different_name_flagged(self):
        """Verify exact copy of target under different name is flagged as critical leakage."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "entity_id": ["S1", "S1"],
            "target": [100.0, 120.0],
            "leaked_var": [100.0, 120.0],
        })
        report = self.detector.detect(df, self.mapping)
        self.assertTrue(report.has_leakage)
        self.assertIn("leaked_var", report.flagged_columns)

    def test_clean_data_has_no_leakage(self):
        """Verify standard historical features do not trigger false positives."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "entity_id": ["S1", "S1"],
            "target": [100.0, 120.0],
            "price": [9.99, 9.99],
            "promotion": [0, 1],
        })
        report = self.detector.detect(df, self.mapping)
        self.assertFalse(report.has_leakage)
        self.assertEqual(len(report.flagged_columns), 0)


if __name__ == "__main__":
    unittest.main()
