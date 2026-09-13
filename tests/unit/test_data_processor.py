"""Unit Tests for GenericBusinessDataProcessor and Core Processing Logic."""

import unittest
import numpy as np
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping, SYNTHETIC_RETAIL_MAPPING
from backend.app.schemas.processing import (
    DataProcessingConfig,
    DuplicateStrategy,
    MissingCategoricalStrategy,
    MissingNumericStrategy,
    MissingTargetStrategy,
    OutlierMethod,
    OutlierStrategy,
)
from backend.app.data_processing.processing.processor import GenericBusinessDataProcessor


class TestGenericBusinessDataProcessor(unittest.TestCase):
    """Test suite verifying end-to-end data processing behaviors."""

    def setUp(self) -> None:
        self.processor = GenericBusinessDataProcessor()
        self.mapping = ColumnMapping(
            date="date",
            entity_id="entity_id",
            target="target",
            product_id="product_id",
            category="category",
            price="price",
            promotion="promotion",
            holiday="holiday",
            inventory="inventory",
        )

    def test_processor_initialization_and_default_config(self) -> None:
        """Verify processor initializes properly with sensible default configurations."""
        config = DataProcessingConfig()
        self.assertEqual(config.missing_numeric_strategy, MissingNumericStrategy.MEDIAN)
        self.assertEqual(config.missing_target_strategy, MissingTargetStrategy.DROP)
        self.assertEqual(config.duplicate_strategy, DuplicateStrategy.KEEP_FIRST)
        self.assertEqual(config.outlier_strategy, OutlierStrategy.FLAG)
        self.assertTrue(config.feature_engineering_enabled)
        self.assertTrue(config.calendar_features_enabled)
        self.assertTrue(config.lag_features_enabled)
        self.assertTrue(config.rolling_features_enabled)
        self.assertEqual(config.train_ratio, 0.70)
        self.assertEqual(config.val_ratio, 0.15)
        self.assertEqual(config.test_ratio, 0.15)

    def test_missing_numeric_imputation(self) -> None:
        """Verify missing numeric values are imputed according to strategy."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "entity_id": ["E1", "E1", "E1", "E1"],
            "product_id": ["P1", "P1", "P1", "P1"],
            "target": [10.0, 20.0, 30.0, 40.0],
            "price": [5.0, np.nan, 15.0, 20.0],  # Median is 15.0
            "inventory": [100.0, 80.0, np.nan, 50.0],
        })
        processed_df, result = self.processor.process(df, mapping=self.mapping)
        self.assertEqual(processed_df["price"].isna().sum(), 0)
        self.assertEqual(processed_df.loc[1, "price"], 15.0)
        self.assertEqual(result.missing_values_before.get("price"), 1)
        self.assertEqual(result.missing_values_after.get("price"), 0)

    def test_missing_categorical_imputation(self) -> None:
        """Verify missing categorical dimensions are converted to explicit Unknown."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "entity_id": ["E1", "E1"],
            "product_id": ["P1", "P1"],
            "category": [np.nan, "Beverages"],
            "target": [15.0, 25.0],
        })
        processed_df, _ = self.processor.process(df, mapping=self.mapping)
        self.assertEqual(processed_df.loc[0, "category"], "Unknown")
        self.assertEqual(processed_df.loc[1, "category"], "Beverages")

    def test_missing_target_safety(self) -> None:
        """Verify target values are NEVER fabricated; dropped by default under zero-leakage rule."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "entity_id": ["E1", "E1", "E1"],
            "product_id": ["P1", "P1", "P1"],
            "target": [10.0, np.nan, 30.0],
        })
        processed_df, result = self.processor.process(df, mapping=self.mapping)
        # Dropped row with missing target
        self.assertEqual(len(processed_df), 2)
        self.assertEqual(result.removed_rows_count, 1)
        self.assertEqual(list(processed_df["target"]), [10.0, 30.0])

    def test_duplicate_and_conflicting_record_handling(self) -> None:
        """Verify business key duplicates and conflicting records are resolved deterministically."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "entity_id": ["E1", "E1", "E1"],
            "product_id": ["P1", "P1", "P1"],
            "target": [100.0, 200.0, 150.0],  # Conflicting target on 2024-01-01
        })
        config = DataProcessingConfig(duplicate_strategy=DuplicateStrategy.KEEP_FIRST)
        processed_df, result = self.processor.process(df, mapping=self.mapping, config=config)
        self.assertEqual(len(processed_df), 2)
        self.assertEqual(result.duplicates_before, 1)
        self.assertEqual(result.conflicting_duplicates_count, 1)
        # First record (target=100.0) retained
        self.assertEqual(processed_df.iloc[0]["target"], 100.0)

    def test_invalid_target_and_negative_price_handling(self) -> None:
        """Verify negative target values and non-positive prices are corrected."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "entity_id": ["E1", "E1"],
            "product_id": ["P1", "P1"],
            "target": [-15.0, 25.0],  # Negative demand
            "price": [-5.0, 10.0],    # Negative price
            "inventory": [-20.0, 50.0], # Negative inventory
        })
        processed_df, result = self.processor.process(df, mapping=self.mapping)
        self.assertGreaterEqual(processed_df["target"].min(), 0.0)
        self.assertEqual(processed_df.iloc[0]["target"], 0.0)
        self.assertGreater(processed_df["price"].min(), 0.0)
        self.assertGreaterEqual(processed_df["inventory"].min(), 0.0)
        self.assertEqual(processed_df.iloc[0]["inventory"], 0.0)

    def test_outlier_flagging_without_deleting_business_spikes(self) -> None:
        """Verify outliers (e.g. genuine promotional spikes) are flagged, NOT deleted by default."""
        # 30 regular days, then 1 massive promo spike
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        targets = [10.0] * 29 + [500.0]  # Outlier
        df = pd.DataFrame({
            "date": dates,
            "entity_id": ["E1"] * 30,
            "product_id": ["P1"] * 30,
            "target": targets,
            "promotion": [0] * 29 + [1],
        })
        processed_df, result = self.processor.process(df, mapping=self.mapping)
        # Preserved all 30 rows!
        self.assertEqual(len(processed_df), 30)
        self.assertIn("target_outlier", processed_df.columns)
        self.assertEqual(processed_df["target_outlier"].iloc[-1], 1)
        self.assertEqual(result.outlier_summary.total_outliers, 1)
        self.assertEqual(result.outlier_summary.strategy, "flag")

    def test_chronological_sorting_and_gap_analysis(self) -> None:
        """Verify data is sorted chronologically and time-series gaps are identified."""
        df = pd.DataFrame({
            "date": ["2024-01-10", "2024-01-01", "2024-01-02", "2024-01-08"],
            "entity_id": ["E1", "E1", "E1", "E1"],
            "product_id": ["P1", "P1", "P1", "P1"],
            "target": [50.0, 10.0, 20.0, 40.0],
        })
        processed_df, result = self.processor.process(df, mapping=self.mapping)
        # Strict chronological order
        dates_list = [str(d.date()) if hasattr(d, "date") else str(d)[:10] for d in processed_df["date"]]
        self.assertEqual(dates_list, ["2024-01-01", "2024-01-02", "2024-01-08", "2024-01-10"])
        # Gap report
        self.assertEqual(result.time_series_integrity.series_with_gaps, 1)
        self.assertGreater(result.time_series_integrity.largest_gap_periods, 1)

    def test_audit_trail_generation(self) -> None:
        """Verify every transformation generates a transparent audit trail entry."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "entity_id": ["E1", "E1"],
            "product_id": ["P1", "P1"],
            "target": [10.0, 20.0],
        })
        _, result = self.processor.process(df, mapping=self.mapping)
        self.assertGreater(len(result.audit_trail), 0)
        operations = [entry.operation for entry in result.audit_trail]
        self.assertIn("date_standardization", operations)
        self.assertIn("chronological_sorting", operations)
        self.assertIn("calendar_feature_engineering", operations)

    def test_before_and_after_quality_score(self) -> None:
        """Verify before and after quality scores are computed and honest."""
        # Create dataset with duplicate and invalid value
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "entity_id": ["E1", "E1", "E1"],
            "product_id": ["P1", "P1", "P1"],
            "target": [-10.0, 20.0, 30.0],  # Invalid target & duplicate
        })
        _, result = self.processor.process(df, mapping=self.mapping)
        self.assertIsNotNone(result.quality_score_before)
        self.assertIsNotNone(result.quality_score_after)
        self.assertGreaterEqual(result.quality_score_after.overall_score, 0.0)
        self.assertLessEqual(result.quality_score_after.overall_score, 100.0)

    def test_reproducibility(self) -> None:
        """Verify identical inputs produce strictly identical outputs and metadata."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "entity_id": ["E1", "E1", "E1"],
            "product_id": ["P1", "P1", "P1"],
            "target": [15.0, 25.0, 35.0],
            "price": [10.0, 10.0, 12.0],
        })
        p1, res1 = self.processor.process(df, mapping=self.mapping)
        p2, res2 = self.processor.process(df, mapping=self.mapping)
        pd.testing.assert_frame_equal(p1, p2)
        self.assertEqual(res1.output_row_count, res2.output_row_count)
        self.assertEqual(res1.generated_features, res2.generated_features)


if __name__ == "__main__":
    unittest.main()
