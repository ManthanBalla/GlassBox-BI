"""Unit Tests for Temporal Walk-Forward Splitting Utility."""

import unittest
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.processing import DataProcessingConfig
from backend.app.data_processing.processing.splitting import TimeSeriesSplitter


class TestTemporalSplitting(unittest.TestCase):
    """Test suite ensuring chronological walk-forward splitting integrity."""

    def setUp(self) -> None:
        self.splitter = TimeSeriesSplitter()
        self.mapping = ColumnMapping(
            date="date",
            entity_id="entity_id",
            target="target",
        )

    def test_chronological_split_boundaries(self) -> None:
        """Verify train, validation, and test partitions strictly follow chronological order."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "entity_id": ["E1"] * 100,
            "target": list(range(100)),
        })
        config = DataProcessingConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        train_df, val_df, test_df, meta = self.splitter.split(df, self.mapping, config)

        self.assertTrue(meta.temporal_order_verified)
        self.assertEqual(meta.train_rows, 70)
        self.assertEqual(meta.val_rows, 15)
        self.assertEqual(meta.test_rows, 15)

        # Strictly non-overlapping chronological bounds
        max_train_date = pd.to_datetime(train_df["date"]).max()
        min_val_date = pd.to_datetime(val_df["date"]).min()
        max_val_date = pd.to_datetime(val_df["date"]).max()
        min_test_date = pd.to_datetime(test_df["date"]).min()

        self.assertLess(max_train_date, min_val_date)
        self.assertLess(max_val_date, min_test_date)

    def test_no_random_shuffling(self) -> None:
        """Verify rows within partitions retain strict original chronological order."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "entity_id": ["E1"] * 20,
            "target": list(range(20)),
        })
        config = DataProcessingConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        train_df, val_df, test_df, _ = self.splitter.split(df, self.mapping, config)

        self.assertTrue(train_df["date"].is_monotonic_increasing)
        self.assertTrue(val_df["date"].is_monotonic_increasing)
        self.assertTrue(test_df["date"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
