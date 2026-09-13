"""Unit Tests for Feature Engineering and Strict Zero-Leakage Verification."""

import unittest
import numpy as np
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.processing import DataProcessingConfig
from backend.app.data_processing.processing.audit import AuditTrailTracker
from backend.app.data_processing.processing.feature_engineering import FeatureEngineer


class TestFeatureEngineering(unittest.TestCase):
    """Test suite ensuring feature correctness and zero lookahead leakage."""

    def setUp(self) -> None:
        self.engineer = FeatureEngineer()
        self.audit = AuditTrailTracker()
        self.mapping = ColumnMapping(
            date="date",
            entity_id="entity_id",
            target="target",
            product_id="product_id",
            price="price",
            promotion="promotion",
            holiday="holiday",
            inventory="inventory",
        )

    def test_calendar_feature_engineering(self) -> None:
        """Verify calendar components are correctly derived from observation dates."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"]),
            "entity_id": ["E1", "E1", "E1"],
            "target": [10.0, 20.0, 30.0],
        })
        config = DataProcessingConfig(
            calendar_features_enabled=True,
            lag_features_enabled=False,
            rolling_features_enabled=False,
            business_features_enabled=False,
        )
        fe_df, features = self.engineer.engineer_features(df, self.mapping, config, self.audit)
        
        self.assertIn("year", fe_df.columns)
        self.assertIn("month", fe_df.columns)
        self.assertIn("day_of_week", fe_df.columns)
        self.assertIn("is_weekend", fe_df.columns)

        # 2024-01-01 was Monday (day_of_week = 0, is_weekend = 0)
        self.assertEqual(fe_df.loc[0, "day_of_week"], 0)
        self.assertEqual(fe_df.loc[0, "is_weekend"], 0)
        self.assertEqual(fe_df.loc[0, "month"], 1)

        # 2024-06-15 was Saturday (day_of_week = 5, is_weekend = 1)
        self.assertEqual(fe_df.loc[1, "day_of_week"], 5)
        self.assertEqual(fe_df.loc[1, "is_weekend"], 1)
        self.assertEqual(fe_df.loc[1, "month"], 6)

    def test_lag_feature_correctness_and_zero_leakage(self) -> None:
        """Verify lag features strictly use historical values and contain NO lookahead data."""
        # Simple sequence for an entity
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "entity_id": ["STORE_A"] * 10,
            "target": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        })
        config = DataProcessingConfig(
            calendar_features_enabled=False,
            lag_features_enabled=True,
            lags=[1, 7],
            rolling_features_enabled=False,
            business_features_enabled=False,
        )
        fe_df, _ = self.engineer.engineer_features(df, self.mapping, config, self.audit)

        # At t=0 (first observation), lag_1 must be NaN (no history exists)
        self.assertTrue(pd.isna(fe_df.loc[0, "lag_1"]))
        # At t=1 (target=20.0), lag_1 must be 10.0 (past value)
        self.assertEqual(fe_df.loc[1, "lag_1"], 10.0)
        # At t=2 (target=30.0), lag_1 must be 20.0
        self.assertEqual(fe_df.loc[2, "lag_1"], 20.0)
        # At t=7 (target=80.0), lag_7 must be 10.0
        self.assertEqual(fe_df.loc[7, "lag_7"], 10.0)

    def test_rolling_feature_shift1_zero_leakage_guarantee(self) -> None:
        """CRITICAL: Verify rolling statistics use shift(1) so current observation T is excluded."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "entity_id": ["STORE_A"] * 5,
            "target": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        config = DataProcessingConfig(
            calendar_features_enabled=False,
            lag_features_enabled=False,
            rolling_features_enabled=True,
            rolling_windows=[3],
            business_features_enabled=False,
        )
        fe_df, _ = self.engineer.engineer_features(df, self.mapping, config, self.audit)

        # At t=0 (target=10.0): rolling_mean_3 must be NaN because shift(1) has no prior history
        self.assertTrue(pd.isna(fe_df.loc[0, "rolling_mean_3"]))

        # At t=1 (target=20.0): prior history is [10.0]. rolling_mean_3 must be 10.0 (NOT 15.0!)
        # If it were 15.0, current observation 20.0 leaked into the rolling window!
        self.assertEqual(fe_df.loc[1, "rolling_mean_3"], 10.0)

        # At t=2 (target=30.0): prior history is [10.0, 20.0]. Mean is (10+20)/2 = 15.0 (NOT 20.0!)
        self.assertEqual(fe_df.loc[2, "rolling_mean_3"], 15.0)

        # At t=3 (target=40.0): prior history is [10.0, 20.0, 30.0]. Mean is (10+20+30)/3 = 20.0
        self.assertEqual(fe_df.loc[3, "rolling_mean_3"], 20.0)

    def test_optional_columns_absent_graceful_handling(self) -> None:
        """Verify feature engineering executes cleanly when optional business features are absent."""
        # Dataset with ONLY required core fields
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "entity_id": ["STORE_A"] * 10,
            "target": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        })
        sparse_mapping = ColumnMapping(
            date="date",
            entity_id="entity_id",
            target="target",
            product_id=None,
            price=None,
            promotion=None,
            holiday=None,
            inventory=None,
        )
        config = DataProcessingConfig()
        fe_df, features = self.engineer.engineer_features(df, sparse_mapping, config, self.audit)

        # Successfully generated calendar, lag, and rolling features
        self.assertIn("year", fe_df.columns)
        self.assertIn("lag_1", fe_df.columns)
        self.assertIn("rolling_mean_7", fe_df.columns)
        # Did not crash on absent price or inventory
        self.assertNotIn("price_change_pct", fe_df.columns)
        self.assertNotIn("inventory_pressure", fe_df.columns)


if __name__ == "__main__":
    unittest.main()
