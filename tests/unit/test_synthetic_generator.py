"""Unit tests for the synthetic retail data generator."""

import unittest
import pandas as pd
from scripts.generate_synthetic_retail_data import generate_retail_dataset


class TestSyntheticRetailGenerator(unittest.TestCase):
    """Test suite verifying the deterministic synthetic data generation logic."""

    def test_reproducibility_with_seed(self):
        """Verify identical seed produces identical datasets."""
        df1 = generate_retail_dataset(num_rows=200, seed=42)
        df2 = generate_retail_dataset(num_rows=200, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        """Verify different seeds produce varying noise/targets."""
        df1 = generate_retail_dataset(num_rows=200, seed=42)
        df2 = generate_retail_dataset(num_rows=200, seed=99)
        self.assertFalse(df1["target"].equals(df2["target"]))

    def test_exact_row_count_generation(self):
        """Verify generator respects exact row count argument."""
        for n in [50, 200, 1000]:
            df = generate_retail_dataset(num_rows=n, seed=42)
            self.assertEqual(len(df), n)

    def test_required_columns_exist(self):
        """Verify all generic organization-agnostic columns are present."""
        df = generate_retail_dataset(num_rows=100, seed=42)
        expected_cols = [
            "date",
            "entity_id",
            "product_id",
            "category",
            "region",
            "store_type",
            "target",
            "price",
            "promotion",
            "holiday",
            "inventory",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_target_values_non_negative(self):
        """Verify target demand is never negative."""
        df = generate_retail_dataset(num_rows=1000, seed=42)
        self.assertTrue((df["target"] >= 0).all())
        self.assertTrue(df["target"].dtype.kind in ("i", "f"))

    def test_numeric_ranges_and_binary_flags(self):
        """Verify prices are positive and indicator flags are binary."""
        df = generate_retail_dataset(num_rows=500, seed=42)
        self.assertTrue((df["price"] > 0).all())
        self.assertTrue((df["inventory"] >= 0).all())
        self.assertTrue(set(df["promotion"].unique()).issubset({0, 1}))
        self.assertTrue(set(df["holiday"].unique()).issubset({0, 1}))

    def test_dates_are_valid_and_parseable(self):
        """Verify dates parse cleanly to datetime."""
        df = generate_retail_dataset(num_rows=300, seed=42)
        parsed_dates = pd.to_datetime(df["date"], errors="coerce")
        self.assertEqual(parsed_dates.isna().sum(), 0)


if __name__ == "__main__":
    unittest.main()
