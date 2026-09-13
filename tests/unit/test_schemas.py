"""Unit tests for Phase 1 Pydantic contract schemas."""

import unittest
from pydantic import ValidationError
from backend.app.schemas.contracts import (
    DatasetMetadata,
    ForecastRequest,
    ForecastResult,
    ExplanationResult,
    RecommendationResult,
)


class TestSchemas(unittest.TestCase):
    """Test suite for validating Pydantic contract schemas."""

    def test_dataset_metadata_valid(self):
        """Verify DatasetMetadata model validation."""
        meta = DatasetMetadata(
            dataset_id="ds_sales_001",
            name="Quarterly Retail Sales",
            row_count=1200,
            column_count=5,
            timestamp_column="date",
            target_column="sales",
            frequency="D",
        )
        self.assertEqual(meta.dataset_id, "ds_sales_001")
        self.assertEqual(meta.row_count, 1200)
        self.assertIsNotNone(meta.created_at)

    def test_forecast_request_constraints(self):
        """Verify ForecastRequest validation constraints."""
        req = ForecastRequest(
            dataset_id="ds_sales_001",
            target_column="sales",
            horizon=60,
            confidence_level=0.95,
        )
        self.assertEqual(req.horizon, 60)
        self.assertEqual(req.model_preference, "auto")

        # Negative or zero horizon must raise ValidationError
        with self.assertRaises(ValidationError):
            ForecastRequest(
                dataset_id="ds_sales_001",
                target_column="sales",
                horizon=0,
            )

    def test_forecast_result_serialization(self):
        """Verify ForecastResult serialization and structure."""
        res = ForecastResult(
            forecast_id="fc_001",
            dataset_id="ds_sales_001",
            horizon=14,
            model_name="BaselineAutoARIMA",
            predictions=[{"timestamp": "2026-10-01", "value": 150.5}],
            metrics={"MAE": 12.3, "RMSE": 15.1},
        )
        data = res.model_dump()
        self.assertEqual(data["forecast_id"], "fc_001")
        self.assertEqual(len(data["predictions"]), 1)
        self.assertEqual(data["metrics"]["MAE"], 12.3)

    def test_explanation_result_serialization(self):
        """Verify ExplanationResult structure."""
        exp = ExplanationResult(
            explanation_id="exp_001",
            forecast_id="fc_001",
            method="shap",
            feature_attributions={"price": -0.42, "marketing_spend": 0.65},
            narrative_summary="Marketing spend was the strongest positive driver.",
        )
        self.assertEqual(exp.feature_attributions["marketing_spend"], 0.65)
        self.assertIn("strongest positive driver", exp.narrative_summary)

    def test_recommendation_result_serialization(self):
        """Verify RecommendationResult structure."""
        rec = RecommendationResult(
            recommendation_id="rec_001",
            forecast_id="fc_001",
            scenario_name="Ad Budget +15%",
            action_items=["Increase digital marketing by 15%", "Reallocate regional budget"],
            risk_assessment="Low volatility expected",
        )
        self.assertEqual(len(rec.action_items), 2)
        self.assertEqual(rec.scenario_name, "Ad Budget +15%")


if __name__ == "__main__":
    unittest.main()
