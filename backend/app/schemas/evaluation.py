"""Pydantic Schemas and Data Contracts for Formal Forecasting Evaluation (Phase 5).

Defines structured contracts for test-set evaluation metrics (MAE, RMSE, MAPE),
point-by-point ground truth comparisons, candidate model benchmark outcomes,
and independent reporting schemas guaranteeing test-set isolation from model selection.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator

from backend.app.schemas.forecasting import ModelMetadata


class EvaluationMetricType(str, Enum):
    """Supported quantitative forecasting evaluation metrics."""
    MAE = "MAE"
    RMSE = "RMSE"
    MAPE = "MAPE"


class EvaluationMetricResult(BaseModel):
    """Formal quantitative error metrics computed against holdout test targets."""
    mae: float = Field(..., description="Mean Absolute Error: (1/n) * sum(|y - y_hat|)")
    rmse: float = Field(..., description="Root Mean Squared Error: sqrt((1/n) * sum((y - y_hat)^2))")
    mape: Optional[float] = Field(
        default=None,
        description="Mean Absolute Percentage Error on non-zero actual targets: (100/n) * sum(|y - y_hat| / |y|)",
    )
    total_observations: int = Field(..., ge=0, description="Total number of evaluated test observations")
    evaluated_observations: int = Field(..., ge=0, description="Observations included in calculation")
    zero_target_count: int = Field(
        default=0,
        ge=0,
        description="Number of actual target observations equal to 0 excluded from MAPE to prevent division by zero",
    )
    zero_handling_strategy: str = Field(
        default="exclude_zeros_from_mape",
        description="Transparent policy documenting zero target handling in MAPE",
    )
    metric_version: str = Field(default="1.0.0", description="Evaluation metric specification version")


class EvaluationPoint(BaseModel):
    """Single-period evaluation point contrasting ground-truth actual vs model prediction."""
    date: str = Field(..., description="Observation date (YYYY-MM-DD)")
    actual: float = Field(..., description="Ground-truth actual target value in test partition")
    prediction: float = Field(..., description="Model projected point estimate")
    lower_bound: Optional[float] = Field(default=None, description="Lower prediction interval bound")
    upper_bound: Optional[float] = Field(default=None, description="Upper prediction interval bound")
    absolute_error: float = Field(..., description="|actual - prediction|")
    percentage_error: Optional[float] = Field(
        default=None,
        description="Percentage error (|actual - prediction| / |actual| * 100), null if actual == 0",
    )


class ModelTestEvaluation(BaseModel):
    """Formal test-set benchmark evaluation result for a single forecasting model architecture."""
    model_name: str = Field(..., description="Model identifier (e.g. prophet, lightgbm, lstm)")
    model_type: Optional[str] = Field(default="forecaster", description="Model architecture family")
    status: str = Field(default="SUCCESS", description="SUCCESS, FAILED, or INSUFFICIENT_TEST_HISTORY")
    error_message: Optional[str] = Field(default=None, description="Failure details if evaluation unsuccessful")
    metrics: Optional[EvaluationMetricResult] = Field(default=None, description="Computed quantitative test metrics")
    mae: Optional[float] = Field(default=None, description="Test-set MAE")
    rmse: Optional[float] = Field(default=None, description="Test-set RMSE")
    mape: Optional[float] = Field(default=None, description="Test-set MAPE (%)")
    test_start_date: Optional[str] = Field(default=None, description="Start date of evaluated test partition")
    test_end_date: Optional[str] = Field(default=None, description="End date of evaluated test partition")
    horizon: int = Field(..., ge=1, description="Evaluated forecast horizon in steps/days")
    predictions: List[EvaluationPoint] = Field(
        default_factory=list,
        description="Detailed point-by-point actual vs prediction comparison",
    )
    evaluation_duration_seconds: float = Field(default=0.0, ge=0.0, description="Evaluation run duration in seconds")
    model_metadata: Optional[ModelMetadata] = Field(
        default=None,
        description="Model hyperparameters and configuration metadata",
    )
    is_validation_winner: bool = Field(
        default=False,
        description="Indicates whether this model was selected in Phase 4 via validation performance",
    )
    validation_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Validation partition metrics from Phase 4 for comparative reference",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_metrics_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            m = data.get("metrics")
            if isinstance(m, dict):
                if data.get("mae") is None:
                    data["mae"] = m.get("mae")
                if data.get("rmse") is None:
                    data["rmse"] = m.get("rmse")
                if data.get("mape") is None:
                    data["mape"] = m.get("mape")
            elif hasattr(m, "mae"):
                if data.get("mae") is None:
                    data["mae"] = m.mae
                if data.get("rmse") is None:
                    data["rmse"] = m.rmse
                if data.get("mape") is None:
                    data["mape"] = m.mape
        return data


class EvaluationRequest(BaseModel):
    """Request contract for executing a formal forecasting benchmark evaluation."""
    dataset_id: str = Field(default="retail_dataset", description="Dataset identifier")
    dataset_path: Optional[str] = Field(default=None, description="Local path to processed or sample dataset")
    entity_id: Optional[str] = Field(default=None, description="Entity/Store identifier (e.g. STORE_001)")
    product_id: Optional[str] = Field(default=None, description="Product/SKU identifier (e.g. PROD_001)")
    target_column: str = Field(default="target", description="Target metric column")
    date_column: str = Field(default="date", description="Date column")
    horizon: int = Field(default=14, ge=1, le=365, description="Evaluation horizon in days (e.g. 7, 14, 28)")
    candidate_models: List[str] = Field(
        default_factory=lambda: ["prophet", "lightgbm", "lstm"],
        description="Candidate models to benchmark: prophet, lightgbm, lstm",
    )
    confidence_level: float = Field(default=0.80, ge=0.50, le=0.99, description="Prediction interval level")
    random_seed: int = Field(default=42, description="Random seed for deterministic reproducibility")


class BenchmarkResult(BaseModel):
    """Complete machine-readable formal benchmark report across candidate forecasting models.

    Strict Architectural Separation Invariant:
    - `phase4_selected_model` records the model selected strictly via VALIDATION in Phase 4.
    - `test_set_used_for_selection` is FIXED to FALSE.
    - Test-set metrics are reported independently and never modify model selection.
    """
    evaluation_id: str = Field(..., description="Unique evaluation task identifier")
    dataset_id: str = Field(default="retail_dataset", description="Evaluated dataset")
    entity_id: Optional[str] = Field(default=None, description="Evaluated entity identifier")
    product_id: Optional[str] = Field(default=None, description="Evaluated product identifier")
    status: str = Field(default="SUCCESS", description="SUCCESS, PARTIAL_SUCCESS, or FAILED")
    horizon: int = Field(..., description="Evaluated forecast horizon in days")
    test_period: Dict[str, Any] = Field(
        ...,
        description="Chronological boundary of holdout test set (start, end, available_days)",
    )
    training_period: Dict[str, Any] = Field(
        ...,
        description="Chronological boundaries of historical training and validation sets",
    )
    data_split_info: Dict[str, Any] = Field(
        ...,
        description="Data partition breakdown: train_rows, val_rows, test_rows, and dates",
    )
    models: List[ModelTestEvaluation] = Field(
        default_factory=list,
        description="Test set evaluations across candidate models",
    )
    phase4_selected_model: str = Field(
        ...,
        description="Model architecture selected in Phase 4 using validation metrics only",
    )
    phase4_selection_metric: str = Field(default="MAE", description="Validation metric used in Phase 4")
    phase4_validation_score: Optional[float] = Field(
        default=None,
        description="Validation score of the Phase 4 selected model",
    )
    selection_source: str = Field(
        default="validation",
        description="Formal declaration that selection source was validation, not test",
    )
    test_set_used_for_selection: bool = Field(
        default=False,
        description="Strict guarantee: test set was NEVER used for model selection or tuning",
    )
    test_winner_for_reporting_only: Optional[str] = Field(
        default=None,
        description="Model with best test MAE reported strictly for quantitative academic comparison",
    )
    benchmark_summary_markdown: Optional[str] = Field(
        default=None,
        description="Formatted ASCII/Markdown benchmark comparison table",
    )
    reproducibility: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reproducibility audit metadata including random seed and library versions",
    )
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings emitted during evaluation")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
