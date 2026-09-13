"""Pydantic Schemas and Configuration for Forecasting Agent (Phase 4).

Defines configuration, model-agnostic forecast requests, point predictions,
uncertainty prediction intervals, candidate evaluation scores, and machine-readable
ForecastResult contracts.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class ForecastModelType(str, Enum):
    """Supported candidate forecasting model architectures."""
    PROPHET = "prophet"
    LIGHTGBM = "lightgbm"
    LSTM = "lstm"
    AUTO = "auto"


class SelectionMetric(str, Enum):
    """Validation metrics used for deterministic model selection."""
    MAE = "MAE"
    RMSE = "RMSE"
    MAPE = "MAPE"


class ForecastStatus(str, Enum):
    """Execution status for candidate models and forecasting agent runs."""
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    NO_VALID_MODEL = "NO_VALID_MODEL"
    FAILED = "FAILED"


class ForecastingConfig(BaseModel):
    """Configuration contract for model hyperparameters and selection rules."""
    
    candidate_models: List[str] = Field(
        default_factory=lambda: ["prophet", "lightgbm", "lstm"],
        description="Candidate models to train and evaluate on validation data",
    )
    forecast_horizon: int = Field(
        default=14,
        ge=1,
        le=365,
        description="Number of future time steps to project",
    )
    selection_metric: SelectionMetric = Field(
        default=SelectionMetric.MAE,
        description="Primary validation error metric for selecting the winning model",
    )
    model_selection_metric: Optional[SelectionMetric] = Field(
        default=SelectionMetric.MAE,
        description="Alias for selection_metric",
    )
    confidence_level: float = Field(
        default=0.80,
        ge=0.50,
        le=0.99,
        description="Confidence level for prediction / uncertainty intervals",
    )
    train_ratio: float = Field(
        default=0.70,
        ge=0.10,
        le=0.90,
        description="Proportion of chronological timeline allocated for model fitting",
    )
    val_ratio: float = Field(
        default=0.15,
        ge=0.05,
        le=0.50,
        description="Proportion allocated for candidate model comparison and selection",
    )
    random_seed: int = Field(
        default=42,
        description="Fixed random seed for deterministic initialization",
    )

    # Model-Specific Hyperparameters
    # Prophet
    prophet_yearly_seasonality: Union[bool, str] = Field(default="auto")
    prophet_weekly_seasonality: Union[bool, str] = Field(default="auto")
    prophet_daily_seasonality: Union[bool, str] = Field(default=False)

    # LightGBM
    lgb_n_estimators: int = Field(default=100, ge=10, le=1000)
    lgb_learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    lgb_max_depth: int = Field(default=5, ge=1, le=15)
    lgb_num_leaves: int = Field(default=31, ge=4, le=256)

    # LSTM (PyTorch)
    lstm_lookback: int = Field(default=14, ge=3, le=60, description="Input sequence length in observation steps")
    lstm_hidden_dim: int = Field(default=32, ge=8, le=256)
    lstm_num_layers: int = Field(default=1, ge=1, le=4)
    lstm_dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    lstm_learning_rate: float = Field(default=0.01, gt=0.0, le=0.1)
    lstm_epochs: int = Field(default=20, ge=1, le=100)
    lstm_batch_size: int = Field(default=16, ge=1, le=256)
    lstm_patience: int = Field(default=5, ge=1, le=20, description="Early stopping patience")

    @model_validator(mode="before")
    @classmethod
    def sync_metric(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "model_selection_metric" in data and "selection_metric" not in data:
                data["selection_metric"] = data["model_selection_metric"]
            elif "selection_metric" in data and ("model_selection_metric" not in data or data["model_selection_metric"] is None):
                data["model_selection_metric"] = data["selection_metric"]
        return data


class ForecastPoint(BaseModel):
    """Single-period forecasted observation point with prediction bounds."""
    date: str = Field(..., description="Observation date (YYYY-MM-DD)")
    prediction: float = Field(..., description="Projected point estimate")
    lower_bound: Optional[float] = Field(default=None, description="Lower prediction interval bound")
    upper_bound: Optional[float] = Field(default=None, description="Upper prediction interval bound")

    @model_validator(mode="before")
    @classmethod
    def reconcile_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "date" not in data:
                if "timestamp" in data:
                    data["date"] = str(data["timestamp"])
                elif "ds" in data:
                    data["date"] = str(data["ds"])
            if "prediction" not in data:
                if "value" in data:
                    data["prediction"] = float(data["value"])
                elif "predicted_value" in data:
                    data["prediction"] = float(data["predicted_value"])
                elif "yhat" in data:
                    data["prediction"] = float(data["yhat"])
        return data


class ModelEvaluationScore(BaseModel):
    """Validation performance metrics for an evaluated candidate model."""
    model_name: str = Field(..., description="Candidate model identifier")
    model_type: Optional[str] = Field(default="forecaster", description="Architecture family")
    status: str = Field(default="SUCCESS", description="SUCCESS, FAILED, or INSUFFICIENT_HISTORY")
    mae: Optional[float] = Field(default=None, description="Mean Absolute Error on validation partition")
    rmse: Optional[float] = Field(default=None, description="Root Mean Squared Error on validation partition")
    mape: Optional[float] = Field(default=None, description="Mean Absolute Percentage Error on validation partition")
    training_duration_seconds: float = Field(default=0.0, ge=0.0)
    training_duration: float = Field(default=0.0, ge=0.0)
    error_message: Optional[str] = Field(default=None, description="Error reason if model evaluation failed")

    @model_validator(mode="before")
    @classmethod
    def sync_duration(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "training_duration_seconds" in data and "training_duration" not in data:
                data["training_duration"] = data["training_duration_seconds"]
            elif "training_duration" in data and "training_duration_seconds" not in data:
                data["training_duration_seconds"] = data["training_duration"]
        return data


class ModelSelectionResult(BaseModel):
    """Structured audit trail of model competition and ranking."""
    candidate_scores: List[ModelEvaluationScore] = Field(
        default_factory=list,
        description="Validation scores across all candidate architectures",
    )
    candidate_rankings: List[ModelEvaluationScore] = Field(
        default_factory=list,
        description="Alias for candidate_scores",
    )
    ranking: List[str] = Field(
        default_factory=list,
        description="Ranked list of candidate models from best to worst",
    )
    selected_model: str = Field(..., description="Chosen winning model architecture")
    selection_metric: str = Field(default="MAE", description="Metric used for ranking")
    selection_reason: str = Field(..., description="Deterministic justification for selection")

    @model_validator(mode="before")
    @classmethod
    def sync_rankings(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "candidate_scores" in data and ("candidate_rankings" not in data or not data["candidate_rankings"]):
                data["candidate_rankings"] = data["candidate_scores"]
            elif "candidate_rankings" in data and ("candidate_scores" not in data or not data["candidate_scores"]):
                data["candidate_scores"] = data["candidate_rankings"]
        return data


class ModelMetadata(BaseModel):
    """Architectural and operational metadata for a fitted model."""
    model_name: str
    model_type: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    training_rows: int = 0
    training_start: Optional[str] = None
    training_end: Optional[str] = None
    feature_names: List[str] = Field(default_factory=list)
    lookback_window: Optional[int] = None
    random_seed: int = 42
    training_duration: float = 0.0
    uncertainty_method: Optional[str] = None
    software_versions: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ForecastRequest(BaseModel):
    """Universal contract for initiating a forecasting task."""
    dataset_id: str = Field(default="retail_dataset", description="Identifier of dataset to forecast")
    dataset_path: Optional[str] = Field(default=None, description="Local path to processed or sample dataset")
    entity_id: Optional[str] = Field(default=None, description="Business entity identifier (e.g. STORE_001)")
    product_id: Optional[str] = Field(default=None, description="Product / SKU identifier (e.g. PROD_001)")
    target_column: str = Field(default="target", description="Target metric column")
    date_column: str = Field(default="date", description="Observation date column")
    horizon: int = Field(default=14, ge=1, le=365, description="Number of future periods to project")
    candidate_models: List[str] = Field(
        default_factory=lambda: ["prophet", "lightgbm", "lstm"],
        description="Candidate models to evaluate: prophet, lightgbm, lstm",
    )
    selection_metric: str = Field(default="MAE", description="Validation metric: MAE, RMSE, MAPE")
    confidence_level: float = Field(default=0.80, ge=0.50, le=0.99, description="Prediction interval level")
    model_preference: Optional[str] = Field(default="auto", description="Model family preference: auto, statistical, ml")

    @model_validator(mode="before")
    @classmethod
    def reconcile_horizon(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Support forecast_horizon alias if provided
            if "forecast_horizon" in data and "horizon" not in data:
                data["horizon"] = data["forecast_horizon"]
        return data


class ForecastResult(BaseModel):
    """Complete machine-readable output contract for a forecasting run."""
    forecast_id: str
    dataset_id: str
    entity_id: Optional[str] = None
    product_id: Optional[str] = None
    horizon: int
    frequency: str = Field(default="D", description="Inferred or configured series frequency")
    model_name: str = Field(..., description="Winning / active model name")
    status: str = Field(default="SUCCESS", description="SUCCESS, PARTIAL_SUCCESS, or FAILED")
    predictions: List[ForecastPoint] = Field(default_factory=list, description="Typed forecast data points")
    forecast_points: List[ForecastPoint] = Field(default_factory=list, description="Typed forecast points")
    confidence_intervals: Optional[Dict[str, Any]] = Field(default=None, description="Upper/lower bounds dict")
    uncertainty_method: Optional[str] = Field(default=None, description="Method used to calculate intervals")
    confidence_level: Optional[float] = Field(default=0.80, description="Confidence level applied")
    metrics: Optional[Dict[str, float]] = Field(default=None, description="Validation error metrics")
    model_selection: Optional[ModelSelectionResult] = Field(default=None, description="Multi-model audit report")
    selection_result: Optional[ModelSelectionResult] = Field(default=None, description="Alias for model_selection")
    model_metadata: Optional[ModelMetadata] = Field(default=None, description="Hyperparameters and feature context")
    training_period: Optional[Dict[str, str]] = Field(default=None, description="Start and end dates of training")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings emitted during modeling")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def sync_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "model_selection" in data and "selection_result" not in data:
                data["selection_result"] = data["model_selection"]
            elif "selection_result" in data and "model_selection" not in data:
                data["model_selection"] = data["selection_result"]
            if "forecast_points" in data and ("predictions" not in data or not data["predictions"]):
                data["predictions"] = data["forecast_points"]
            elif "predictions" in data and ("forecast_points" not in data or not data["forecast_points"]):
                data["forecast_points"] = data["predictions"]
        return data
