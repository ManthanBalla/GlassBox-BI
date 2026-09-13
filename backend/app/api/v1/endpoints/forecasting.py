"""Forecasting Agent Endpoints for API v1.

Provides REST interfaces to execute model-agnostic forecasting competitions,
train candidate architectures (Prophet, LightGBM, LSTM), inspect models/config,
and generate future forecasts with uncertainty bounds.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
import torch

from backend.app.schemas.forecasting import (
    ForecastingConfig,
    ForecastModelType,
    ForecastRequest,
    ForecastResult,
    ModelMetadata,
)
from backend.app.forecasting.agent import GenericForecastingAgent

router = APIRouter()
agent = GenericForecastingAgent()


@router.post("/run", response_model=ForecastResult, summary="Execute Time-Series Forecast")
def run_forecast(request: ForecastRequest) -> ForecastResult:
    """Executes model-agnostic candidate evaluation, validation selection, and future forecasting.

    Temporal Integrity & Leakage Protection:
    - Splits training and validation chronologically.
    - Selects the winning model based solely on validation metric (e.g. MAE).
    - Test partition is strictly excluded and protected for Phase 5.
    - Generates multi-step future point forecasts with prediction intervals.
    """
    try:
        result = agent.run_forecast(request)
        return result
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Forecasting pipeline encountered an unhandled error: {str(exc)}",
        )


@router.post("/train", response_model=ForecastResult, summary="Train and Evaluate Candidate Models")
def train_and_evaluate(request: ForecastRequest) -> ForecastResult:
    """Trains candidate forecasting models on historical data and selects the top performer."""
    return run_forecast(request)


@router.get("/models", summary="List Supported Candidate Forecasting Architectures")
def list_supported_models() -> Dict[str, Any]:
    """Returns technical metadata, feature dependencies, and history requirements for all forecasters."""
    return {
        "candidate_models": [
            {
                "name": "prophet",
                "display_name": "Facebook Prophet",
                "model_type": "statistical_additive",
                "minimum_history_rows": 14,
                "uncertainty_method": "prophet_bayesian_interval",
                "supports_prediction_interval": True,
                "supported_features": ["date", "target", "calendar_seasonality"],
                "description": "Decomposable time series model fitting non-linear trends with yearly/weekly seasonality.",
            },
            {
                "name": "lightgbm",
                "display_name": "LightGBM Regressor",
                "model_type": "gradient_boosted_trees",
                "minimum_history_rows": 28,
                "uncertainty_method": "validation_residuals",
                "supports_prediction_interval": True,
                "supported_features": ["lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "calendar", "exogenous"],
                "description": "Fast gradient boosted tree model using multi-step recursive forecasting without future leakage.",
            },
            {
                "name": "lstm",
                "display_name": "PyTorch LSTM Recurrent Neural Network",
                "model_type": "deep_learning_sequence",
                "minimum_history_rows": 20,
                "uncertainty_method": "validation_residuals",
                "supports_prediction_interval": True,
                "supported_features": ["standardized_lag_sequence"],
                "description": "Lightweight sequence-to-one LSTM network with early stopping and train-only scaler normalization.",
            },
        ],
        "default_selection_metric": "MAE",
        "supported_selection_metrics": ["MAE", "RMSE", "MAPE"],
    }


@router.get("/config", response_model=ForecastingConfig, summary="Get Default Forecasting Configuration")
def get_default_forecasting_config() -> ForecastingConfig:
    """Returns the system default hyperparameters, training bounds, and selection criteria."""
    return ForecastingConfig()


@router.get("/health", summary="Forecasting Subsystem Health Check")
def get_forecasting_health() -> Dict[str, Any]:
    """Verifies that all required machine learning and deep learning dependencies are operational."""
    persistence_dir = Path("models/saved")
    persistence_dir.mkdir(parents=True, exist_ok=True)

    return {
        "status": "healthy",
        "dependencies": {
            "prophet": True,
            "lightgbm": True,
            "torch": True,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "sklearn": True,
        },
        "persistence_directory": str(persistence_dir.resolve()),
        "persistence_accessible": persistence_dir.exists(),
    }


@router.get("/sample", response_model=ForecastResult, summary="Execute Fast Demonstration Forecast")
def run_sample_forecast(
    model: str = Query("lightgbm", description="Candidate model to test ('prophet', 'lightgbm', 'lstm')"),
    horizon: int = Query(7, ge=1, le=60, description="Forecast horizon in days"),
) -> ForecastResult:
    sample_path = Path("data/processed/synthetic/retail_processed.csv")
    if not sample_path.exists():
        sample_path = Path("data/sample/retail_processed_sample.csv")
    if not sample_path.exists():
        sample_path = Path("data/sample/retail_sample.csv")
    if not sample_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Benchmark retail dataset not found in data/processed/ or data/sample/ directory.",
        )

    req = ForecastRequest(
        dataset_path=str(sample_path),
        entity_id="STORE_001",
        product_id="PROD_001",
        horizon=horizon,
        candidate_models=[model.lower()],
        selection_metric="MAE",
        confidence_level=0.80,
    )
    return agent.run_forecast(req)
