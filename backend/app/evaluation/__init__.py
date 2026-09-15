"""Formal Forecasting Evaluation Module for GlassBox-BI (Phase 5).

Provides model-agnostic test-set evaluation, deterministic metrics (MAE, RMSE, MAPE),
and benchmark engine contrasting Prophet, LightGBM, and PyTorch LSTM.
"""

from backend.app.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    compute_evaluation_metrics,
)
from backend.app.evaluation.schemas import (
    EvaluationMetricType,
    EvaluationMetricResult,
    EvaluationPoint,
    ModelTestEvaluation,
    EvaluationRequest,
    BenchmarkResult,
)
from backend.app.evaluation.evaluator import FormalForecastEvaluator
from backend.app.evaluation.benchmark import ForecastingBenchmarkEngine

__all__ = [
    "calculate_mae",
    "calculate_rmse",
    "calculate_mape",
    "compute_evaluation_metrics",
    "EvaluationMetricType",
    "EvaluationMetricResult",
    "EvaluationPoint",
    "ModelTestEvaluation",
    "EvaluationRequest",
    "BenchmarkResult",
    "FormalForecastEvaluator",
    "ForecastingBenchmarkEngine",
]
