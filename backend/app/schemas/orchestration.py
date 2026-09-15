"""Pydantic Schemas and Contracts for Multi-Agent Orchestration (Phase 8).

Defines strongly-typed contracts for workflow coordination, state tracking,
stage execution records, failure isolation policies, and unified results.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from backend.app.schemas.decisions import BusinessContext, DecisionResult
from backend.app.schemas.evaluation import BenchmarkResult, ModelTestEvaluation
from backend.app.schemas.explainability import ExplanationResult
from backend.app.schemas.forecasting import ForecastResult
from backend.app.schemas.processing import DataProcessingResult


class PipelineStage(str, Enum):
    """Explicit enumeration of ordered stages in the multi-agent pipeline."""
    DATA_PROCESSING = "DATA_PROCESSING"
    FORECASTING = "FORECASTING"
    EVALUATION = "EVALUATION"
    EXPLAINABILITY = "EXPLAINABILITY"
    DECISION_INTELLIGENCE = "DECISION_INTELLIGENCE"


class WorkflowStatus(str, Enum):
    """High-level lifecycle status of the entire orchestrated workflow."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class StageStatus(str, Enum):
    """Execution status of an individual pipeline stage."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class StageExecutionResult(BaseModel):
    """Standardized, auditable record of an individual stage's execution."""
    stage: PipelineStage = Field(..., description="Pipeline stage identifier")
    status: StageStatus = Field(default=StageStatus.PENDING, description="Execution status")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when stage execution began",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when stage execution completed",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Execution duration in milliseconds",
    )
    output: Optional[Any] = Field(
        default=None,
        description="Structured stage output payload or reference",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Operational warnings emitted during stage execution",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Error messages if stage failed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stage-specific execution metadata",
    )


class OrchestrationRequest(BaseModel):
    """Input parameters to trigger an orchestrated multi-agent workflow."""
    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to input dataset file (CSV). Defaults to benchmark data if omitted.",
    )
    entity_id: Optional[str] = Field(
        default="STORE_001",
        description="Target business entity identifier",
    )
    product_id: Optional[str] = Field(
        default="PROD_001",
        description="Target product / SKU identifier",
    )
    horizon: int = Field(
        default=14,
        ge=1,
        le=90,
        description="Forecasting and evaluation horizon (steps)",
    )
    selection_metric: str = Field(
        default="MAE",
        description="Model validation selection criterion (MAE, RMSE, MAPE)",
    )
    confidence_level: float = Field(
        default=0.80,
        ge=0.50,
        le=0.99,
        description="Confidence interval coverage level",
    )
    candidate_models: List[str] = Field(
        default_factory=lambda: ["lightgbm", "prophet", "lstm"],
        description="Candidate forecasting architectures to evaluate",
    )
    run_evaluation: bool = Field(
        default=True,
        description="Whether to execute Phase 5 formal test-set evaluation",
    )
    evaluation_horizon: Optional[int] = Field(
        default=14,
        ge=1,
        description="Evaluation horizon on the holdout test set",
    )
    run_explainability: bool = Field(
        default=True,
        description="Whether to execute Phase 6 XAI feature attribution",
    )
    explanation_method: str = Field(
        default="auto",
        description="Explanation algorithm: auto, shap, lime, or component_based",
    )
    business_context: Optional[BusinessContext] = Field(
        default=None,
        description="Operational business context for prescriptive decisions",
    )
    fidelity_threshold: float = Field(
        default=0.65,
        ge=0.10,
        le=1.0,
        description="Reliability threshold for XAI explanation fidelity",
    )
    uncertainty_threshold: float = Field(
        default=0.35,
        ge=0.05,
        le=1.0,
        description="Dispersion threshold for forecast prediction intervals",
    )
    random_seed: int = Field(
        default=42,
        description="Deterministic random seed for reproducibility",
    )


class OrchestrationAuditRecord(BaseModel):
    """Immutable audit trail summarizing the complete multi-agent workflow."""
    workflow_id: str = Field(..., description="Unique workflow execution identifier")
    started_at: datetime = Field(..., description="UTC start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="UTC completion timestamp")
    total_duration_ms: float = Field(default=0.0, ge=0.0, description="Total execution time in ms")
    stage_sequence: List[str] = Field(
        default_factory=list,
        description="Exact chronological sequence of executed stages",
    )
    stage_statuses: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of each stage (SUCCESS, FAILED, SKIPPED)",
    )
    stage_durations_ms: Dict[str, float] = Field(
        default_factory=dict,
        description="Duration of each stage in milliseconds",
    )
    selected_model: Optional[str] = Field(default=None, description="Winning forecasting model")
    forecast_horizon: Optional[int] = Field(default=None, description="Forecast horizon steps")
    explanation_method: Optional[str] = Field(default=None, description="Applied XAI method")
    recommendation_count: int = Field(default=0, ge=0, description="Prescriptive recommendations count")
    warnings_count: int = Field(default=0, ge=0, description="Total operational warnings")
    errors_count: int = Field(default=0, ge=0, description="Total errors encountered")
    final_status: str = Field(default="PENDING", description="Final workflow status")


class OrchestrationResult(BaseModel):
    """Unified response payload representing the completed multi-agent workflow."""
    workflow_id: str = Field(..., description="Unique workflow execution identifier")
    status: WorkflowStatus = Field(..., description="Final workflow status")
    stage_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary status and duration for each pipeline stage",
    )
    data_processing: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 3 data processing result summary",
    )
    forecast: Optional[ForecastResult] = Field(
        default=None,
        description="Phase 4 forecasting agent output",
    )
    evaluation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Phase 5 forecast evaluation metrics and benchmark",
    )
    explanation: Optional[ExplanationResult] = Field(
        default=None,
        description="Phase 6 explainability agent attributions",
    )
    decisions: Optional[DecisionResult] = Field(
        default=None,
        description="Phase 7 decision intelligence prescriptive recommendations",
    )
    warnings: List[str] = Field(default_factory=list, description="Aggregated pipeline warnings")
    errors: List[str] = Field(default_factory=list, description="Aggregated pipeline errors")
    audit: Optional[OrchestrationAuditRecord] = Field(
        default=None,
        description="Immutable audit trail record",
    )
    execution_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="High-level operational recap across all stages",
    )


class OrchestrationState(BaseModel):
    """Strongly-typed, observable state of an active or completed workflow."""
    workflow_id: str = Field(..., description="Unique workflow execution identifier")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING, description="Current workflow status")
    current_stage: Optional[PipelineStage] = Field(default=None, description="Currently executing stage")
    completed_stages: List[PipelineStage] = Field(
        default_factory=list,
        description="Chronologically completed stages",
    )
    failed_stages: List[PipelineStage] = Field(
        default_factory=list,
        description="Stages that encountered fatal execution errors",
    )
    skipped_stages: List[PipelineStage] = Field(
        default_factory=list,
        description="Stages bypassed due to upstream failures",
    )
    stage_results: Dict[str, StageExecutionResult] = Field(
        default_factory=dict,
        description="Map of stage name to its execution record",
    )
    warnings: List[str] = Field(default_factory=list, description="Cumulative operational warnings")
    errors: List[str] = Field(default_factory=list, description="Cumulative pipeline errors")
    timestamps: Dict[str, str] = Field(
        default_factory=dict,
        description="Key timestamp markers (started_at, completed_at, etc.)",
    )
    input_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Initial request and dataset metadata",
    )
    final_result: Optional[OrchestrationResult] = Field(
        default=None,
        description="Assembled final workflow result",
    )
    execution_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Executive execution summary",
    )
