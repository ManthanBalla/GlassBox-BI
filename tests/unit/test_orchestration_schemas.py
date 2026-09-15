"""Unit Tests for Orchestration Schemas and State Contracts (Phase 8)."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from backend.app.schemas.orchestration import (
    OrchestrationAuditRecord,
    OrchestrationRequest,
    OrchestrationResult,
    OrchestrationState,
    PipelineStage,
    StageExecutionResult,
    StageStatus,
    WorkflowStatus,
)


def test_pipeline_stage_enum_values():
    """Verify all 5 pipeline stages are explicitly defined."""
    assert PipelineStage.DATA_PROCESSING.value == "DATA_PROCESSING"
    assert PipelineStage.FORECASTING.value == "FORECASTING"
    assert PipelineStage.EVALUATION.value == "EVALUATION"
    assert PipelineStage.EXPLAINABILITY.value == "EXPLAINABILITY"
    assert PipelineStage.DECISION_INTELLIGENCE.value == "DECISION_INTELLIGENCE"
    assert len(PipelineStage) == 5


def test_workflow_and_stage_status_enums():
    """Verify workflow and stage lifecycle statuses."""
    assert WorkflowStatus.PENDING.value == "PENDING"
    assert WorkflowStatus.RUNNING.value == "RUNNING"
    assert WorkflowStatus.COMPLETED.value == "COMPLETED"
    assert WorkflowStatus.FAILED.value == "FAILED"
    assert WorkflowStatus.PARTIAL.value == "PARTIAL"

    assert StageStatus.SUCCESS.value == "SUCCESS"
    assert StageStatus.SKIPPED.value == "SKIPPED"


def test_stage_execution_result_defaults():
    """Verify StageExecutionResult default values and validation."""
    res = StageExecutionResult(
        stage=PipelineStage.DATA_PROCESSING,
    )
    assert res.stage == PipelineStage.DATA_PROCESSING
    assert res.status == StageStatus.PENDING
    assert res.duration_ms == 0.0
    assert res.warnings == []
    assert res.errors == []
    assert res.metadata == {}
    assert isinstance(res.started_at, datetime)


def test_orchestration_request_defaults_and_constraints():
    """Verify OrchestrationRequest validation and default values."""
    req = OrchestrationRequest()
    assert req.horizon == 14
    assert req.selection_metric == "MAE"
    assert req.confidence_level == 0.80
    assert "lightgbm" in req.candidate_models
    assert req.run_evaluation is True
    assert req.run_explainability is True
    assert req.fidelity_threshold == 0.65
    assert req.random_seed == 42

    # Invalid horizon
    with pytest.raises(ValidationError):
        OrchestrationRequest(horizon=0)

    # Invalid confidence level
    with pytest.raises(ValidationError):
        OrchestrationRequest(confidence_level=0.40)


def test_orchestration_state_lifecycle():
    """Verify state updates across workflow lifecycle."""
    state = OrchestrationState(workflow_id="wf_test_001")
    assert state.workflow_id == "wf_test_001"
    assert state.status == WorkflowStatus.PENDING
    assert state.completed_stages == []
    assert state.failed_stages == []
    assert state.skipped_stages == []

    # Update state
    state.status = WorkflowStatus.RUNNING
    state.current_stage = PipelineStage.DATA_PROCESSING
    state.completed_stages.append(PipelineStage.DATA_PROCESSING)
    assert len(state.completed_stages) == 1
    assert state.completed_stages[0] == PipelineStage.DATA_PROCESSING


def test_orchestration_audit_record_validation():
    """Verify audit record structure and serialization."""
    now = datetime.now(timezone.utc)
    audit = OrchestrationAuditRecord(
        workflow_id="wf_audit_001",
        started_at=now,
        completed_at=now,
        total_duration_ms=250.5,
        stage_sequence=["DATA_PROCESSING", "FORECASTING"],
        stage_statuses={"DATA_PROCESSING": "SUCCESS", "FORECASTING": "SUCCESS"},
        stage_durations_ms={"DATA_PROCESSING": 120.0, "FORECASTING": 130.5},
        selected_model="lightgbm",
        forecast_horizon=14,
        explanation_method="shap",
        recommendation_count=2,
        final_status="COMPLETED",
    )
    assert audit.workflow_id == "wf_audit_001"
    assert audit.total_duration_ms == 250.5
    assert audit.selected_model == "lightgbm"
    dump = audit.model_dump(mode="json")
    assert dump["final_status"] == "COMPLETED"
