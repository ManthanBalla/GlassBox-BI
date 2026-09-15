"""Workflow Audit Trail Tracker for Multi-Agent Orchestration (Phase 8).

Builds immutable audit records capturing end-to-end data lineage, stage execution timings,
model choices, explanation methods, and governance warnings.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.app.schemas.orchestration import (
    OrchestrationAuditRecord,
    OrchestrationState,
    PipelineStage,
    StageStatus,
)


class WorkflowAuditTracker:
    """Creates standardized audit trail records from orchestration state."""

    @classmethod
    def build_audit_record(
        cls,
        state: OrchestrationState,
        started_at: datetime,
        completed_at: Optional[datetime] = None,
        total_duration_ms: float = 0.0,
    ) -> OrchestrationAuditRecord:
        """Constructs an immutable OrchestrationAuditRecord from completed workflow state."""
        stage_seq: List[str] = [s.value for s in state.completed_stages + state.failed_stages]
        stage_statuses: Dict[str, str] = {
            k: v.status.value for k, v in state.stage_results.items()
        }
        stage_durations: Dict[str, float] = {
            k: round(v.duration_ms, 2) for k, v in state.stage_results.items()
        }

        # Extract selected model if forecast succeeded
        selected_model = None
        forecast_horizon = None
        fcst_result = state.stage_results.get(PipelineStage.FORECASTING.value)
        if fcst_result and fcst_result.output:
            out = fcst_result.output
            if hasattr(out, "model_name"):
                selected_model = out.model_name
            elif isinstance(out, dict):
                selected_model = out.get("model_name")
            if hasattr(out, "horizon"):
                forecast_horizon = out.horizon
            elif isinstance(out, dict):
                forecast_horizon = out.get("horizon")

        # Extract explanation method
        explanation_method = None
        expl_result = state.stage_results.get(PipelineStage.EXPLAINABILITY.value)
        if expl_result and expl_result.output:
            out = expl_result.output
            if hasattr(out, "method"):
                explanation_method = out.method
            elif isinstance(out, dict):
                explanation_method = out.get("method")

        # Extract recommendation count
        rec_count = 0
        dec_result = state.stage_results.get(PipelineStage.DECISION_INTELLIGENCE.value)
        if dec_result and dec_result.output:
            out = dec_result.output
            if hasattr(out, "recommendations"):
                rec_count = len(out.recommendations)
            elif isinstance(out, dict):
                rec_count = len(out.get("recommendations", []))

        return OrchestrationAuditRecord(
            workflow_id=state.workflow_id,
            started_at=started_at,
            completed_at=completed_at or datetime.now(timezone.utc),
            total_duration_ms=round(total_duration_ms, 2),
            stage_sequence=stage_seq,
            stage_statuses=stage_statuses,
            stage_durations_ms=stage_durations,
            selected_model=selected_model,
            forecast_horizon=forecast_horizon,
            explanation_method=explanation_method,
            recommendation_count=rec_count,
            warnings_count=len(state.warnings),
            errors_count=len(state.errors),
            final_status=state.status.value,
        )
