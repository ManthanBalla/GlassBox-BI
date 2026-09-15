"""Multi-Agent Orchestrator for GlassBox-BI (Phase 8).

Main coordinator integrating Phase 3 (Data Processing), Phase 4 (Forecasting),
Phase 5 (Forecast Evaluation), Phase 6 (Explainability), and Phase 7 (Decision Intelligence)
into a controlled, auditable, and deterministic multi-agent workflow.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.app.schemas.orchestration import (
    OrchestrationRequest,
    OrchestrationResult,
    OrchestrationState,
    WorkflowStatus,
)
from backend.app.orchestration.base import BaseOrchestrator
from backend.app.orchestration.executor import SequentialWorkflowExecutor
from backend.app.orchestration.registry import AgentRegistry
from backend.app.orchestration.validation import WorkflowValidator


class MultiAgentOrchestrator(BaseOrchestrator):
    """Central deterministic coordinator executing the complete GlassBox-BI agent pipeline."""

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        executor: Optional[SequentialWorkflowExecutor] = None,
    ) -> None:
        self.registry = registry or AgentRegistry()
        self.executor = executor or SequentialWorkflowExecutor(registry=self.registry)
        self._state_store: Dict[str, OrchestrationState] = {}

    def create_workflow(self, request: OrchestrationRequest) -> OrchestrationState:
        """Initializes a new workflow execution context and state."""
        now = datetime.now(timezone.utc)
        workflow_id = f"wf_{now.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

        state = OrchestrationState(
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            timestamps={"created_at": now.isoformat()},
            input_metadata={
                "dataset_path": request.dataset_path,
                "entity_id": request.entity_id,
                "product_id": request.product_id,
                "horizon": request.horizon,
                "selection_metric": request.selection_metric,
                "confidence_level": request.confidence_level,
                "candidate_models": request.candidate_models,
                "run_evaluation": request.run_evaluation,
                "run_explainability": request.run_explainability,
                "explanation_method": request.explanation_method,
            },
        )

        self._state_store[workflow_id] = state
        return state

    def execute(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Validates inputs and executes the complete multi-agent pipeline sequentially."""
        # 1. Fail-early validation
        validation_issues = WorkflowValidator.validate_request(request)
        if validation_issues:
            # Create a failed state for record-keeping
            state = self.create_workflow(request)
            state.status = WorkflowStatus.FAILED
            state.errors.extend(validation_issues)
            raise ValueError(f"Workflow request validation failed: {'; '.join(validation_issues)}")

        # 2. Create initialized state
        state = self.create_workflow(request)

        # 3. Execute sequentially through the executor
        result = self.executor.execute(request=request, state=state)

        # 4. Save state to cache
        self._state_store[state.workflow_id] = state
        return result

    def get_state(self, workflow_id: str) -> Optional[OrchestrationState]:
        """Retrieves current observable state for a given workflow identifier."""
        return self._state_store.get(workflow_id)

    def validate_state(self, state: OrchestrationState) -> List[str]:
        """Verifies state consistency and invariant adherence."""
        return WorkflowValidator.validate_state(state)

    def list_workflows(self) -> List[Dict[str, Any]]:
        """Returns metadata summaries for all cached workflow executions."""
        summaries = []
        for wf_id, s in self._state_store.items():
            summaries.append({
                "workflow_id": wf_id,
                "status": s.status.value,
                "completed_stages": [st.value for st in s.completed_stages],
                "failed_stages": [st.value for st in s.failed_stages],
                "created_at": s.timestamps.get("created_at"),
                "completed_at": s.timestamps.get("completed_at"),
            })
        return summaries
