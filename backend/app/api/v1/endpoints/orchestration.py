"""Multi-Agent Orchestration API Endpoints for GlassBox-BI (Phase 8).

Exposes REST endpoints for triggering end-to-end multi-agent pipelines,
inspecting stage dependencies, discovering registered agents, and tracking workflow states.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from backend.app.schemas.decisions import BusinessContext
from backend.app.schemas.orchestration import (
    OrchestrationRequest,
    OrchestrationResult,
    OrchestrationState,
    PipelineStage,
    WorkflowStatus,
)
from backend.app.orchestration.agent import MultiAgentOrchestrator
from backend.app.orchestration.graph import WorkflowGraph

router = APIRouter()
_orchestrator = MultiAgentOrchestrator()


@router.get(
    "/health",
    summary="Multi-Agent Orchestration Subsystem Health",
    response_model=Dict[str, Any],
)
def get_orchestration_health() -> Dict[str, Any]:
    """Returns the operational status, registered agents, and dependency graph of the orchestrator."""
    return {
        "status": "HEALTHY",
        "module": "Phase 8 — Multi-Agent Orchestration",
        "pipeline_sequence": [s.value for s in WorkflowGraph.get_sequence()],
        "registered_agents": _orchestrator.registry.list_registered_stages(),
        "execution_mode": "deterministic_sequential_state_machine",
        "llm_dependency": False,
        "self_correction_enabled": False,  # Strict Phase 9 boundary declaration
        "automatic_retries_enabled": False,  # Strict Phase 9 boundary declaration
    }


@router.get(
    "/stages",
    summary="Pipeline Stage Catalog and Dependency Graph",
    response_model=List[Dict[str, Any]],
)
def get_orchestration_stages() -> List[Dict[str, Any]]:
    """Returns the ordered list of pipeline stages, predecessor dependencies, and agent bindings."""
    stages_info = []
    for stage in WorkflowGraph.get_sequence():
        deps = WorkflowGraph.get_dependencies(stage)
        agent = _orchestrator.registry.get_agent(stage)
        stages_info.append({
            "stage": stage.value,
            "predecessor_dependencies": [d.value for d in deps],
            "is_blocking_failure": WorkflowGraph.is_blocking_failure(stage),
            "agent_implementation": type(agent).__name__,
            "description": f"Phase {stage.value} automated execution block",
        })
    return stages_info


@router.get(
    "/sample",
    summary="Fast Sample Demonstration Workflow Execution",
    response_model=OrchestrationResult,
)
def get_orchestration_sample() -> OrchestrationResult:
    """Executes a fast, reproducible sample workflow on committed benchmark data (STORE_001 / PROD_001, 7-day horizon)."""
    sample_request = OrchestrationRequest(
        dataset_path="data/processed/synthetic/retail_processed.csv",
        entity_id="STORE_001",
        product_id="PROD_001",
        horizon=7,
        selection_metric="MAE",
        candidate_models=["lightgbm"],
        run_evaluation=True,
        evaluation_horizon=7,
        run_explainability=True,
        explanation_method="auto",
        business_context=BusinessContext(
            current_inventory=80.0,
            reorder_point=50.0,
            lead_time_days=7,
            current_price=19.99,
            promotion_active=True,
        ),
    )
    try:
        return _orchestrator.execute(sample_request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sample orchestration execution failed: {str(exc)}",
        )


@router.post(
    "/run",
    summary="Execute Full Multi-Agent Pipeline",
    response_model=OrchestrationResult,
)
def run_orchestration(request: OrchestrationRequest) -> OrchestrationResult:
    """Executes the complete multi-agent pipeline sequentially across Data Processing, Forecasting,

    Evaluation, Explainability, and Decision Intelligence.
    """
    try:
        return _orchestrator.execute(request)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Orchestration pipeline encountered an unexpected error: {str(exc)}",
        )


@router.get(
    "/{workflow_id}",
    summary="Retrieve Workflow State by ID",
    response_model=Dict[str, Any],
)
def get_workflow_by_id(workflow_id: str) -> Dict[str, Any]:
    """Returns the current state and results for an executed workflow identifier."""
    state = _orchestrator.get_state(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow with ID '{workflow_id}' not found in active session cache.",
        )
    return state.model_dump(mode="json")
