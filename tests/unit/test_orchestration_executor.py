"""Unit and Integration Tests for Sequential Workflow Executor (Phase 8)."""

import pytest

from backend.app.schemas.decisions import BusinessContext
from backend.app.schemas.orchestration import (
    OrchestrationRequest,
    PipelineStage,
    StageStatus,
    WorkflowStatus,
)
from backend.app.orchestration.agent import MultiAgentOrchestrator


def test_section_26_important_execution_order_and_cardinality():
    """Section 26 Requirement: Verify all 5 stages execute exactly once in exact order.

    Expected:
    [
        "DATA_PROCESSING",
        "FORECASTING",
        "EVALUATION",
        "EXPLAINABILITY",
        "DECISION_INTELLIGENCE"
    ]
    """
    orchestrator = MultiAgentOrchestrator()
    req = OrchestrationRequest(
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
            current_inventory=90.0,
            reorder_point=60.0,
            lead_time_days=7,
        ),
    )

    result = orchestrator.execute(req)
    state = orchestrator.get_state(result.workflow_id)
    assert state is not None

    # Verify status
    assert result.status == WorkflowStatus.COMPLETED
    assert state.status == WorkflowStatus.COMPLETED

    # Verify execution order and exactly-once execution
    execution_order = [s.value for s in state.completed_stages]
    expected_order = [
        "DATA_PROCESSING",
        "FORECASTING",
        "EVALUATION",
        "EXPLAINABILITY",
        "DECISION_INTELLIGENCE",
    ]
    assert execution_order == expected_order
    assert len(state.completed_stages) == 5
    assert len(state.failed_stages) == 0
    assert len(state.skipped_stages) == 0


def test_end_to_end_data_flow_between_stages():
    """Verify data contracts are transferred safely and populated across all stages."""
    orchestrator = MultiAgentOrchestrator()
    req = OrchestrationRequest(
        dataset_path="data/processed/synthetic/retail_processed.csv",
        entity_id="STORE_001",
        product_id="PROD_001",
        horizon=7,
        selection_metric="MAE",
        candidate_models=["lightgbm"],
        run_evaluation=True,
        evaluation_horizon=7,
        run_explainability=True,
        business_context=BusinessContext(
            current_inventory=50.0,
            reorder_point=80.0,
            lead_time_days=7,
        ),
    )

    result = orchestrator.execute(req)

    # 1. Data processing output present
    assert result.data_processing is not None
    assert result.data_processing["output_row_count"] > 0
    assert result.data_processing["quality_score_after"] is not None

    # 2. Forecast output present
    assert result.forecast is not None
    assert result.forecast.model_name == "lightgbm"
    assert len(result.forecast.predictions) == 7
    assert "validation_MAE" in result.forecast.metrics

    # 3. Evaluation output present
    assert result.evaluation is not None
    assert result.evaluation["horizon"] == 7
    assert len(result.evaluation["models"]) > 0

    # 4. Explanation output present
    assert result.explanation is not None
    assert result.explanation.fidelity is not None
    assert result.explanation.fidelity.fidelity_score > 0.50

    # 5. Decision intelligence output present
    assert result.decisions is not None
    assert len(result.decisions.recommendations) > 0
    assert result.decisions.primary_recommendation is not None
    assert result.decisions.audit_trail.decision_confidence > 0.0

    # 6. Audit trail present
    assert result.audit is not None
    assert result.audit.selected_model == "lightgbm"
    assert result.audit.total_duration_ms > 0.0
    assert result.audit.final_status == "COMPLETED"


def test_deterministic_repeatability():
    """Verify two runs with identical inputs produce identical decisions and scores."""
    orchestrator = MultiAgentOrchestrator()
    req = OrchestrationRequest(
        dataset_path="data/processed/synthetic/retail_processed.csv",
        entity_id="STORE_001",
        product_id="PROD_001",
        horizon=7,
        selection_metric="MAE",
        candidate_models=["lightgbm"],
        run_evaluation=True,
        evaluation_horizon=7,
        run_explainability=True,
        business_context=BusinessContext(
            current_inventory=100.0,
            reorder_point=50.0,
            lead_time_days=7,
        ),
        random_seed=42,
    )

    res1 = orchestrator.execute(req)
    res2 = orchestrator.execute(req)

    assert res1.status == res2.status
    assert res1.forecast.model_name == res2.forecast.model_name

    # Check predictions identical
    p1 = [round(p.prediction, 4) for p in res1.forecast.predictions]
    p2 = [round(p.prediction, 4) for p in res2.forecast.predictions]
    assert p1 == p2

    # Check decisions identical
    assert res1.decisions.primary_recommendation.action == res2.decisions.primary_recommendation.action
    assert res1.decisions.primary_recommendation.priority == res2.decisions.primary_recommendation.priority
    assert res1.decisions.audit_trail.decision_confidence == res2.decisions.audit_trail.decision_confidence
