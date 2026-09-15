"""Unit and Failure Isolation Tests for Multi-Agent Orchestration (Phase 8).

Verifies error isolation boundaries, skip policies, partial result preservation,
and strictly enforces the absence of automatic retries or self-correction loops.
"""

from unittest.mock import MagicMock
import pytest

from backend.app.schemas.decisions import BusinessContext
from backend.app.schemas.orchestration import (
    OrchestrationRequest,
    PipelineStage,
    StageStatus,
    WorkflowStatus,
)
from backend.app.orchestration.agent import MultiAgentOrchestrator
from backend.app.orchestration.registry import AgentRegistry


def test_invalid_input_fail_early():
    """Verify fail-early validation on invalid request parameters."""
    orchestrator = MultiAgentOrchestrator()

    # 1. Non-existent dataset path
    req_bad_path = OrchestrationRequest(dataset_path="data/non_existent_file_12345.csv")
    with pytest.raises(ValueError) as excinfo:
        orchestrator.execute(req_bad_path)
    assert "dataset_path does not exist" in str(excinfo.value)

    # 2. Invalid selection metric
    req_bad_metric = OrchestrationRequest(selection_metric="INVALID_METRIC")
    with pytest.raises(ValueError) as excinfo:
        orchestrator.execute(req_bad_metric)
    assert "Invalid selection_metric" in str(excinfo.value)


def test_data_processing_failure_halts_downstream():
    """Scenario A: If data processing fails, downstream stages are not executed."""
    reg = AgentRegistry()

    # Mock processor to raise RuntimeError
    mock_processor = MagicMock()
    mock_processor.process.side_effect = RuntimeError("Synthetic data corruption error.")
    reg.register(PipelineStage.DATA_PROCESSING, mock_processor)

    # Mock forecaster to ensure it is NEVER called
    mock_forecaster = MagicMock()
    reg.register(PipelineStage.FORECASTING, mock_forecaster)

    orchestrator = MultiAgentOrchestrator(registry=reg)
    req = OrchestrationRequest(dataset_path="data/sample/retail_sample.csv")

    result = orchestrator.execute(req)

    # Workflow is FAILED
    assert result.status == WorkflowStatus.FAILED
    assert PipelineStage.DATA_PROCESSING in [PipelineStage(s) for s in result.audit.stage_sequence]

    # Forecaster must NOT have been called
    mock_forecaster.run_forecast.assert_not_called()

    # Downstream stages marked SKIPPED
    state = orchestrator.get_state(result.workflow_id)
    assert PipelineStage.FORECASTING in state.skipped_stages
    assert PipelineStage.EVALUATION in state.skipped_stages
    assert PipelineStage.EXPLAINABILITY in state.skipped_stages
    assert PipelineStage.DECISION_INTELLIGENCE in state.skipped_stages


def test_section_27_important_boundary_no_self_correction_on_forecast_failure():
    """Section 27 Requirement: Proves Phase 8 does NOT perform self-correction or retries.

    When forecasting fails:
    - Reports FORECASTING failure
    - Exactly 1 forecast attempt is made (NO second attempt, NO retry loop)
    - NO retraining
    - NO model re-selection
    - Downstream stages SKIPPED
    """
    reg = AgentRegistry()

    mock_forecaster = MagicMock()
    mock_forecaster.run_forecast.side_effect = RuntimeError("Candidate training divergence error.")
    reg.register(PipelineStage.FORECASTING, mock_forecaster)

    mock_explainer = MagicMock()
    reg.register(PipelineStage.EXPLAINABILITY, mock_explainer)

    mock_decision = MagicMock()
    reg.register(PipelineStage.DECISION_INTELLIGENCE, mock_decision)

    orchestrator = MultiAgentOrchestrator(registry=reg)
    req = OrchestrationRequest(
        dataset_path="data/processed/synthetic/retail_processed.csv",
        candidate_models=["lightgbm"],
    )

    result = orchestrator.execute(req)

    # Status must be FAILED
    assert result.status == WorkflowStatus.FAILED
    assert "Candidate training divergence error" in result.errors[0]

    # CRITICAL INVARIANT: Forecaster was called EXACTLY ONCE (no automatic retry or self-correction)
    assert mock_forecaster.run_forecast.call_count == 1

    # Downstream explainability and decision intelligence must NOT be called
    mock_explainer.explain_local.assert_not_called()
    mock_decision.run_decision.assert_not_called()

    # Data processing was successful and preserved
    state = orchestrator.get_state(result.workflow_id)
    assert PipelineStage.DATA_PROCESSING in state.completed_stages
    assert PipelineStage.FORECASTING in state.failed_stages
    assert PipelineStage.EVALUATION in state.skipped_stages
    assert PipelineStage.EXPLAINABILITY in state.skipped_stages
    assert PipelineStage.DECISION_INTELLIGENCE in state.skipped_stages


def test_evaluation_failure_preserves_pipeline_partial():
    """Scenario C1: Evaluation failure marks stage failed, but downstream stages continue."""
    reg = AgentRegistry()

    # Mock evaluation engine to fail
    mock_eval = MagicMock()
    mock_eval.run_benchmark.side_effect = RuntimeError("Evaluation holdout mismatch.")
    reg.register(PipelineStage.EVALUATION, mock_eval)

    orchestrator = MultiAgentOrchestrator(registry=reg)
    req = OrchestrationRequest(
        dataset_path="data/processed/synthetic/retail_processed.csv",
        candidate_models=["lightgbm"],
        horizon=7,
        run_evaluation=True,
        run_explainability=True,
    )

    result = orchestrator.execute(req)

    # Non-blocking evaluation failure -> PARTIAL status
    assert result.status == WorkflowStatus.PARTIAL
    assert result.stage_summary["EVALUATION"]["status"] == StageStatus.FAILED.value
    assert result.stage_summary["DATA_PROCESSING"]["status"] == StageStatus.SUCCESS.value
    assert result.stage_summary["FORECASTING"]["status"] == StageStatus.SUCCESS.value
    assert result.stage_summary["EXPLAINABILITY"]["status"] == StageStatus.SUCCESS.value
    assert result.stage_summary["DECISION_INTELLIGENCE"]["status"] == StageStatus.SUCCESS.value

    # Forecast, explanation, and decisions are fully preserved
    assert result.forecast is not None
    assert result.explanation is not None
    assert result.decisions is not None


def test_explainability_failure_allows_decisions_with_missing_explanation_penalty():
    """Scenario C2: Explanation failure preserves forecast, decisions run with fallback penalty."""
    reg = AgentRegistry()

    # Mock explainer to fail
    mock_explainer = MagicMock()
    mock_explainer.explain_local.side_effect = RuntimeError("Tree SHAP memory error.")
    reg.register(PipelineStage.EXPLAINABILITY, mock_explainer)

    orchestrator = MultiAgentOrchestrator(registry=reg)
    req = OrchestrationRequest(
        dataset_path="data/processed/synthetic/retail_processed.csv",
        candidate_models=["lightgbm"],
        horizon=7,
        run_evaluation=False,
        run_explainability=True,
    )

    result = orchestrator.execute(req)

    # Workflow is PARTIAL
    assert result.status == WorkflowStatus.PARTIAL
    assert result.stage_summary["EXPLAINABILITY"]["status"] == StageStatus.FAILED.value
    assert result.stage_summary["DECISION_INTELLIGENCE"]["status"] == StageStatus.SUCCESS.value

    # Forecast preserved, explanation is None
    assert result.forecast is not None
    assert result.explanation is None

    # Decision executed with missing explanation assumption
    assert result.decisions is not None
    assert result.decisions.primary_recommendation is not None
    assert any("explanation" in a.lower() for a in result.decisions.primary_recommendation.assumptions)


def test_decision_failure_preserves_earlier_outputs():
    """Scenario D: Decision failure preserves data, forecast, evaluation, and explanation."""
    reg = AgentRegistry()

    mock_dec = MagicMock()
    mock_dec.run_decision.side_effect = RuntimeError("Decision engine constraint violation.")
    reg.register(PipelineStage.DECISION_INTELLIGENCE, mock_dec)

    orchestrator = MultiAgentOrchestrator(registry=reg)
    req = OrchestrationRequest(
        dataset_path="data/processed/synthetic/retail_processed.csv",
        candidate_models=["lightgbm"],
        horizon=7,
        run_evaluation=False,
        run_explainability=True,
    )

    result = orchestrator.execute(req)

    # Status PARTIAL
    assert result.status == WorkflowStatus.PARTIAL
    assert result.stage_summary["DECISION_INTELLIGENCE"]["status"] == StageStatus.FAILED.value

    # Earlier outputs preserved
    assert result.data_processing is not None
    assert result.forecast is not None
    assert result.explanation is not None
    assert result.decisions is None
