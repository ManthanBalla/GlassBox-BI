"""Unit Tests for Workflow Execution Graph and Dependency Policies (Phase 8)."""

from backend.app.schemas.orchestration import PipelineStage
from backend.app.orchestration.graph import WorkflowGraph


def test_canonical_pipeline_sequence():
    """Verify exact 5-stage sequential order."""
    expected = [
        PipelineStage.DATA_PROCESSING,
        PipelineStage.FORECASTING,
        PipelineStage.EVALUATION,
        PipelineStage.EXPLAINABILITY,
        PipelineStage.DECISION_INTELLIGENCE,
    ]
    assert WorkflowGraph.get_sequence() == expected


def test_stage_dependencies():
    """Verify declared dependencies for each pipeline stage."""
    assert WorkflowGraph.get_dependencies(PipelineStage.DATA_PROCESSING) == []
    assert WorkflowGraph.get_dependencies(PipelineStage.FORECASTING) == [PipelineStage.DATA_PROCESSING]
    assert WorkflowGraph.get_dependencies(PipelineStage.EVALUATION) == [PipelineStage.FORECASTING]
    assert WorkflowGraph.get_dependencies(PipelineStage.EXPLAINABILITY) == [PipelineStage.FORECASTING]
    assert WorkflowGraph.get_dependencies(PipelineStage.DECISION_INTELLIGENCE) == [PipelineStage.FORECASTING]


def test_blocking_failure_stages():
    """Verify which stages constitute blocking failures."""
    assert WorkflowGraph.is_blocking_failure(PipelineStage.DATA_PROCESSING) is True
    assert WorkflowGraph.is_blocking_failure(PipelineStage.FORECASTING) is True
    assert WorkflowGraph.is_blocking_failure(PipelineStage.EVALUATION) is False
    assert WorkflowGraph.is_blocking_failure(PipelineStage.EXPLAINABILITY) is False
    assert WorkflowGraph.is_blocking_failure(PipelineStage.DECISION_INTELLIGENCE) is False


def test_can_execute_rules():
    """Verify dependency check logic in can_execute."""
    # Initially: DATA_PROCESSING can execute
    can_run, _ = WorkflowGraph.can_execute(PipelineStage.DATA_PROCESSING, completed_stages=[], failed_stages=[])
    assert can_run is True

    # FORECASTING cannot run before DATA_PROCESSING completes
    can_run, reason = WorkflowGraph.can_execute(PipelineStage.FORECASTING, completed_stages=[], failed_stages=[])
    assert can_run is False
    assert "has not completed" in reason

    # After DATA_PROCESSING completes: FORECASTING can run
    can_run, _ = WorkflowGraph.can_execute(
        PipelineStage.FORECASTING,
        completed_stages=[PipelineStage.DATA_PROCESSING],
        failed_stages=[],
    )
    assert can_run is True

    # If FORECASTING fails: EVALUATION, EXPLAINABILITY, DECISION cannot run
    for s in [PipelineStage.EVALUATION, PipelineStage.EXPLAINABILITY, PipelineStage.DECISION_INTELLIGENCE]:
        can_run, reason = WorkflowGraph.can_execute(
            stage=s,
            completed_stages=[PipelineStage.DATA_PROCESSING],
            failed_stages=[PipelineStage.FORECASTING],
        )
        assert can_run is False
        assert "blocking stage 'FORECASTING' failed" in reason


def test_stages_to_skip_after_blocking_failure():
    """Verify downstream skip calculation after blocking failure."""
    remaining = [
        PipelineStage.EVALUATION,
        PipelineStage.EXPLAINABILITY,
        PipelineStage.DECISION_INTELLIGENCE,
    ]
    to_skip = WorkflowGraph.get_stages_to_skip_after_failure(
        failed_stage=PipelineStage.FORECASTING,
        remaining_stages=remaining,
    )
    assert to_skip == remaining
