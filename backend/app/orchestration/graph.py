"""Deterministic Workflow Execution Graph for Multi-Agent Orchestration (Phase 8).

Defines the explicit pipeline sequence, predecessor dependencies, and failure isolation boundaries.
"""

from typing import Dict, List, Optional, Set, Tuple

from backend.app.schemas.orchestration import PipelineStage


class WorkflowGraph:
    """Lightweight directed state machine governing stage sequence and dependencies."""

    # Explicit linear sequence
    SEQUENCE: List[PipelineStage] = [
        PipelineStage.DATA_PROCESSING,
        PipelineStage.FORECASTING,
        PipelineStage.EVALUATION,
        PipelineStage.EXPLAINABILITY,
        PipelineStage.DECISION_INTELLIGENCE,
    ]

    # Predecessor dependencies required before a stage can execute
    DEPENDENCIES: Dict[PipelineStage, List[PipelineStage]] = {
        PipelineStage.DATA_PROCESSING: [],
        PipelineStage.FORECASTING: [PipelineStage.DATA_PROCESSING],
        PipelineStage.EVALUATION: [PipelineStage.FORECASTING],
        PipelineStage.EXPLAINABILITY: [PipelineStage.FORECASTING],
        PipelineStage.DECISION_INTELLIGENCE: [PipelineStage.FORECASTING],
    }

    # Stages that cause a total pipeline halt if they fail
    BLOCKING_FAILURE_STAGES: Set[PipelineStage] = {
        PipelineStage.DATA_PROCESSING,
        PipelineStage.FORECASTING,
    }

    @classmethod
    def get_sequence(cls) -> List[PipelineStage]:
        """Returns the canonical execution sequence."""
        return list(cls.SEQUENCE)

    @classmethod
    def get_dependencies(cls, stage: PipelineStage) -> List[PipelineStage]:
        """Returns required preceding stages for the given stage."""
        return list(cls.DEPENDENCIES.get(stage, []))

    @classmethod
    def is_blocking_failure(cls, stage: PipelineStage) -> bool:
        """Determines if a failure in this stage halts the entire remaining pipeline."""
        return stage in cls.BLOCKING_FAILURE_STAGES

    @classmethod
    def can_execute(
        cls,
        stage: PipelineStage,
        completed_stages: List[PipelineStage],
        failed_stages: List[PipelineStage],
    ) -> Tuple[bool, Optional[str]]:
        """Determines whether a stage is eligible to execute given current workflow state.

        Returns:
            Tuple of (can_run: bool, reason: Optional[str]).
        """
        # Check if any blocking failure occurred upstream
        for failed in failed_stages:
            if cls.is_blocking_failure(failed):
                return False, f"Cannot execute '{stage.value}': blocking stage '{failed.value}' failed."

        # Check required dependencies
        required = cls.get_dependencies(stage)
        for req in required:
            if req not in completed_stages:
                # If required stage failed and is blocking, it's already caught above.
                # If required stage is not in completed stages:
                if req in failed_stages:
                    return False, f"Cannot execute '{stage.value}': required predecessor '{req.value}' failed."
                return False, f"Cannot execute '{stage.value}': required predecessor '{req.value}' has not completed."

        return True, None

    @classmethod
    def get_stages_to_skip_after_failure(
        cls,
        failed_stage: PipelineStage,
        remaining_stages: List[PipelineStage],
    ) -> List[PipelineStage]:
        """Calculates which remaining stages must be skipped if a given stage fails."""
        if cls.is_blocking_failure(failed_stage):
            return list(remaining_stages)

        skipped: List[PipelineStage] = []
        for stage in remaining_stages:
            deps = cls.get_dependencies(stage)
            if failed_stage in deps:
                skipped.append(stage)

        return skipped
