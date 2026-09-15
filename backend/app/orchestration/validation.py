"""Workflow Input and State Validator for Multi-Agent Orchestration (Phase 8).

Performs strict, fail-early assertions on requests, datasets, horizons, and states.
"""

from pathlib import Path
from typing import List, Optional

from backend.app.schemas.orchestration import (
    OrchestrationRequest,
    OrchestrationState,
    PipelineStage,
    WorkflowStatus,
)


class WorkflowValidator:
    """Validates orchestration requests, data source paths, and runtime state invariants."""

    SUPPORTED_MODELS = {"lightgbm", "prophet", "lstm"}
    SUPPORTED_METRICS = {"MAE", "RMSE", "MAPE"}

    @classmethod
    def validate_request(cls, request: OrchestrationRequest) -> List[str]:
        """Validates input request parameters before pipeline initiation.

        Returns:
            List of validation issue strings (empty if valid).
        """
        issues: List[str] = []

        # 1. Dataset validation
        if request.dataset_path:
            p = Path(request.dataset_path)
            if not p.exists():
                issues.append(f"Specified dataset_path does not exist: '{request.dataset_path}'")
            elif not p.is_file():
                issues.append(f"Specified dataset_path is not a valid file: '{request.dataset_path}'")
            elif not p.name.endswith(".csv"):
                issues.append(f"Specified dataset_path must be a CSV file: '{request.dataset_path}'")

        # 2. Horizon validation
        if request.horizon <= 0:
            issues.append(f"Forecast horizon must be positive, got {request.horizon}")
        elif request.horizon > 90:
            issues.append(f"Forecast horizon cannot exceed 90 days for operational safety, got {request.horizon}")

        # 3. Metric validation
        if request.selection_metric.upper() not in cls.SUPPORTED_METRICS:
            issues.append(
                f"Invalid selection_metric '{request.selection_metric}'. Supported: {sorted(cls.SUPPORTED_METRICS)}"
            )

        # 4. Confidence level
        if not (0.50 <= request.confidence_level <= 0.99):
            issues.append(f"confidence_level must be within [0.50, 0.99], got {request.confidence_level}")

        # 5. Candidate models
        if not request.candidate_models:
            issues.append("At least one candidate model must be specified in candidate_models.")
        else:
            invalid_models = [m for m in request.candidate_models if m.lower() not in cls.SUPPORTED_MODELS]
            if invalid_models:
                issues.append(
                    f"Unsupported model(s) in candidate_models: {invalid_models}. "
                    f"Supported architectures: {sorted(cls.SUPPORTED_MODELS)}"
                )

        # 6. Evaluation horizon
        if request.run_evaluation and request.evaluation_horizon is not None:
            if request.evaluation_horizon <= 0:
                issues.append(f"evaluation_horizon must be positive, got {request.evaluation_horizon}")

        # 7. Thresholds
        if not (0.10 <= request.fidelity_threshold <= 1.0):
            issues.append(f"fidelity_threshold must be in [0.10, 1.0], got {request.fidelity_threshold}")
        if not (0.05 <= request.uncertainty_threshold <= 1.0):
            issues.append(f"uncertainty_threshold must be in [0.05, 1.0], got {request.uncertainty_threshold}")

        return issues

    @classmethod
    def validate_state(cls, state: OrchestrationState) -> List[str]:
        """Validates the internal consistency and invariants of an OrchestrationState."""
        issues: List[str] = []

        # Completed vs failed disjointness
        intersection = set(state.completed_stages) & set(state.failed_stages)
        if intersection:
            issues.append(f"Stage(s) marked as both completed and failed: {[s.value for s in intersection]}")

        # Status consistency
        if state.status == WorkflowStatus.COMPLETED:
            if state.failed_stages:
                issues.append("Workflow marked as COMPLETED but contains failed_stages.")
            if not state.completed_stages:
                issues.append("Workflow marked as COMPLETED but has no completed_stages.")

        elif state.status == WorkflowStatus.FAILED:
            if not state.failed_stages and not state.errors:
                issues.append("Workflow marked as FAILED but has no recorded failed_stages or errors.")

        # Stage result keys match stage values
        for k, v in state.stage_results.items():
            if k != v.stage.value:
                issues.append(f"Stage result key '{k}' does not match inner stage enum '{v.stage.value}'")

        return issues
