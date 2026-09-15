"""Abstract Base Interfaces for Multi-Agent Orchestration (Phase 8).

Defines standard contracts for workflow state machines, stage executors,
and multi-agent coordinators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from backend.app.schemas.orchestration import (
    OrchestrationRequest,
    OrchestrationResult,
    OrchestrationState,
    PipelineStage,
)


class BaseOrchestrator(ABC):
    """Abstract interface defining the lifecycle of a multi-agent orchestrator."""

    @abstractmethod
    def create_workflow(self, request: OrchestrationRequest) -> OrchestrationState:
        """Initializes a new workflow execution context and state."""
        pass

    @abstractmethod
    def execute(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Executes the full multi-agent pipeline sequentially and returns the unified result."""
        pass

    @abstractmethod
    def get_state(self, workflow_id: str) -> Optional[OrchestrationState]:
        """Retrieves current observable state for a given workflow identifier."""
        pass

    @abstractmethod
    def validate_state(self, state: OrchestrationState) -> List[str]:
        """Verifies state consistency, completed stages, and invariant adherence."""
        pass
