"""Multi-Agent Orchestration Package for GlassBox-BI (Phase 8).

Coordinates Phase 3 through Phase 7 into a deterministic, auditable pipeline.
"""

from backend.app.schemas.orchestration import (
    PipelineStage,
    WorkflowStatus,
    StageStatus,
    StageExecutionResult,
    OrchestrationRequest,
    OrchestrationAuditRecord,
    OrchestrationResult,
    OrchestrationState,
)
from backend.app.orchestration.base import BaseOrchestrator
from backend.app.orchestration.registry import AgentRegistry
from backend.app.orchestration.graph import WorkflowGraph
from backend.app.orchestration.validation import WorkflowValidator
from backend.app.orchestration.audit import WorkflowAuditTracker
from backend.app.orchestration.executor import SequentialWorkflowExecutor
from backend.app.orchestration.agent import MultiAgentOrchestrator

__all__ = [
    "PipelineStage",
    "WorkflowStatus",
    "StageStatus",
    "StageExecutionResult",
    "OrchestrationRequest",
    "OrchestrationAuditRecord",
    "OrchestrationResult",
    "OrchestrationState",
    "BaseOrchestrator",
    "AgentRegistry",
    "WorkflowGraph",
    "WorkflowValidator",
    "WorkflowAuditTracker",
    "SequentialWorkflowExecutor",
    "MultiAgentOrchestrator",
]
