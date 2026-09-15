"""Explicit Agent Registry for Multi-Agent Orchestration (Phase 8).

Maps pipeline stages to concrete agent implementations.
Enforces explicit, compile-time registered components without dynamic reflection.
"""

from typing import Any, Callable, Dict, Optional, Type

from backend.app.schemas.orchestration import PipelineStage
from backend.app.data_processing.processing.processor import GenericBusinessDataProcessor
from backend.app.forecasting.agent import GenericForecastingAgent
from backend.app.evaluation.benchmark import ForecastingBenchmarkEngine
from backend.app.explainability.agent import ExplainabilityAgent
from backend.app.decision_intelligence.agent import DecisionIntelligenceAgent


class AgentRegistry:
    """Explicit mapping between pipeline stages and concrete agent implementations."""

    def __init__(self) -> None:
        self._registry: Dict[PipelineStage, Any] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Registers default agent instances for Phase 3 through Phase 7."""
        self._registry[PipelineStage.DATA_PROCESSING] = GenericBusinessDataProcessor()
        self._registry[PipelineStage.FORECASTING] = GenericForecastingAgent()
        self._registry[PipelineStage.EVALUATION] = ForecastingBenchmarkEngine()
        self._registry[PipelineStage.EXPLAINABILITY] = ExplainabilityAgent()
        self._registry[PipelineStage.DECISION_INTELLIGENCE] = DecisionIntelligenceAgent()

    def register(self, stage: PipelineStage, agent: Any) -> None:
        """Explicitly registers or overrides an agent implementation for a stage."""
        if not isinstance(stage, PipelineStage):
            raise TypeError(f"Invalid stage type: {type(stage)}. Expected PipelineStage enum.")
        self._registry[stage] = agent

    def get_agent(self, stage: PipelineStage) -> Any:
        """Retrieves the registered agent implementation for the given stage."""
        if stage not in self._registry:
            raise KeyError(f"No agent registered for pipeline stage '{stage.value}'.")
        return self._registry[stage]

    def has_stage(self, stage: PipelineStage) -> bool:
        """Checks if a stage has an active agent registration."""
        return stage in self._registry

    def list_registered_stages(self) -> Dict[str, str]:
        """Returns a human-readable mapping of stages to agent class names."""
        return {
            stage.value: type(agent).__name__
            for stage, agent in self._registry.items()
        }
