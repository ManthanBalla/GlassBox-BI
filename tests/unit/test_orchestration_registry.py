"""Unit Tests for Agent Registry (Phase 8)."""

import pytest

from backend.app.schemas.orchestration import PipelineStage
from backend.app.orchestration.registry import AgentRegistry
from backend.app.data_processing.processing.processor import GenericBusinessDataProcessor
from backend.app.forecasting.agent import GenericForecastingAgent
from backend.app.evaluation.benchmark import ForecastingBenchmarkEngine
from backend.app.explainability.agent import ExplainabilityAgent
from backend.app.decision_intelligence.agent import DecisionIntelligenceAgent


def test_registry_contains_all_five_stages():
    """Verify registry pre-configures all five phases by default."""
    reg = AgentRegistry()
    for stage in PipelineStage:
        assert reg.has_stage(stage) is True

    assert isinstance(reg.get_agent(PipelineStage.DATA_PROCESSING), GenericBusinessDataProcessor)
    assert isinstance(reg.get_agent(PipelineStage.FORECASTING), GenericForecastingAgent)
    assert isinstance(reg.get_agent(PipelineStage.EVALUATION), ForecastingBenchmarkEngine)
    assert isinstance(reg.get_agent(PipelineStage.EXPLAINABILITY), ExplainabilityAgent)
    assert isinstance(reg.get_agent(PipelineStage.DECISION_INTELLIGENCE), DecisionIntelligenceAgent)


def test_registry_list_registered_stages():
    """Verify list_registered_stages returns human-readable mapping."""
    reg = AgentRegistry()
    summary = reg.list_registered_stages()
    assert len(summary) == 5
    assert summary["DATA_PROCESSING"] == "GenericBusinessDataProcessor"
    assert summary["FORECASTING"] == "GenericForecastingAgent"
    assert summary["EVALUATION"] == "ForecastingBenchmarkEngine"
    assert summary["EXPLAINABILITY"] == "ExplainabilityAgent"
    assert summary["DECISION_INTELLIGENCE"] == "DecisionIntelligenceAgent"


def test_registry_custom_override():
    """Verify explicit registration of mock or alternative agent instances."""
    reg = AgentRegistry()
    dummy = object()
    reg.register(PipelineStage.FORECASTING, dummy)
    assert reg.get_agent(PipelineStage.FORECASTING) is dummy


def test_registry_invalid_stage_type():
    """Verify TypeError when registering non-PipelineStage."""
    reg = AgentRegistry()
    with pytest.raises(TypeError):
        reg.register("INVALID_STAGE", object())  # type: ignore
