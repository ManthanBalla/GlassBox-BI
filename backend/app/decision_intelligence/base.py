"""Base Abstract Interface for Decision Intelligence Engines in GlassBox-BI.

Defines the contract for model-agnostic, deterministic decision engines.
Future decision engines (e.g. SME Cash-Flow, Supply Chain Optimization) can implement
this contract without modifying the core Decision Intelligence Agent.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from backend.app.schemas.decisions import (
    BusinessContext,
    DecisionConfigSchema,
    DecisionResult,
    RecommendationItem,
)
from backend.app.schemas.explainability import ExplanationResult
from backend.app.schemas.forecasting import ForecastResult


class BaseDecisionEngine(ABC):
    """Abstract interface defining the contract for prescriptive decision engines."""

    def __init__(self, config: Optional[DecisionConfigSchema] = None) -> None:
        self.config = config or DecisionConfigSchema()

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Returns the unique identifier of this decision engine."""
        pass

    @property
    @abstractmethod
    def supported_domains(self) -> List[str]:
        """Returns list of supported domain types (e.g. ['retail', 'sme_finance'])."""
        pass

    @abstractmethod
    def evaluate_business_context(
        self,
        context: Optional[BusinessContext],
        forecast_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluates operational metrics (e.g., inventory coverage, stockout risk, price dynamics)."""
        pass

    @abstractmethod
    def generate_recommendations(
        self,
        forecast: ForecastResult,
        explanation: Optional[ExplanationResult],
        context: Optional[BusinessContext],
        config: Optional[DecisionConfigSchema] = None,
    ) -> List[RecommendationItem]:
        """Evaluates deterministic business rules and generates ranked recommendations."""
        pass

    @abstractmethod
    def score_recommendation(
        self,
        recommendation: RecommendationItem,
        context: Optional[BusinessContext],
        forecast: ForecastResult,
    ) -> float:
        """Calculates transparent rule-based decision strength score (0-100)."""
        pass

    @abstractmethod
    def validate_decision(self, result: DecisionResult) -> List[str]:
        """Validates that output conforms to safety and evidence integrity contracts."""
        pass
