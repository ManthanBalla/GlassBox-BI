"""Decision Intelligence Schemas for GlassBox-BI (Phase 7).

Defines structured Pydantic contracts for Business Context, Prescriptive Recommendations,
Decision Scoring, Evidence Traceability, Trade-Offs, What-If Scenarios, and Audit Records.
All decision logic is deterministic, transparent, and model-agnostic with zero LLM dependency.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.forecasting import ForecastResult
from backend.app.schemas.explainability import ExplanationResult


class DecisionPriority(str, Enum):
    """Deterministic urgency levels based on quantifiable business risk."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class BusinessActionCategory(str, Enum):
    """Functional taxonomy for prescriptive business actions."""
    INVENTORY = "INVENTORY"
    PRICING = "PRICING"
    PROMOTION = "PROMOTION"
    MONITORING = "MONITORING"
    RISK = "RISK"
    PLANNING = "PLANNING"


class BusinessContext(BaseModel):
    """Operational business constraints and state parameters.
    
    Generic structure supporting retail inventory management and extensible to SME cash-flow.
    Missing fields are handled gracefully and never fabricated.
    """
    current_inventory: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Current on-hand physical inventory units",
    )
    reorder_point: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Inventory threshold triggering standard replenishment",
    )
    safety_stock: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Buffer inventory required to absorb demand surges",
    )
    lead_time_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=365,
        description="Replenishment fulfillment duration in days",
    )
    current_price: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Active selling price per unit",
    )
    unit_cost: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Unit acquisition / procurement cost",
    )
    promotion_active: Optional[bool] = Field(
        default=None,
        description="Whether a promotional campaign or discount is currently active",
    )
    supplier_capacity: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum order quantity fulfillment capacity per cycle",
    )
    minimum_order_quantity: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Minimum order batch quantity imposed by supplier",
    )
    budget_limit: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum available procurement expenditure budget",
    )
    holding_cost_rate: Optional[float] = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Estimated annual or cycle inventory holding cost fraction",
    )
    stockout_cost_per_unit: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated penalty or lost margin per unmet customer demand unit",
    )
    service_level_target: Optional[float] = Field(
        default=0.95,
        ge=0.50,
        le=0.999,
        description="Target customer order fulfillment probability",
    )
    business_type: str = Field(
        default="retail",
        description="Domain context: retail, sme_finance, manufacturing",
    )
    entity_id: Optional[str] = Field(default=None, description="Store or entity identifier")
    product_id: Optional[str] = Field(default=None, description="Product SKU identifier")


class TradeOff(BaseModel):
    """Explicit business balance between expected benefit and operational cost/risk."""
    benefit: str = Field(..., description="Primary upside or advantage achieved by the action")
    trade_off: str = Field(..., description="Associated cost, exposure, or opportunity trade-off")
    quantified_impact: Optional[str] = Field(
        default=None,
        description="Quantified delta (e.g. holding cost delta vs stockout risk reduction)",
    )


class RecommendationItem(BaseModel):
    """Machine-readable prescriptive recommendation for an operational business action."""
    recommendation_id: str = Field(..., description="Unique recommendation identifier")
    action: str = Field(..., description="Specific recommended operational action")
    category: BusinessActionCategory = Field(..., description="Business action taxonomy")
    priority: DecisionPriority = Field(..., description="Urgency level: CRITICAL, HIGH, MEDIUM, LOW")
    rationale: str = Field(..., description="Transparent, plain-language business justification")
    evidence: List[str] = Field(
        default_factory=list,
        description="Traceable list linking forecast, business context, and XAI contributions",
    )
    expected_impact: Optional[str] = Field(default=None, description="Estimated operational or financial impact")
    risk: str = Field(..., description="Identified business risk if action is taken or omitted")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Decision confidence score (separate from forecast uncertainty)",
    )
    recommendation_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Rule-based decision strength index (0-100, not a statistical probability)",
    )
    supporting_forecast: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of forecast dynamics (trend, total demand, peak, uncertainty width)",
    )
    supporting_features: List[str] = Field(
        default_factory=list,
        description="Key Phase 6 feature attributions driving this decision",
    )
    trade_offs: Optional[TradeOff] = Field(
        default=None,
        description="Explicit trade-off analysis (benefit vs cost/risk)",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Documented assumptions used when context or parameters are missing",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Operational alerts emitted during decision evaluation",
    )
    requires_human_review: bool = Field(
        default=False,
        description="Whether this recommendation mandates human manager approval",
    )
    rule_id: str = Field(..., description="Identifier of the deterministic rule that generated this recommendation")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionAuditRecord(BaseModel):
    """Reproducibility audit record documenting inputs, evaluated rules, and execution context."""
    decision_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entity_id: Optional[str] = None
    product_id: Optional[str] = None
    model_name: Optional[str] = None
    rules_evaluated: List[str] = Field(default_factory=list)
    rules_triggered: List[str] = Field(default_factory=list)
    evidence_used: List[str] = Field(default_factory=list)
    primary_rule_id: Optional[str] = None
    decision_confidence: float = 0.0
    recommendation_score: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    requires_human_review: bool = False
    execution_duration_ms: float = 0.0


class DecisionResult(BaseModel):
    """Complete machine-readable output contract for a Decision Intelligence run."""
    decision_id: str = Field(..., description="Unique decision execution identifier")
    entity_id: Optional[str] = None
    product_id: Optional[str] = None
    forecast_id: Optional[str] = None
    primary_recommendation: Optional[RecommendationItem] = Field(
        default=None,
        description="Highest priority recommendation requiring immediate attention",
    )
    recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description="Full ranked list of triggered recommendations",
    )
    context_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sanitized summary of business context evaluated",
    )
    forecast_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of forecast values, trend, and uncertainty metrics",
    )
    explanation_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of Phase 6 attribution drivers and explanation fidelity",
    )
    rules_evaluated: List[str] = Field(default_factory=list, description="All decision rules checked")
    rules_triggered: List[str] = Field(default_factory=list, description="Rules meeting execution criteria")
    audit_trail: DecisionAuditRecord = Field(..., description="Audit log for complete reproducibility")
    warnings: List[str] = Field(default_factory=list, description="Aggregated operational warnings")
    requires_human_review: bool = Field(
        default=False,
        description="True if any triggered recommendation requires human sign-off",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionRequest(BaseModel):
    """Universal contract for requesting decision intelligence recommendations."""
    dataset_id: str = Field(default="retail_dataset", description="Dataset identifier")
    dataset_path: Optional[str] = Field(default=None, description="Custom dataset path if applicable")
    entity_id: Optional[str] = Field(default="STORE_001", description="Store/Entity identifier")
    product_id: Optional[str] = Field(default="PROD_001", description="Product SKU identifier")
    forecast_result: Optional[ForecastResult] = Field(
        default=None,
        description="Existing Phase 4/5 ForecastResult to evaluate",
    )
    explanation_result: Optional[ExplanationResult] = Field(
        default=None,
        description="Existing Phase 6 ExplanationResult to incorporate",
    )
    business_context: Optional[BusinessContext] = Field(
        default=None,
        description="Operational context (inventory, reorder point, lead time, pricing)",
    )
    confidence_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Threshold below which recommendations mandate human review",
    )
    uncertainty_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=2.0,
        description="Relative uncertainty interval width triggering conservative planning rules",
    )
    fidelity_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Explanation fidelity threshold below which fidelity warnings are emitted",
    )


class ScenarioRequest(BaseModel):
    """Request contract for what-if decision simulation."""
    scenario_name: str = Field(..., description="Descriptive scenario name (e.g., 'Inventory Cut 30%')")
    base_request: DecisionRequest = Field(..., description="Baseline decision request")
    inventory_delta: Optional[float] = Field(
        default=None,
        description="Absolute or relative change to current inventory",
    )
    price_delta_percent: Optional[float] = Field(
        default=None,
        description="Percentage change in price (e.g. -10 for 10% discount)",
    )
    promotion_override: Optional[bool] = Field(
        default=None,
        description="Override active promotion flag in business context",
    )
    lead_time_override: Optional[int] = Field(
        default=None,
        description="Override supplier fulfillment lead time in days",
    )


class ScenarioResult(BaseModel):
    """Comparative outcome of what-if scenario simulation vs baseline."""
    scenario_id: str = Field(..., description="Unique scenario simulation identifier")
    scenario_name: str
    baseline_decision: DecisionResult
    scenario_decision: DecisionResult
    delta_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured comparison between baseline and scenario recommendations",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionConfigSchema(BaseModel):
    """Configurable parameters, thresholds, and weights for Decision Intelligence."""
    uncertainty_relative_threshold: float = Field(
        default=0.40,
        description="Relative prediction interval width (upper - lower) / prediction considered wide",
    )
    stockout_critical_coverage_days: float = Field(
        default=2.0,
        description="Inventory days-of-supply below which stockout risk is CRITICAL",
    )
    overstock_coverage_multiplier: float = Field(
        default=2.5,
        description="Coverage multiple relative to lead time considered excess overstock",
    )
    fidelity_warning_threshold: float = Field(
        default=0.65,
        description="Explanation fidelity below which confidence is downgraded",
    )
    default_lead_time_days: int = Field(
        default=7,
        description="Default fulfillment lead time if unspecified in context",
    )
    holding_cost_default_rate: float = Field(
        default=0.15,
        description="Default inventory holding cost fraction",
    )
