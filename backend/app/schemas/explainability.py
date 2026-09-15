"""Pydantic Schemas and Contracts for Explainability Agent (Phase 6).

Defines structured contracts for SHAP, LIME, and component-based explainability,
distinguishing local instance-level feature attributions, global feature importance rankings,
reconstruction fidelity scores, model compatibility matrices, and immutable audit trails.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class ExplanationMethod(str, Enum):
    """Supported explanation algorithms and strategies."""
    SHAP = "shap"
    LIME = "lime"
    COMPONENT_BASED = "component_based"
    AUTO = "auto"


class ExplanationType(str, Enum):
    """Scope of explanation."""
    LOCAL = "local"
    GLOBAL = "global"


class FeatureContribution(BaseModel):
    """Local attribution of a specific feature to an individual model prediction."""
    feature: str = Field(..., description="Feature identifier (e.g. lag_7, rolling_mean_7, trend)")
    value: Optional[float] = Field(default=None, description="Actual feature value for the explained observation")
    contribution: float = Field(..., description="Signed contribution value to the prediction (e.g. SHAP value or LIME weight)")
    shap_value: Optional[float] = Field(default=None, description="Direct SHAP attribution value if method is SHAP")
    absolute_contribution: float = Field(..., description="Magnitude of the feature contribution (|contribution|)")
    direction: str = Field(..., description="'positive' if contribution > 0, 'negative' if < 0, 'neutral' if 0")
    rank: int = Field(..., ge=1, description="Rank ordered by absolute contribution descending (1 is highest)")

    @model_validator(mode="before")
    @classmethod
    def reconcile_contribution(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "shap_value" in data and data.get("contribution") is None:
                data["contribution"] = data["shap_value"]
            elif "contribution" in data and data.get("shap_value") is None:
                data["shap_value"] = data["contribution"]
            if "contribution" in data and "absolute_contribution" not in data:
                data["absolute_contribution"] = round(abs(float(data["contribution"])), 4)
            if "contribution" in data and "direction" not in data:
                c = float(data["contribution"])
                data["direction"] = "positive" if c > 1e-6 else ("negative" if c < -1e-6 else "neutral")
        return data


class GlobalFeatureImportance(BaseModel):
    """Global aggregate importance of a feature across a reference dataset."""
    feature: str = Field(..., description="Feature identifier")
    importance_score: float = Field(..., description="Aggregate importance metric (e.g. mean(|SHAP|) or mean(|LIME|))")
    rank: int = Field(..., ge=1, description="Rank ordered by importance descending (1 is most important)")
    normalized_importance: Optional[float] = Field(
        default=None,
        description="Relative importance normalized across all features (sums to 1.0)",
    )


class ExplanationFidelity(BaseModel):
    """Quantitative evaluation of explainer fidelity and surrogate reconstruction quality."""
    fidelity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized fidelity metric (1.0 = perfect reconstruction of model output)",
    )
    reconstruction_error: float = Field(
        ...,
        ge=0.0,
        description="Absolute difference between model prediction and sum of base value + contributions",
    )
    surrogate_r2: Optional[float] = Field(
        default=None,
        description="R-squared goodness of fit of the local surrogate model (LIME)",
    )
    base_value: Optional[float] = Field(
        default=None,
        description="Base / expected model prediction across the background reference sample",
    )
    reconstructed_prediction: Optional[float] = Field(
        default=None,
        description="Sum of base_value and all feature contributions",
    )
    actual_prediction: Optional[float] = Field(
        default=None,
        description="Actual output produced by the forecasting model",
    )
    explanation_status: str = Field(
        default="HIGH_FIDELITY",
        description="Status indicator: HIGH_FIDELITY, MODERATE_FIDELITY, APPROXIMATE, or DECOMPOSED",
    )
    method_notes: Optional[str] = Field(
        default=None,
        description="Method-specific documentation explaining how fidelity was evaluated",
    )


class ExplanationAuditTrail(BaseModel):
    """Immutable audit trail payload for model explainability operations."""
    explanation_id: str = Field(..., description="Unique explanation task identifier")
    forecast_id: Optional[str] = Field(default=None, description="Associated forecast identifier if linked")
    model_name: str = Field(..., description="Explained forecasting model (lightgbm, prophet, lstm)")
    model_type: Optional[str] = Field(default=None, description="Architecture class family")
    entity_id: Optional[str] = Field(default=None, description="Business entity / store identifier")
    product_id: Optional[str] = Field(default=None, description="Product / SKU identifier")
    prediction_date: Optional[str] = Field(default=None, description="Date of the explained forecast observation")
    prediction: float = Field(..., description="Explained forecasted value")
    explanation_method: str = Field(..., description="Algorithm used: shap, lime, or component_based")
    explanation_type: str = Field(..., description="local or global")
    feature_count: int = Field(..., ge=0, description="Total number of features attributed")
    top_positive_features: List[str] = Field(default_factory=list, description="Top positive drivers by name")
    top_negative_features: List[str] = Field(default_factory=list, description="Top negative drivers by name")
    random_seed: Optional[int] = Field(default=42, description="Random seed used for reproducibility")
    sample_size: Optional[int] = Field(default=None, description="Number of background / reference samples used")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LocalExplanationRequest(BaseModel):
    """Request contract for explaining an individual forecast prediction."""
    dataset_id: str = Field(default="retail_dataset", description="Dataset identifier")
    dataset_path: Optional[str] = Field(default=None, description="Custom dataset path if applicable")
    entity_id: Optional[str] = Field(default="STORE_001", description="Store/Entity identifier")
    product_id: Optional[str] = Field(default="PROD_001", description="Product/SKU identifier")
    model_name: str = Field(default="lightgbm", description="Model architecture: lightgbm, prophet, lstm")
    method: ExplanationMethod = Field(
        default=ExplanationMethod.AUTO,
        description="Explanation method: shap, lime, component_based, or auto",
    )
    prediction_date: Optional[str] = Field(
        default=None,
        description="Specific forecast date to explain (defaults to the first forecasted date)",
    )
    step_index: int = Field(
        default=0,
        ge=0,
        le=365,
        description="Forecast step index to explain (0 = step 1, 1 = step 2, etc.)",
    )
    forecast_horizon: int = Field(default=14, ge=1, le=365, description="Forecast horizon")
    background_samples: int = Field(
        default=50,
        ge=5,
        le=500,
        description="Number of historical training samples for background reference distribution",
    )
    num_lime_samples: int = Field(
        default=500,
        ge=50,
        le=5000,
        description="Number of perturbed samples for LIME local surrogate fitting",
    )
    random_seed: int = Field(default=42, description="Fixed seed for reproducible sampling and perturbations")


class GlobalExplanationRequest(BaseModel):
    """Request contract for computing global feature importance across a reference dataset."""
    dataset_id: str = Field(default="retail_dataset", description="Dataset identifier")
    dataset_path: Optional[str] = Field(default=None, description="Custom dataset path if applicable")
    entity_id: Optional[str] = Field(default="STORE_001", description="Store/Entity identifier")
    product_id: Optional[str] = Field(default="PROD_001", description="Product/SKU identifier")
    model_name: str = Field(default="lightgbm", description="Model architecture: lightgbm, prophet, lstm")
    method: ExplanationMethod = Field(
        default=ExplanationMethod.AUTO,
        description="Explanation method: shap, lime, component_based, or auto",
    )
    sample_size: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Sample size of observations used to compute average feature importance",
    )
    top_k: Optional[int] = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of top features to return in ranked output",
    )
    random_seed: int = Field(default=42, description="Fixed seed for reproducibility")


class ExplanationResult(BaseModel):
    """Machine-readable explanation payload containing attributions, fidelity, and audit trail."""
    explanation_id: str = Field(..., description="Unique explanation identifier")
    model_name: str = Field(..., description="Name of explained model architecture")
    method: str = Field(..., description="Explanation method utilized (shap, lime, component_based)")
    explanation_type: str = Field(..., description="local or global")
    prediction: float = Field(..., description="Target model prediction being explained")
    prediction_date: Optional[str] = Field(default=None, description="Date of explained forecast point")
    forecast_id: Optional[str] = Field(default=None, description="Associated forecast run identifier")
    base_value: Optional[float] = Field(
        default=None,
        description="Expected model prediction / intercept over reference background",
    )
    features: List[FeatureContribution] = Field(
        default_factory=list,
        description="Full ranked list of feature contributions for local explanations",
    )
    top_positive_contributors: List[FeatureContribution] = Field(
        default_factory=list,
        description="Top features pushing the forecast higher",
    )
    top_negative_contributors: List[FeatureContribution] = Field(
        default_factory=list,
        description="Top features pulling the forecast lower",
    )
    global_importance: List[GlobalFeatureImportance] = Field(
        default_factory=list,
        description="Ranked global feature importance list for global explanations",
    )
    feature_attributions: Dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary mapping feature names to signed contribution values (Phase 1 contract compatibility)",
    )
    decomposition: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Decomposed components if applicable",
    )
    narrative_summary: Optional[str] = Field(
        default=None,
        description="Plain-language interpretation summary",
    )
    fidelity: ExplanationFidelity = Field(..., description="Explanation fidelity and reconstruction metrics")
    audit_trail: ExplanationAuditTrail = Field(..., description="Traceability and metadata payload")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings emitted during attribution")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def populate_compatibility_attributions(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "feature_attributions" not in data or not data["feature_attributions"]:
                feats = data.get("features", [])
                if feats:
                    fa = {}
                    for f in feats:
                        if isinstance(f, dict):
                            fa[f.get("feature", "")] = float(f.get("contribution", 0.0))
                        elif hasattr(f, "feature") and hasattr(f, "contribution"):
                            fa[f.feature] = float(f.contribution)
                    data["feature_attributions"] = fa
        return data


class MethodCompatibilityInfo(BaseModel):
    """Documentation of supported explainability methods per model architecture."""
    model_name: str
    supported_methods: List[str]
    default_method: str
    native_tree_shap: bool
    sequence_attribution: bool
    component_decomposition: bool
    notes: str


class ExplainabilityConfigSchema(BaseModel):
    """Configuration options and defaults for Explainability Agent."""
    default_background_samples: int = 50
    default_lime_samples: int = 500
    default_random_seed: int = 42
    max_features_to_display: int = 20
    fidelity_threshold_high: float = 0.95
    fidelity_threshold_moderate: float = 0.80
