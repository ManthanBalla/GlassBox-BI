"""Explainability API Endpoints for GlassBox-BI (Phase 6).

Exposes REST endpoints for local and global model explanations using SHAP, LIME,
and component-based decomposition, with model compatibility checks and sample endpoints.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status
import shap

from backend.app.explainability.agent import ExplainabilityAgent
from backend.app.explainability.validation import MODEL_COMPATIBILITY
from backend.app.schemas.explainability import (
    ExplainabilityConfigSchema,
    ExplanationResult,
    GlobalExplanationRequest,
    LocalExplanationRequest,
    MethodCompatibilityInfo,
)

router = APIRouter()
_agent = ExplainabilityAgent()


@router.get(
    "/health",
    summary="Explainability Subsystem Health",
    response_model=Dict[str, Any],
)
def get_explainability_health() -> Dict[str, Any]:
    """Returns the operational status of the Explainability Agent and installed libraries."""
    return {
        "status": "HEALTHY",
        "module": "Phase 6 — Explainability Agent",
        "libraries": {
            "shap": shap.__version__,
            "lime": "0.2.0.1",
        },
        "supported_methods": ["shap", "lime", "component_based"],
        "supported_models": ["lightgbm", "lstm", "prophet"],
    }


@router.get(
    "/methods",
    summary="Explainability Method Compatibility Matrix",
    response_model=Dict[str, MethodCompatibilityInfo],
)
def get_method_compatibility() -> Dict[str, MethodCompatibilityInfo]:
    """Provides explicit model-to-method compatibility rules ensuring scientific honesty."""
    return MODEL_COMPATIBILITY


@router.get(
    "/config",
    summary="Explainability Default Configuration",
    response_model=ExplainabilityConfigSchema,
)
def get_explainability_config() -> ExplainabilityConfigSchema:
    """Returns default sampling sizes, random seeds, and fidelity thresholds."""
    return ExplainabilityConfigSchema()


@router.post(
    "/local",
    summary="Generate Local Forecast Explanation",
    response_model=ExplanationResult,
)
def explain_local_forecast(request: LocalExplanationRequest) -> ExplanationResult:
    """Generates an instance-level local explanation (SHAP, LIME, or Component) for a forecast."""
    try:
        result = _agent.explain_local(request)
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability execution error: {str(exc)}",
        )


@router.post(
    "/global",
    summary="Generate Global Feature Importance",
    response_model=ExplanationResult,
)
def explain_global_forecast(request: GlobalExplanationRequest) -> ExplanationResult:
    """Calculates global aggregate feature importance over a reference sample."""
    try:
        result = _agent.explain_global(request)
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Global explainability execution error: {str(exc)}",
        )


@router.get(
    "/sample",
    summary="Sample Local Explanation",
    response_model=ExplanationResult,
)
def get_sample_explanation() -> ExplanationResult:
    """Returns a fast sample local explanation on the synthetic retail dataset for instant UI testing."""
    sample_req = LocalExplanationRequest(
        entity_id="STORE_001",
        product_id="PROD_001",
        model_name="lightgbm",
        method="shap",
        background_samples=25,
    )
    try:
        return _agent.explain_local(sample_req)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate sample explanation: {str(exc)}",
        )
