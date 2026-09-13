"""Contract specification discovery endpoint for API v1."""

from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()


@router.get("/specs", summary="List Future Module Contracts")
def get_contract_specs() -> Dict[str, Any]:
    """Provides metadata about the future module communication contracts defined in Phase 1."""
    return {
        "contracts": [
            {
                "name": "DatasetMetadata",
                "phase": "Phase 2 — Dataset Ingestion",
                "description": "Schema for tabular/time-series dataset metadata and cataloging.",
            },
            {
                "name": "ForecastRequest",
                "phase": "Phase 4 — Forecasting Agent",
                "description": "Specification for initiating multi-step horizon forecasting tasks.",
            },
            {
                "name": "ForecastResult",
                "phase": "Phase 4 & 5 — Forecasting & Evaluation",
                "description": "Payload schema containing predictions, bounds, and accuracy metrics.",
            },
            {
                "name": "ExplanationResult",
                "phase": "Phase 6 — Explainability Agent",
                "description": "Schema for feature attributions, SHAP scores, and decomposition.",
            },
            {
                "name": "RecommendationResult",
                "phase": "Phase 7 — Decision Intelligence Agent",
                "description": "Prescriptive action items and scenario simulation results.",
            },
        ]
    }
