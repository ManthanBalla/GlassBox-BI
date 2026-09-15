"""API v1 Router Aggregator."""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import (
    health,
    contracts,
    ingestion,
    processing,
    forecasting,
    evaluation,
    explainability,
    decisions,
    orchestration,
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(contracts.router, prefix="/contracts", tags=["Contracts"])
api_router.include_router(ingestion.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(processing.router, prefix="/process", tags=["Processing"])
api_router.include_router(forecasting.router, prefix="/forecast", tags=["Forecasting"])
api_router.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation"])
api_router.include_router(explainability.router, prefix="/explainability", tags=["Explainability"])
api_router.include_router(decisions.router, prefix="/decisions", tags=["Decisions"])
api_router.include_router(orchestration.router, prefix="/orchestration", tags=["Orchestration"])

