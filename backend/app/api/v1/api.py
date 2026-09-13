"""API v1 Router Aggregator."""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import health, contracts, ingestion, processing, forecasting

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(contracts.router, prefix="/contracts", tags=["Contracts"])
api_router.include_router(ingestion.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(processing.router, prefix="/process", tags=["Processing"])
api_router.include_router(forecasting.router, prefix="/forecast", tags=["Forecasting"])
