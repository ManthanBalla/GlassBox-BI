"""API v1 Router Aggregator."""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import health, contracts

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(contracts.router, prefix="/contracts", tags=["Contracts"])
