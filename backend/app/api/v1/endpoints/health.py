"""Health and diagnostic endpoints for API v1."""

from fastapi import APIRouter
from backend.app.schemas.health import HealthResponse
from backend.app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="API v1 Health Check")
def get_v1_health() -> HealthResponse:
    """Returns the operational status of the GlassBox-BI API v1."""
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.version,
        phase="Phase 1 — Application Skeleton",
    )
