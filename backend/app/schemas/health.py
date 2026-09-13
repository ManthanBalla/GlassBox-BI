"""Health and diagnostic schemas."""

from datetime import datetime, timezone
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health check response schema."""
    status: str = Field(default="healthy", description="Operational status of the backend")
    app_name: str = Field(default="GlassBox-BI", description="Application name")
    version: str = Field(default="0.2.0-alpha", description="Application semantic version")
    phase: str = Field(default="Phase 1 — Application Skeleton", description="Current development phase")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp of the response")
