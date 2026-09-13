"""GlassBox-BI Backend Application.

Phase 1 — Application Skeleton.
Provides modular API routing (/api/v1/), health diagnostic endpoints,
Pydantic contracts, and CORS configuration for frontend communication.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.schemas.health import HealthResponse
from backend.app.api.v1.api import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("glassbox-bi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan management."""
    logger.info("Initializing %s v%s in [%s] mode", settings.app_name, settings.version, settings.app_env)
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=f"{settings.app_name} Backend API",
    description="Multi-Agent Explainable AI Framework for Business Analytics, Forecasting, and Decision Intelligence",
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler preventing internal stack trace leaks."""
    logger.error("Unhandled error processing request %s %s: %s", request.method, request.url.path, str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "An internal server error occurred. Please try again later."},
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"], summary="Root Health Check")
def health_check() -> HealthResponse:
    """Primary health check endpoint verifying backend availability."""
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.version,
        phase="Phase 1 — Application Skeleton",
    )


# Mount versioned API router
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.debug,
    )
