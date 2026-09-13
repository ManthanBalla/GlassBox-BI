"""GlassBox-BI Backend Entry Point.

Phase 0: Provides basic application scaffolding and health check endpoints.
No business logic, forecasting models, or agent workflows are implemented in this phase.
"""

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="GlassBox-BI Backend API",
        description="Explainable Business Intelligence and Forecasting Engine",
        version="0.1.0-alpha",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {
            "status": "online",
            "project": "GlassBox-BI",
            "phase": "Phase 0 - Project Foundation",
            "version": "0.1.0-alpha",
        }

    @app.get("/health")
    def health_check():
        return {"status": "healthy"}

except ImportError:
    # Fallback minimal app representation if dependencies are not yet installed
    app = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
