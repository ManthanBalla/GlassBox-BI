# Changelog

All notable changes to the **GlassBox-BI** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0-alpha] - Phase 1: Application Skeleton (2026-09-13)

### Added
- **Backend Application Foundation (FastAPI)**:
  - Configured asynchronous application lifespan management, central structured logging, and global exception handlers.
  - Implemented modular `/api/v1` router aggregator in `backend/app/api/v1/api.py`.
  - Added primary health check endpoints: `GET /health` (root) and `GET /api/v1/health` returning `HealthResponse`.
  - Added schema contract discovery endpoint: `GET /api/v1/contracts/specs`.
  - Enabled CORS middleware supporting local frontend development origins (`localhost:3000`, `127.0.0.1:3000`).
  - Added `backend/app/services/` layer boundary.
- **Pydantic v2 Contract Layer (`backend/app/schemas/`)**:
  - `HealthResponse`: Diagnostic schema with version, timestamp, and phase status.
  - `DatasetMetadata`: Future dataset ingestion and cataloging specification.
  - `ForecastRequest`: Multi-step time-series forecasting parameter contract.
  - `ForecastResult`: Standardized prediction output schema with confidence bounds and metrics.
  - `ExplanationResult`: Glass-box XAI schema for SHAP attributions and signal decomposition.
  - `RecommendationResult`: Prescriptive decision intelligence contract for scenario analysis.
- **Frontend Dashboard Shell (Next.js + TypeScript + Tailwind CSS)**:
  - Initialized Next.js 16 (App Router) in `frontend/` with TypeScript, Tailwind CSS, and custom glassmorphism tokens.
  - Created responsive `Sidebar` with future-phase navigation badges.
  - Created `Header` component with GlassBox-BI title, subtitle, and runtime mode indicators.
  - Created `BackendStatus` component featuring live connectivity testing to `GET /health`, real-time latency measurement, and graceful failure handling.
  - Created `ArchitectureCard` and `ContractsViewer` components for interactive exploration of upcoming modules and Pydantic schemas.
- **Automated & Integration Testing**:
  - `tests/unit/test_api_v1.py`: Tests for `/health`, `/api/v1/health`, contract specs discovery, and CORS headers.
  - `tests/unit/test_schemas.py`: Validation of all 5 domain contract schemas and constraint checking.
  - Updated `tests/unit/test_foundation.py` to verify `schemas` and `services` directories.
  - Frontend production build verification (`npm run build`).

---

## [0.1.0-alpha] - Phase 0: Project Foundation & Development Governance (2026-09-13)

### Added
- **Repository Governance**:
  - Initialized `.gitignore` for Python, Frontend, Node, virtual environments, IDEs, and environment files.
  - Initialized `.env.example` defining configuration schema for the application, server, database, and LLM providers.
  - Created `README.md` with platform vision, architecture summary, setup instructions, and phase roadmap.
  - Created `PROJECT_STATE.md` tracking active phase, completed milestones, architectural decisions, run guides, and testing status.
  - Created `CHANGELOG.md` for phase-by-phase version history.
- **Modular Directory Skeleton**:
  - Established `backend/app/` package structure with clean architectural boundaries.
  - Created `backend/requirements.txt` with baseline dependencies.
  - Created `backend/app/main.py` providing a lightweight baseline health-check server.
  - Created `frontend/` directory placeholder.
  - Created `data/` and `models/` directory trees.
- **Architectural Documentation**:
  - `docs/architecture.md`: Detailed system architecture, Mermaid dataflow diagram, module boundaries, and design principles.
  - `docs/development_phases.md`: 13-phase implementation roadmap spanning Phase 0 through Phase 12 with acceptance criteria.
- **Automated Sanity Testing**:
  - Initialized `tests/` hierarchy with `tests/unit/test_foundation.py` to verify package importability and folder structure completeness.

---

## [Upcoming Releases]

- **Phase 2 (v0.3.0-alpha)**: Dataset Ingestion
- **Phase 3 (v0.4.0-alpha)**: Data Processing Agent
- **Phase 4 (v0.5.0-alpha)**: Forecasting Agent
- **Phase 5 (v0.6.0-alpha)**: Forecast Evaluation
- **Phase 6 (v0.7.0-alpha)**: Explainability Agent
- **Phase 7 (v0.8.0-alpha)**: Decision Intelligence Agent
- **Phase 8 (v0.9.0-alpha)**: Multi-Agent Orchestration
- **Phase 9 (v0.10.0-alpha)**: Feedback & Self-Correction Loop
- **Phase 10 (v0.11.0-alpha)**: Dashboard
- **Phase 11 (v0.12.0-alpha)**: MLflow, Testing & Deployment
- **Phase 12 (v1.0.0)**: Final Integration & Validation
