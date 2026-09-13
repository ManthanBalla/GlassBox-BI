# Changelog

All notable changes to the **GlassBox-BI** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  - Established `backend/app/` package structure with clean architectural boundaries:
    - `core/`: Settings and configurations.
    - `api/`: API routing and endpoint schemas.
    - `data_processing/`: Modular placeholder for data ingestion, cleaning, and transformation.
    - `forecasting/`: Modular placeholder for time-series forecasting models.
    - `explainability/`: Modular placeholder for XAI, SHAP, and signal decomposition.
    - `decision_intelligence/`: Modular placeholder for what-if simulations and recommendations.
    - `agents/`: Modular placeholder for autonomous agent roles.
    - `orchestration/`: Modular placeholder for multi-agent execution graphs.
  - Created `backend/requirements.txt` with baseline dependencies for FastAPI, Pydantic, and Pytest.
  - Created `backend/app/main.py` providing a lightweight baseline health-check server.
  - Created `frontend/` directory with `.gitkeep` ready for UI initialization.
  - Created `data/raw/` and `data/processed/` with `.gitkeep` for dataset management.
  - Created `models/checkpoints/` and `models/saved/` with `.gitkeep` for model persistence.
- **Architectural Documentation**:
  - `docs/architecture.md`: Detailed system architecture, Mermaid dataflow diagram, module boundaries, and design principles.
  - `docs/development_phases.md`: Multi-phase implementation roadmap spanning Phase 0 through Phase 7 with acceptance criteria.
- **Automated Sanity Testing**:
  - Initialized `tests/` hierarchy with `tests/unit/test_foundation.py` to verify package importability and folder structure completeness.

---

## [Upcoming Releases]

- **Phase 1 (v0.2.0-alpha)**: Data Processing & Ingestion Engine
- **Phase 2 (v0.3.0-alpha)**: Forecasting Engine (Statistical & ML Baselines)
- **Phase 3 (v0.4.0-alpha)**: Explainability Engine (XAI & SHAP Decomposition)
- **Phase 4 (v0.5.0-alpha)**: Decision Intelligence Engine
- **Phase 5 (v0.6.0-alpha)**: Multi-Agent Orchestration & Feedback Loops
- **Phase 6 (v0.7.0-alpha)**: Frontend BI Dashboard
- **Phase 7 (v1.0.0)**: End-to-End Integration, Validation & Final Release
