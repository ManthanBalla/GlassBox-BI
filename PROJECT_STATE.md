# GlassBox-BI — Project State & Governance Tracker

This document serves as the single source of truth for current project progress, architectural decisions, running instructions, environment configuration, and test status.

---

## 1. Current Phase

**Phase 0 — Project Foundation & Development Governance**
- **Status**: Completed
- **Phase Date**: September 2026
- **Version**: `0.1.0-alpha`

---

## 2. Completed Work (Phase 0)

- [x] **Repository Verification**: Inspected directory, verified Git status (`main` branch pointing to remote `https://github.com/ManthanBalla/GlassBox-BI.git`).
- [x] **Modular Directory Structure**: Created modular directory skeleton separating backend, frontend, data, models, docs, and tests.
  - `backend/app/api`: API route definitions.
  - `backend/app/core`: Configuration and logging.
  - `backend/app/data_processing`: Ingestion and preprocessing placeholder module.
  - `backend/app/forecasting`: Forecasting engine placeholder module.
  - `backend/app/explainability`: Explainability (XAI) placeholder module.
  - `backend/app/decision_intelligence`: Scenario analysis placeholder module.
  - `backend/app/agents`: Autonomous agents placeholder module.
  - `backend/app/orchestration`: Multi-agent pipeline orchestrator placeholder module.
  - `data/raw` & `data/processed`: Local data storage directories.
  - `models/checkpoints` & `models/saved`: Model storage directories.
  - `frontend`: Frontend dashboard placeholder directory.
  - `tests/unit` & `tests/integration`: Unit and integration test suites.
- [x] **Project Governance Documents**:
  - `README.md`: Overview, architecture map, setup guide, roadmap.
  - `PROJECT_STATE.md`: Living status, environment variables, run guide, testing state.
  - `CHANGELOG.md`: Chronological log of versions and phase changes.
  - `.gitignore`: Rules for Python, Frontend, IDE, virtual environments, and secrets.
  - `.env.example`: Configuration template for local setup.
- [x] **Technical Documentation**:
  - `docs/architecture.md`: Comprehensive system blueprint and dataflow specifications.
  - `docs/development_phases.md`: Phase-by-phase roadmap (Phase 0 through Phase 7).
- [x] **Automated Testing Setup**:
  - Initialized `tests/unit/test_foundation.py` verifying directory structure, import integrity, and settings sanity.
- [x] **Baseline Code**:
  - `backend/app/main.py`: Minimal FastAPI health-check application.
  - `backend/requirements.txt`: Minimal Phase 0 dependencies.

---

## 3. Pending Phases

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Data Processing & Ingestion Engine | **Next Recommended Phase** |
| **Phase 2** | Forecasting Engine (Statistical & ML Baselines) | Pending |
| **Phase 3** | Explainability Engine (SHAP & Signal Decomposition) | Pending |
| **Phase 4** | Decision Intelligence Engine (Scenario Simulation & Prescriptions) | Pending |
| **Phase 5** | Agentic Layer & Multi-Agent Orchestration | Pending |
| **Phase 6** | Frontend BI Dashboard UI | Pending |
| **Phase 7** | End-to-End Integration, Validation & Final Packaging | Pending |

---

## 4. Architecture Decisions

| ADR ID | Title | Status | Rationale |
|---|---|---|---|
| **ADR-001** | Modular Separation of Engines | Accepted | Separate data processing, forecasting, explainability, and decision intelligence into isolated Python packages to ensure independent testability and maintainability. |
| **ADR-002** | Zero Premature Logic in Phase 0 | Accepted | Strictly avoided implementing machine learning models, SHAP routines, or agent loops in Phase 0 to guarantee an uncluttered foundation. |
| **ADR-003** | FastAPI for Backend Service | Accepted | FastAPI provides high-performance asynchronous execution, native Pydantic validation, and auto-generated OpenAPI documentation. |
| **ADR-004** | Isolated Data & Model Directories | Accepted | Datasets (`data/`) and model weights (`models/`) are kept outside the source code tree and ignored by Git to prevent repository bloat. |

---

## 5. Known Issues

- None. Repository foundation is clean and verified.

---

## 6. How to Run the Project

### Setup Environment
```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or source .venv/bin/activate (Linux/macOS)

# 2. Install baseline requirements
pip install -r backend/requirements.txt

# 3. Copy environment template
cp .env.example .env
```

### Run Tests
```bash
pytest tests/
```

### Run Backend Server
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 7. Environment Variables Reference

The platform uses `.env` (derived from `.env.example`). The variables are documented below:

| Variable | Default Value | Description |
|---|---|---|
| `APP_NAME` | `GlassBox-BI` | Name of the application service. |
| `APP_ENV` | `development` | Environment mode (`development`, `staging`, `production`). |
| `DEBUG` | `true` | Enables verbose debug logs and hot reload. |
| `LOG_LEVEL` | `INFO` | Application log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `SECRET_KEY` | *(placeholder)* | Cryptographic key for session/token verification. |
| `BACKEND_HOST` | `127.0.0.1` | Local listening host address for FastAPI. |
| `BACKEND_PORT` | `8000` | Port for the backend API server. |
| `ALLOWED_ORIGINS`| `http://localhost:3000` | Comma-separated CORS allowed origins. |
| `DATABASE_URL` | `sqlite:///./glassbox.db` | Connection string for metadata database. |
| `GEMINI_API_KEY` | *(optional/placeholder)* | API key for Gemini models (Phase 5 agents). |
| `OPENAI_API_KEY` | *(optional/placeholder)* | API key for OpenAI models (optional fallback). |
| `LLM_MODEL` | `gemini-2.0-flash` | Default foundation model for agentic workflows. |
| `DATA_DIR` | `./data` | Filepath root for dataset storage. |
| `MODEL_DIR` | `./models` | Filepath root for serialized models. |

---

## 8. Testing Status

- **Framework**: `pytest`
- **Unit Tests**: `tests/unit/test_foundation.py`
  - Directory existence test: **Passing**
  - Backend package import test: **Passing**
  - Modular package boundary test: **Passing**
- **Integration Tests**: Ready for Phase 1 API integration.
- **Coverage**: 100% of Phase 0 foundation code.
