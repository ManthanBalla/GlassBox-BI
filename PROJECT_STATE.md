# GlassBox-BI — Project State & Governance Tracker

This document serves as the single source of truth for current project progress, architectural decisions, running instructions, environment configuration, and test status.

---

## 1. Current Phase

**Phase 1 — Application Skeleton**
- **Status**: Completed
- **Phase Date**: September 2026
- **Version**: `0.2.0-alpha`

---

## 2. Completed Work

### Phase 0 — Project Foundation & Governance
- [x] **Repository Verification**: Initialized Git repository on branch `main` with remote origin tracking.
- [x] **Modular Directory Structure**: Created modular directory skeleton separating backend, frontend, data, models, docs, and tests.
- [x] **Project Governance Documents**: `README.md`, `PROJECT_STATE.md`, `CHANGELOG.md`, `.gitignore`, `.env.example`.
- [x] **Technical Documentation**: `docs/architecture.md` and 13-phase roadmap in `docs/development_phases.md`.
- [x] **Automated Testing Setup**: Base test suite with directory structure sanity tests.

### Phase 1 — Application Skeleton
- [x] **FastAPI Backend Application**:
  - Operational server running with asynchronous lifespan events and structured logging.
  - Global exception handling preventing raw internal error/stack trace leaks.
  - Modular API routing structure: `/api/v1` router aggregator.
  - Health check endpoints: `GET /health` (root) and `GET /api/v1/health`.
  - CORS middleware enabled for local frontend development (`http://localhost:3000`, `http://127.0.0.1:3000`).
- [x] **Pydantic v2 Contract Layer**:
  - `HealthResponse`: System health, version, uptime timestamp.
  - `DatasetMetadata`: Contract for future dataset cataloging (Phase 2).
  - `ForecastRequest` & `ForecastResult`: Contracts for forecasting agent pipelines (Phase 4).
  - `ExplanationResult`: Contract for SHAP attributions and time-series signal decomposition (Phase 6).
  - `RecommendationResult`: Contract for scenario analysis and prescriptive insights (Phase 7).
  - Contract discovery endpoint: `GET /api/v1/contracts/specs`.
- [x] **Next.js TypeScript Frontend**:
  - Modern dashboard shell created in `frontend/` (Next.js 16, TypeScript, Tailwind CSS, App Router).
  - Glassmorphic dark UI with custom gradient tokens.
  - Navigation sidebar with future-phase badges (Dashboard, Dataset, Forecasting, Explainability, Decisions, Agent Activity).
  - Interactive Contracts Explorer displaying all Phase 1 Pydantic contracts.
  - Architecture blueprint cards for planned analytical engines.
- [x] **Frontend → Backend Communication Highway**:
  - Live connection verification pinging `GET /health`.
  - Three distinct operational states: `Checking Connection...`, `Backend Status: Connected` (with roundtrip latency in ms), and `Backend Status: Unavailable`.
  - Graceful failure handling without internal exception exposure.
  - Configurable backend target URL via `NEXT_PUBLIC_API_URL`.
- [x] **Automated & Manual Verification**:
  - 12 unit tests passing in `tests/unit/` (pytest and unittest).
  - Frontend production build compiles cleanly (`npm run build`, Turbopack).
  - Browser verification of live connected and disconnected states.

---

## 3. Pending Phases

| Phase | Description | Status |
|---|---|---|
| **Phase 2** | Dataset Ingestion | **Next Recommended Phase** |
| **Phase 3** | Data Processing Agent | Pending |
| **Phase 4** | Forecasting Agent | Pending |
| **Phase 5** | Forecast Evaluation | Pending |
| **Phase 6** | Explainability Agent | Pending |
| **Phase 7** | Decision Intelligence Agent | Pending |
| **Phase 8** | Multi-Agent Orchestration | Pending |
| **Phase 9** | Feedback & Self-Correction Loop | Pending |
| **Phase 10** | Dashboard | Pending |
| **Phase 11** | MLflow, Testing & Deployment | Pending |
| **Phase 12** | Final Integration & Validation | Pending |

---

## 4. Architecture Decisions

| ADR ID | Title | Status | Rationale |
|---|---|---|---|
| **ADR-001** | Modular Separation of Engines | Accepted | Separate data processing, forecasting, explainability, and decision intelligence into isolated Python packages to ensure independent testability and maintainability. |
| **ADR-002** | Zero Premature Logic in Phase 0 & 1 | Accepted | Strictly avoided implementing machine learning models, SHAP routines, or agent loops in Phases 0 and 1 to guarantee an uncluttered foundation. |
| **ADR-003** | FastAPI for Backend Service | Accepted | FastAPI provides high-performance asynchronous execution, native Pydantic validation, and auto-generated OpenAPI documentation. |
| **ADR-004** | Isolated Data & Model Directories | Accepted | Datasets (`data/`) and model weights (`models/`) are kept outside the source code tree and ignored by Git to prevent repository bloat. |
| **ADR-005** | Next.js App Router & Tailwind for Frontend | Accepted | Next.js with TypeScript and Tailwind CSS provides a reactive, type-safe development environment with rapid build times and native environment variable handling. |
| **ADR-006** | Pydantic v2 Schema Contracts for Module Decoupling | Accepted | Domain schemas (`DatasetMetadata`, `ForecastRequest`, etc.) serve as explicit data contracts, enabling frontend and backend development to progress with guaranteed interface stability. |

---

## 5. Known Limitations

- **No Active ML / Agent Logic**: By design, predictive models, SHAP interpretability, and agent reasoning loops are not implemented in Phase 1 and will be introduced starting from Phase 3/4.
- **Local Storage Only**: Datasets and model checkpoints currently point to local directories (`data/`, `models/`).

---

## 6. How to Run the Project

### Prerequisites
- Python 3.10+
- Node.js v18+ & npm

### Running the Backend
```bash
# 1. Install backend dependencies
pip install -r backend/requirements.txt

# 2. Configure environment (optional, defaults provided)
cp .env.example .env

# 3. Start the FastAPI development server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend will be live at `http://127.0.0.1:8000`.
- Health Check: `http://127.0.0.1:8000/health`
- Swagger Documentation: `http://127.0.0.1:8000/docs`
- Contract Discovery: `http://127.0.0.1:8000/api/v1/contracts/specs`

### Running the Frontend
```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install dependencies (if not already installed)
npm install

# 3. Start Next.js development server
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### Running Automated Tests
```bash
# Run backend test suite
pytest tests/
# or
python -m unittest discover -s tests -p "test_*.py"

# Run frontend build check
cd frontend && npm run build
```

---

## 7. Environment Variables Reference

| Variable | Default Value | Description |
|---|---|---|
| `APP_NAME` | `GlassBox-BI` | Name of the application service. |
| `APP_ENV` | `development` | Environment mode (`development`, `staging`, `production`). |
| `DEBUG` | `true` | Enables verbose debug logs and hot reload. |
| `LOG_LEVEL` | `INFO` | Application log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `SECRET_KEY` | *(placeholder)* | Cryptographic key for session/token verification. |
| `BACKEND_HOST` | `127.0.0.1` | Local listening host address for FastAPI. |
| `BACKEND_PORT` | `8000` | Port for the backend API server. |
| `ALLOWED_ORIGINS`| `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated CORS allowed origins. |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | Target FastAPI backend URL for the Next.js frontend. |
| `DATABASE_URL` | `sqlite:///./glassbox.db` | Connection string for metadata database. |
| `GEMINI_API_KEY` | *(optional/placeholder)* | API key for Gemini models (Phase 3-9 agents). |
| `OPENAI_API_KEY` | *(optional/placeholder)* | API key for OpenAI models (optional fallback). |
| `LLM_MODEL` | `gemini-2.0-flash` | Default foundation model for agentic workflows. |
| `DATA_DIR` | `./data` | Filepath root for dataset storage. |
| `MODEL_DIR` | `./models` | Filepath root for serialized models. |

---

## 8. Testing Status

- **Backend Test Framework**: `pytest` / `unittest`
  - `tests/unit/test_foundation.py`: Directory structure and package importability (3 tests) — **Passing**
  - `tests/unit/test_api_v1.py`: Root health, v1 health, contract specs, CORS headers (4 tests) — **Passing**
  - `tests/unit/test_schemas.py`: Validation of all 5 Pydantic contract schemas (5 tests) — **Passing**
- **Frontend Build**: `npm run build` (Turbopack) — **Passing** (Static pages generated, 0 TypeScript errors)
- **End-to-End Communication**: Live browser verification of connected and graceful failure states — **Passing**
