# GlassBox-BI — Project State & Governance Tracker

This document serves as the single source of truth for current project progress, architectural decisions, running instructions, environment configuration, and test status.

---

## 1. Current Phase

**Phase 2 — Dataset Ingestion & Generic Data Foundation**
- **Status**: Completed
- **Phase Date**: September 2026
- **Version**: `0.3.0-alpha`

---

## 2. Completed Work

### Phase 0 — Project Foundation & Governance
- [x] **Repository Verification**: Initialized Git repository on branch `main` with remote origin tracking.
- [x] **Modular Directory Structure**: Created modular directory skeleton separating backend, frontend, data, models, docs, and tests.
- [x] **Project Governance Documents**: `README.md`, `PROJECT_STATE.md`, `CHANGELOG.md`, `.gitignore`, `.env.example`.
- [x] **Technical Documentation**: `docs/architecture.md` and 13-phase roadmap in `docs/development_phases.md`.
- [x] **Automated Testing Setup**: Base test suite with directory structure sanity tests.

### Phase 1 — Application Skeleton
- [x] **FastAPI Backend Application**: Operational server with lifespan events, structured logging, CORS middleware, global error handling.
- [x] **Pydantic v2 Contract Layer**: Core schemas (`HealthResponse`, `DatasetMetadata`, `ForecastRequest`, `ForecastResult`, `ExplanationResult`, `RecommendationResult`).
- [x] **Next.js TypeScript Frontend**: Dashboard shell with glassmorphism styling, sidebar navigation, live backend health status monitor, and contracts explorer.
- [x] **Frontend-Backend Communication**: Live health checking with latency measurement and graceful failure handling.

### Phase 2 — Dataset Ingestion & Generic Data Foundation
- [x] **Deterministic Synthetic Retail Dataset**:
  - Reusable generator: `scripts/generate_synthetic_retail_data.py` (`seed=42`).
  - Generated exactly **50,000 data rows** to `data/raw/synthetic/retail_50k.csv` (3.42 MB).
  - Multi-component demand generation: base demand + trend + weekly seasonality + yearly seasonality + promo lift + holiday spikes + price elasticity + store effects + noise.
  - Zero Data Leakage Rule: Inventory and features do not use future target values.
  - Generated committed sample dataset: `data/sample/retail_sample.csv` (150 rows, identical schema).
- [x] **Git Safety Policy**:
  - `data/raw/` and `data/processed/` are strictly ignored by `.gitignore`.
  - `data/sample/` is explicitly tracked for automated testing and documentation.
- [x] **Universal Canonical Business Data Contract**:
  - `BusinessTimeSeriesRecord` in `backend/app/schemas/data_contract.py`.
  - Required fields: `date`, `entity_id`, `target`.
  - Optional business features: `product_id`, `category`, `region`, `store_type`, `price`, `promotion`, `holiday`, `inventory`, `additional_features`.
- [x] **Dataset-Specific Column Mapping Layer**:
  - `ColumnMapping` adapter decoupling core architecture from specific vendors.
  - Pre-configured benchmark templates: `SYNTHETIC_RETAIL_MAPPING`, `WALMART_MAPPING_TEMPLATE`, `ROSSMANN_MAPPING_TEMPLATE`, `DEFAULT_CANONICAL_MAPPING`.
  - `auto_detect_column_mapping` heuristic synonym auto-detection.
- [x] **Data Validation Engine (`backend/app/data_processing/validation.py`)**:
  - Schema integrity, null rates, duplicate rows/keys, date validation, non-negative target checks, price/inventory ranges, and time-series continuity.
- [x] **Temporal Lookahead Leakage Detection (`backend/app/data_processing/leakage.py`)**:
  - Flags future-token feature names (`future_`, `lead_`, `next_`, `target_t+`), timestamps exceeding observation dates, and duplicate target columns.
  - Emits prescriptive recommendation: chronological walk-forward splitting over random cross-validation.
- [x] **Explainable Data Quality Scoring (`backend/app/data_processing/quality.py`)**:
  - Deterministic 0–100 quality score across Schema, Missing Values, Duplicates, Temporal Integrity, Numeric Validity, Consistency.
  - Complete transparent audit log explaining every deducted point.
- [x] **Statistical Profiling Service (`backend/app/data_processing/profiling.py`)**:
  - Calculates descriptive statistics, percentiles, IQR outlier counts, unique cardinalities, and time frequency inference.
- [x] **Source-Agnostic Ingestion Service (`backend/app/data_processing/ingestion.py`)**:
  - Robust CSV ingestion from paths, raw bytes, or IO buffers with graceful error handling.
- [x] **API v1 Dataset Endpoints (`backend/app/api/v1/endpoints/ingestion.py`)**:
  - `POST /api/v1/datasets/ingest/file` (multipart CSV upload)
  - `POST /api/v1/datasets/ingest/local` (server-side local file ingestion)
  - `GET /api/v1/datasets/sample-summary` (instant summary of committed sample dataset)
  - `GET /api/v1/datasets/mapping-templates` (benchmark templates discovery)
- [x] **Automated Testing**:
  - 39 unit tests passing across all test modules (`test_synthetic_generator.py`, `test_validation.py`, `test_ingestion.py`, `test_api_v1.py`, `test_schemas.py`, `test_foundation.py`).

---

## 3. Pending Phases

| Phase | Description | Status |
|---|---|---|
| **Phase 3** | Data Processing Agent | **Next Recommended Phase** |
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
| **ADR-002** | Zero Premature Logic | Accepted | Strictly avoid implementing machine learning models, SHAP routines, or agent loops ahead of their designated phases. |
| **ADR-003** | FastAPI for Backend Service | Accepted | High-performance asynchronous execution, native Pydantic validation, and auto-generated OpenAPI documentation. |
| **ADR-004** | Isolated Data & Model Directories | Accepted | Large datasets (`data/raw/`) and model weights are kept outside Git tracking to prevent repository bloat. |
| **ADR-005** | Next.js App Router & Tailwind for Frontend | Accepted | Reactive, type-safe development environment with rapid build times and native environment variable handling. |
| **ADR-006** | Pydantic v2 Schema Contracts for Module Decoupling | Accepted | Domain schemas serve as explicit data contracts, enabling frontend and backend development to progress with guaranteed interface stability. |
| **ADR-007** | Organization-Agnostic Canonical Data Contract | Accepted | GlassBox-BI is strictly decoupled from specific vendors (such as Walmart or Rossmann). All external tabular datasets map their columns into `BusinessTimeSeriesRecord` via adapters. |
| **ADR-008** | Explainable Arithmetic Data Quality Scoring | Accepted | Data quality scores are computed via deterministic, explainable arithmetic with transparent deduction logs rather than opaque machine learning models. |

---

## 5. Known Limitations

- **No Feature Engineering or Model Training**: By design, cleaning transformations, feature engineering pipelines, forecasting models, and autonomous agents are not implemented in Phase 2 (scheduled for Phases 3–9).
- **The Synthetic Dataset is for Testing**: The 50,000-row synthetic retail dataset is a development/testing dataset. Real benchmark datasets (Walmart, Rossmann) will be integrated later without modifying the canonical data contract.

---

## 6. How to Run the Project

### Generate Synthetic Retail Dataset
```bash
# Generates data/raw/synthetic/retail_50k.csv (50,000 rows) and data/sample/retail_sample.csv (150 rows)
python scripts/generate_synthetic_retail_data.py --rows 50000 --seed 42
```

### Running Backend Server
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Endpoints:
- Health Check: `http://127.0.0.1:8000/health`
- Dataset Sample Summary: `http://127.0.0.1:8000/api/v1/datasets/sample-summary`
- Mapping Templates: `http://127.0.0.1:8000/api/v1/datasets/mapping-templates`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`

### Running Frontend
```bash
cd frontend
npm run dev
```
Dashboard available at `http://localhost:3000`.

### Running Automated Tests
```bash
pytest tests/
# or
python -m unittest discover -s tests -p "test_*.py"
```

---

## 7. Testing Status

- **Test Framework**: `pytest` / `unittest`
  - `tests/unit/test_synthetic_generator.py`: 7 tests passing (reproducibility, 50k rows, non-negative target, column schema)
  - `tests/unit/test_validation.py`: 9 tests passing (schema validation, missing values, duplicates, date checks, leakage detection)
  - `tests/unit/test_ingestion.py`: 11 tests passing (canonical contract, column mapping, quality scorer, profiler, ingestion service, API endpoints)
  - `tests/unit/test_api_v1.py`: 4 tests passing (system health, API v1 health, contract specs discovery, CORS)
  - `tests/unit/test_schemas.py`: 5 tests passing (Pydantic contract validation)
  - `tests/unit/test_foundation.py`: 3 tests passing (directory structure and package importability)
  - **Total**: **39/39 passing** (100% pass rate in 0.90s)
- **Frontend Build**: `npm run build` — **Passing** (0 TypeScript errors)
