# Changelog

All notable changes to the **GlassBox-BI** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0-alpha] - Phase 2: Dataset Ingestion & Generic Data Foundation (2026-09-13)

### Added
- **Deterministic Synthetic Retail Dataset Generator**:
  - Implemented `scripts/generate_synthetic_retail_data.py` (`seed=42`) producing exactly 50,000 data rows to `data/raw/synthetic/retail_50k.csv` (3.42 MB).
  - Multi-component additive/multiplicative demand formula: base demand, trend, weekly/yearly seasonality, promo boost, holiday spikes, price elasticity, store effects, and realistic noise.
  - Zero Data Leakage: Inventory and features do not use future target values.
  - Generated lightweight, 150-row representative sample dataset `data/sample/retail_sample.csv` tracked in Git.
- **Git Safety Configuration**:
  - Updated `.gitignore` to strictly exclude `data/raw/` and `data/processed/`, while explicitly preserving `data/sample/`.
- **Universal Canonical Business Data Contract**:
  - Defined `BusinessTimeSeriesRecord` in `backend/app/schemas/data_contract.py` with required fields (`date`, `entity_id`, `target`) and optional business features (`product_id`, `category`, `region`, `store_type`, `price`, `promotion`, `holiday`, `inventory`, `additional_features`).
- **Dataset-Specific Column Mapping Layer**:
  - Implemented `ColumnMapping` adapter model decoupling GlassBox-BI from specific vendors.
  - Pre-configured benchmark templates: `SYNTHETIC_RETAIL_MAPPING`, `WALMART_MAPPING_TEMPLATE`, `ROSSMANN_MAPPING_TEMPLATE`, `DEFAULT_CANONICAL_MAPPING`.
  - Implemented `auto_detect_column_mapping` heuristic synonym matcher for zero-configuration ingestion.
- **Multi-Dimensional Validation Engine**:
  - Implemented `DataValidator` in `backend/app/data_processing/validation.py` covering schema integrity, null rates, exact & key duplicate rows, date parsing/sorting, non-negative target validation, price/inventory ranges, and time-series length.
- **Temporal Lookahead Leakage Detection**:
  - Implemented `TemporalLeakageDetector` in `backend/app/data_processing/leakage.py` flagging future lookahead keywords, post-observation timestamps, and duplicate targets, prescribing chronological walk-forward splitting.
- **Deterministic & Explainable Data Quality Scoring**:
  - Implemented `DataQualityScorer` in `backend/app/data_processing/quality.py` calculating a 0–100 quality score across Schema, Missing Values, Duplicates, Temporal Integrity, Numeric Validity, and Consistency with transparent deduction logs.
- **Statistical Data Profiling Service**:
  - Implemented `DataProfiler` in `backend/app/data_processing/profiling.py` computing descriptive statistics, percentiles, IQR outlier counts, cardinalities, and time frequency inference.
- **Source-Agnostic Ingestion Service**:
  - Implemented `DatasetIngestionService` in `backend/app/data_processing/ingestion.py` orchestrating CSV parsing, mapping resolution, validation, leakage checks, profiling, and quality scoring with graceful error handling.
- **API v1 Dataset Endpoints**:
  - Mounted `/api/v1/datasets/` in `backend/app/api/v1/api.py` with endpoints for file upload (`/ingest/file`), local file ingestion (`/ingest/local`), committed sample summary (`/sample-summary`), and mapping templates discovery (`/mapping-templates`).
- **Automated Test Coverage**:
  - Added `tests/unit/test_synthetic_generator.py` (7 tests).
  - Added `tests/unit/test_validation.py` (9 tests).
  - Added `tests/unit/test_ingestion.py` (11 tests).
  - Total test suite expanded to 39 unit tests passing cleanly with 100% success rate.

---

## [0.2.0-alpha] - Phase 1: Application Skeleton (2026-09-13)

### Added
- **Backend Application Foundation (FastAPI)**:
  - Configured asynchronous lifespan management, central structured logging, and global exception handlers.
  - Implemented modular `/api/v1` router aggregator in `backend/app/api/v1/api.py`.
  - Added primary health check endpoints: `GET /health` and `GET /api/v1/health`.
  - Enabled CORS middleware supporting local frontend development origins.
- **Pydantic v2 Contract Layer (`backend/app/schemas/`)**:
  - Initialized contracts for `HealthResponse`, `DatasetMetadata`, `ForecastRequest`, `ForecastResult`, `ExplanationResult`, `RecommendationResult`.
- **Frontend Dashboard Shell (Next.js + TypeScript + Tailwind CSS)**:
  - Initialized Next.js 16 (App Router) in `frontend/` with custom glassmorphism design tokens.
  - Created `Sidebar`, `Header`, `BackendStatus`, `ArchitectureCard`, and `ContractsViewer` components.
- **Automated & Integration Testing**:
  - Tests for `/health`, `/api/v1/health`, contract specs, and schema validation.

---

## [0.1.0-alpha] - Phase 0: Project Foundation & Development Governance (2026-09-13)

### Added
- **Repository Governance**:
  - Initialized `.gitignore`, `.env.example`, `README.md`, `PROJECT_STATE.md`, `CHANGELOG.md`.
- **Modular Directory Skeleton**:
  - Established `backend/app/`, `frontend/`, `data/`, `models/`, `docs/`, `tests/`.
- **Architectural Documentation**:
  - Created `docs/architecture.md` and 13-phase roadmap in `docs/development_phases.md`.

---

## [Upcoming Releases]

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
