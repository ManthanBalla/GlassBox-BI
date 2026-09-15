# GlassBox-BI — Project State & Governance Tracker

This document serves as the single source of truth for current project progress, architectural decisions, running instructions, environment configuration, and test status.

---

## 1. Current Phase

**Phase 5 — Forecast Evaluation**
- **Status**: Completed
- **Phase Date**: September 2026
- **Version**: `0.6.0-alpha`
- **Next Phase**: Phase 6 — Explainability Agent

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
  - `POST /api/v1/datasets/ingest/file`, `POST /api/v1/datasets/ingest/local`, `GET /api/v1/datasets/sample-summary`, `GET /api/v1/datasets/mapping-templates`.

### Phase 3 — Data Processing Agent
- [x] **Generic Business Data Processor (`GenericBusinessDataProcessor`)**:
  - Decoupled from specific retailers; operates directly on `BusinessTimeSeriesRecord` canonical contract.
  - Full end-to-end pipeline: Cleaning -> Duplicate Resolution -> Invalid Value Rectification -> Missing Value Imputation -> Chronological Sorting -> Gap Integrity -> Outlier Profiling -> Leakage-Safe Feature Engineering -> Quality Score Re-evaluation -> Temporal Splitting -> Machine-Readable Audit Report.
- [x] **Data Processing Pipeline Modules (`backend/app/data_processing/processing/`)**:
  - `cleaning.py`: Type normalization, whitespace trimming, ISO date parsing, numeric coercion.
  - `duplicates.py`: Dynamic composite business key (`date` + `entity_id` + `product_id`), exact duplicate detection, conflicting record resolution, deterministic policy (`keep_first`, `keep_last`, `flag`).
  - `invalid_values.py`: Rectification of non-positive prices, negative inventory, negative target values, and binary clamping for promotion/holiday flags.
  - `missing_values.py`: Imputation via entity-grouped median, forward-fill, mode, and explicit "Unknown". Strict target safety: missing target values are never fabricated from future observations (dropped by default).
  - `outliers.py`: Group-aware IQR and Z-Score outlier detection. **Default behavior is DETECT & FLAG** (`target_outlier = 1`) to preserve legitimate promotion/holiday demand spikes.
  - `time_series.py`: Enforces strict chronological sorting, infers cadence ('D', 'W'), and computes `TimeSeriesIntegrityReport` (missing periods, largest gap, continuity percentage).
  - `feature_engineering.py`: Zero-leakage calendar features, strictly historical lag features (`lag_1`, `lag_7`, `lag_14`, `lag_28`), and rolling features (`rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28`, `rolling_std_7`, `rolling_std_28`) using `shift(1).rolling(w)`.
  - `splitting.py`: Chronological walk-forward train/val/test splitting utility (70/15/15) generating `TemporalSplitMetadata` without random shuffling or model training.
  - `audit.py`: Thread-safe `AuditTrailTracker` logging every transformation step with before/after summaries.
- [x] **API Endpoints (`backend/app/api/v1/endpoints/processing.py`)**:
  - `POST /api/v1/process/file`: Ingest & process uploaded CSV.
  - `POST /api/v1/process/local`: Process server-side CSV.
  - `GET /api/v1/process/config`: Return default `DataProcessingConfig`.
  - `GET /api/v1/process/sample-summary`: Execute processing pipeline on committed retail sample.
- [x] **High-Performance Processing Execution**:
  - `scripts/process_retail_dataset.py`: Processed 50,000 synthetic rows in **0.91 seconds** with 23 generated features.
  - Full output saved to `data/processed/synthetic/retail_processed.csv` (12.54 MB, Git ignored).
  - Sample output saved to `data/sample/retail_processed_sample.csv` (18.79 KB, Git tracked).
- [x] **Automated Testing**:
  - 61 unit tests passing across 10 modules (100% pass rate in 1.88s).

### Phase 4 — Forecasting Agent
- [x] **Common Model Interface (`BaseForecastModel`)**:
  - Defines abstract lifecycle methods: `fit`, `predict`, `evaluate_validation`, `compute_residual_prediction_interval`, `get_model_metadata`, `save_model`, `load_model`.
  - Zero model-specific branching across the core pipeline.
- [x] **Prophet Forecasting Adapter (`ProphetForecaster`)**:
  - Mapping: `date` -> `ds`, `target` -> `y`. Fits on chronological training data without lookahead.
  - Bayesian uncertainty interval projection via Stan/Prophet backend.
  - Non-negative demand clamping and JSON checkpoint serialization via `model_to_json`.
- [x] **LightGBM Forecasting Adapter (`LightGBMForecaster`)**:
  - Feature matrix incorporating calendar attributes, lags (`lag_1`, `lag_7`, `lag_14`, `lag_28`), and rolling statistics.
  - Multi-step recursive forecasting without future target leakage (future lags dynamically constructed from predictions).
  - Empirical validation-residual prediction intervals and `joblib` persistence.
- [x] **PyTorch LSTM Forecasting Adapter (`LSTMForecaster`)**:
  - Clean sequence-to-one architecture (`nn.LSTM` -> `nn.Dropout` -> `nn.Linear`).
  - **Strict Zero-Leakage Scaling**: `StandardScaler` parameters (`mean`, `std`) fitted solely on training partition.
  - Early stopping with configurable patience, deterministic random seeds, and PyTorch checkpoint persistence (`.pt`).
- [x] **Generic Forecasting Agent (`GenericForecastingAgent`)**:
  - Organization-agnostic series filtering (`entity_id` + `product_id`).
  - Minimum history verification (gracefully yields `INSUFFICIENT_HISTORY` when observations < 15).
  - Chronological walk-forward train/val/test splitting.
  - Candidate model competition and dynamic ranking based on validation metrics (**MAE**, **RMSE**, **MAPE**).
  - **Model Failure Isolation**: Errors in individual candidate forecasters do not crash the agent.
  - Winning model refit on combined (Train + Validation) partition prior to future forecasting.
  - Multi-step future horizon projection with transparent uncertainty bounds.
  - Checkpoint persistence under `models/saved/<model_name>/`.
- [x] **REST API Endpoints (`backend/app/api/v1/endpoints/forecasting.py`)**:
  - `POST /api/v1/forecast/run`: Executes full candidate competition, validation ranking, and future forecast.
  - `POST /api/v1/forecast/train`: Triggers model competition and checkpoint saving.
  - `GET /api/v1/forecast/models`: Discovers supported models, feature requirements, and history constraints.
  - `GET /api/v1/forecast/config`: Returns system default hyperparameters and selection criteria.
  - `GET /api/v1/forecast/health`: Checks status of Prophet, LightGBM, PyTorch, and local persistence directory.
  - `GET /api/v1/forecast/sample`: Fast one-click demonstration forecast on benchmark data.
- [x] **Minimal Development Frontend Panel (`ForecastingPanel.tsx`)**:
  - Entity/product series selectors, horizon selector (7, 14, 28 days), candidate toggles, metric choice (MAE/RMSE/MAPE).
  - Live execution button with loading indicators.
  - Candidate validation ranking comparison table with training durations and status.
  - Interactive SVG time-series chart rendering point predictions and shaded uncertainty bands.
  - Forecast points table with lower and upper bounds.
- [x] **Automated Testing Suite**:
  - 79 tests passing (18 new Phase 4 unit tests across models, agent competition, API, and leakage prevention).

### Phase 5 — Forecast Evaluation
- [x] **Formal Evaluation Module (`backend/app/evaluation/`)**:
  - `metrics.py`: Deterministic MAE, RMSE, and zero-safe MAPE with explicit division-by-zero protection.
  - `evaluator.py`: `FormalForecastEvaluator` assessing any `BaseForecastModel` on the holdout test set with horizon validation.
  - `benchmark.py`: `ForecastingBenchmarkEngine` coordinating multi-model benchmark across Prophet, LightGBM, and PyTorch LSTM.
  - `schemas.py`: Pydantic contracts re-exporting `BenchmarkResult`, `ModelTestEvaluation`, `EvaluationRequest`.
- [x] **Zero-Target Handling Strategy**:
  - Defensible, transparent exclusion of zero actual observations from MAPE (`|y_i| <= 1e-7`) with explicit tracking in `zero_target_count`.
- [x] **Strict Model Selection Invariant**:
  - Holdout test set is NEVER used for model selection or tuning (`test_set_used_for_selection = False`).
  - Phase 4 validation winner (`lightgbm`) is independently labeled; test metrics are reported purely for academic evidence.
- [x] **Synthetic 50K Benchmark Execution**:
  - Evaluated on 14-day holdout test set (2024-08-01 to 2024-08-14, 37 total holdout days):
    - LightGBM: Test MAE 1.7051, RMSE 2.7491, MAPE 7.40% (Phase 4 Validation Winner: Val MAE 1.8736)
    - Prophet: Test MAE 2.3206, RMSE 2.7221, MAPE 11.12% (Val MAE 2.3711)
    - PyTorch LSTM: Test MAE 2.6528, RMSE 3.4797, MAPE 12.66% (Val MAE 1.9655)
- [x] **REST API Endpoints (`backend/app/api/v1/endpoints/evaluation.py`)**:
  - `POST /api/v1/evaluation/run`: Executes test-set benchmark evaluation.
  - `GET /api/v1/evaluation/health`: Subsystem health and dependency verification.
  - `GET /api/v1/evaluation/sample`: Fast sample demonstration benchmark.
  - `GET /api/v1/evaluation/metrics`: Quantitative metric formulas and invariant declarations.
- [x] **Minimal Development Frontend Component (`EvaluationPanel.tsx`)**:
  - Series/horizon controls, candidate toggles, formal benchmark comparison table, and actual vs prediction point comparisons.
- [x] **Automated Testing Suite**:
  - 103 tests passing (24 new Phase 5 unit tests across metrics, evaluator, benchmark engine, API, and invariants). Zero regressions.

---

## 3. Pending Phases

| Phase | Description | Status |
|---|---|---|
| **Phase 6** | Explainability Agent (SHAP / Interpretability) | **Next Recommended Phase** |
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
| **ADR-009** | Outlier Detection & Flagging by Default | Accepted | Retail demand spikes during promotions and holidays are legitimate business signals. GlassBox-BI detects and flags anomalies (`target_outlier = 1`) rather than deleting or clipping them by default. |
| **ADR-010** | Shifted Rolling Windows for Zero-Leakage Features | Accepted | All rolling aggregations apply `shift(1)` prior to rolling window computation to guarantee that current-period target values never contaminate historical feature vectors. |
| **ADR-011** | Model-Agnostic Common Interface for Forecasters | Accepted | Enforces `BaseForecastModel` contract across Prophet, LightGBM, and LSTM, allowing candidate model competition, ranking, and future expansion without altering agent orchestration code. |
| **ADR-012** | Strict Validation vs Holdout Test Separation | Accepted | Phase 4 uses the validation partition exclusively for internal model ranking and selection. The holdout test partition is strictly excluded and reserved for formal benchmark evaluation in Phase 5. |
| **ADR-013** | Zero Target Leakage in Multi-Step Recursive Forecasting | Accepted | For future multi-step horizons, LightGBM and LSTM compute future lags recursively from the model's own prior predictions, strictly preventing lookahead into actual future targets. |
| **ADR-014** | PyTorch for Local Deep-Learning Stack | Accepted | On Windows / Python 3.14 environments where TensorFlow wheel distributions are unavailable, PyTorch provides native sequence modeling, deterministic seeding, and high performance. |
| **ADR-015** | Defensible Zero-Target Exclusion for Evaluation MAPE | Accepted | When ground-truth targets are zero, division by zero is mathematically undefined. GlassBox-BI excludes zeros from MAPE calculation, reporting the exact excluded count in `zero_target_count` for full auditability. |
| **ADR-016** | Strict Model Selection and Test Benchmark Decoupling | Accepted | Model selection is conducted exclusively in Phase 4 via Validation MAE. Test set evaluation in Phase 5 is reported independently as scientific benchmark evidence and never alters model selection. |

---

## 5. Known Boundaries & Limitations

- **Validation vs. Test Boundary**: Phase 4 validation metrics are utilized strictly for internal candidate model ranking and selection. Phase 5 evaluates models on the untouched test partition independently.
- **Explainability & SHAP Deferral**: While `BenchmarkResult` preserves model performance and metadata, SHAP/LIME explainability is strictly deferred to Phase 6.
- **Decision Intelligence Deferral**: Prescriptive business recommendations, inventory simulations, and risk scorings are strictly deferred to Phase 7.
- **Multi-Agent Orchestration**: Autonomous multi-agent coordination (LangGraph/LangChain) is deferred to Phase 8.
- **Development Benchmark Dataset**: The 50,000-row synthetic retail dataset is the primary development benchmark. Real benchmark datasets (Walmart, Rossmann) will be integrated in subsequent phases without modifying the canonical data contract.

---

## 6. How to Run the Project

### Generate and Process Synthetic Retail Dataset
```bash
# 1. Generates data/raw/synthetic/retail_50k.csv (50,000 rows) and data/sample/retail_sample.csv (150 rows)
python scripts/generate_synthetic_retail_data.py --rows 50000 --seed 42

# 2. Runs GenericBusinessDataProcessor on 50k rows, generating retail_processed.csv & sample
python scripts/process_retail_dataset.py
```

### Running Backend Server
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Endpoints:
- Health Check: `http://127.0.0.1:8000/health`
- Dataset Sample Summary: `http://127.0.0.1:8000/api/v1/datasets/sample-summary`
- Data Processing Sample Summary: `http://127.0.0.1:8000/api/v1/process/sample-summary`
- Processing Config: `http://127.0.0.1:8000/api/v1/process/config`
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
python -m unittest discover -s tests/unit
```

---

## 7. Testing Status

- **Test Framework**: `pytest` / `unittest`
  - `tests/unit/test_synthetic_generator.py`: 7 tests passing (reproducibility, 50k rows, non-negative target, column schema)
  - `tests/unit/test_validation.py`: 9 tests passing (schema validation, missing values, duplicates, date checks, leakage detection)
  - `tests/unit/test_ingestion.py`: 11 tests passing (canonical contract, column mapping, quality scorer, profiler, ingestion service, API endpoints)
  - `tests/unit/test_data_processor.py`: 11 tests passing (initialization, config defaults, missing numeric/categorical/target, duplicates, invalid values, outlier flagging, sorting, gaps, quality scores, reproducibility)
  - `tests/unit/test_feature_engineering.py`: 4 tests passing (calendar features, strictly historical lags, shift-1 rolling zero leakage, absent optional columns)
  - `tests/unit/test_temporal_splitting.py`: 2 tests passing (chronological cutoff boundaries, ratio adherence, no random shuffling)
  - `tests/unit/test_processing_api.py`: 5 tests passing (config endpoint, sample processing, local file processing, 404 handling, file upload)
  - `tests/unit/test_api_v1.py`: 4 tests passing (system health, API v1 health, contract specs discovery, CORS)
  - `tests/unit/test_schemas.py`: 5 tests passing (Pydantic contract validation)
  - `tests/unit/test_foundation.py`: 3 tests passing (directory structure and package importability)
  - **Total**: **61/61 passing** (100% pass rate in 1.88s)
- **Frontend Build**: `npm run build` — **Passing** (0 TypeScript errors)

