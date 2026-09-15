# Changelog

All notable changes to the **GlassBox-BI** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0-alpha] - Phase 6: Explainability Agent (2026-09-15)

### Added
- **Explainability Architecture & Base Interface (`backend/app/explainability/`)**:
  - `base.py`: Abstract `BaseExplainer` defining standard lifecycle: `explain_local`, `explain_global`, `get_metadata`, with ranked signed contributions and global importance helpers.
  - `schemas.py` & `backend/app/schemas/explainability.py`: Type-safe contracts for `LocalExplanationRequest`, `GlobalExplanationRequest`, `ExplanationResult`, `FeatureContribution`, `GlobalFeatureImportance`, `ExplanationFidelity`, `ExplanationAuditTrail`, `MethodCompatibilityInfo`.
  - `feature_adapter.py`: `ExplainabilityFeatureAdapter` extracting exact input vectors and sampling zero-leakage training backgrounds.
  - `validation.py`: `ExplainabilityValidator` and explicit `MODEL_COMPATIBILITY` matrix.
  - `agent.py`: `ExplainabilityAgent` coordinator dispatching model-specific strategies, verifying model immutability, and producing audit payloads.
- **Model-Specific Explainability Strategies**:
  - **LightGBM**: Native `shap.TreeExplainer` computing exact additive Shapley attributions across tabular calendar, lag, rolling, and business regressors. `LIMEExplainer` generating local linear surrogates with reproducible random seeds (`seed=42`).
  - **PyTorch LSTM**: Model-agnostic sequence attribution explaining the univariate lookback target sequence (`lag_1` to `lag_{lookback}`). Zero fabrication of tabular calendar/business features.
  - **Prophet**: `ProphetComponentExplainer` providing exact additive decomposition into `trend`, `weekly_seasonality`, `yearly_seasonality`, and `holiday_effects`. Formally reported as `component_based`.
- **Explanation Fidelity & Quality Metrics**:
  - Additive reconstruction error: $|\hat{y} - (\text{base\_value} + \sum \phi_i)|$.
  - Normalized fidelity score: $\max(0, 1 - \text{error} / (|\hat{y}| + 10^{-6}))$.
  - LIME local surrogate $R^2$ goodness-of-fit.
- **Model Immutability & Zero Data Leakage**:
  - Pre- and post-explanation model state snapshots verify model weights, trees, historical buffers, and scalers are 100% immutable.
  - Background reference distributions are sampled strictly from historical training partitions (`train_df`).
- **REST API Endpoints (`backend/app/api/v1/endpoints/explainability.py`)**:
  - `GET /api/v1/explainability/health`: Subsystem health and dependency verification.
  - `GET /api/v1/explainability/methods`: Explicit model-to-method compatibility matrix.
  - `GET /api/v1/explainability/config`: Default configuration and fidelity thresholds.
  - `POST /api/v1/explainability/local`: Instance-level local feature attribution.
  - `POST /api/v1/explainability/global`: Dataset-level global feature importance ranking.
  - `GET /api/v1/explainability/sample`: Fast pre-computed sample explanation.
- **Minimal Development Frontend Component (`frontend/src/components/ExplainabilityPanel.tsx`)**:
  - Model and method selectors, Local vs Global mode toggle, positive/negative impact breakdown with colored bars, ranked feature table, fidelity metrics card, and audit drawer.
- **Verification Script & Empirical Attribution**:
  - `scripts/verify_phase6_explainability.py`: Evaluated live explanations on synthetic retail dataset (`STORE_001` / `PROD_001`):
    - LightGBM + SHAP: Base Value `18.6584`, Prediction `18.3315`, Fidelity Score `99.99%`, Error `0.0025`.
    - LightGBM + LIME: Surrogate Intercept `17.8001`, Surrogate $R^2$ `0.204`.
    - PyTorch LSTM + SHAP: Sequence attributions over historical target window (`lag_1` to `lag_14`), Fidelity Score `100.0%`.
    - Prophet: Exact decomposition into `trend` (+20.7879) and `weekly_seasonality` (-1.0209).
- **Automated Testing Suite**:
  - Added 31 new Phase 6 unit tests across `test_explainability_base.py`, `test_shap_explainer.py`, `test_lime_explainer.py`, `test_prophet_explainer.py`, `test_explainability_agent.py`, and `test_explainability_api.py`.
  - Expanded total test suite from 103 to **134 passing tests** (100% pass rate, 0 regressions).

---

## [0.6.0-alpha] - Phase 5: Formal Forecasting Evaluation (2026-09-15)

### Added
- **Formal Evaluation Module (`backend/app/evaluation/`)**:
  - `metrics.py`: Exact deterministic implementations of MAE, RMSE, and zero-safe MAPE with division-by-zero protection.
  - `evaluator.py`: `FormalForecastEvaluator` assessing any `BaseForecastModel` against the holdout test partition with strict horizon bounds checking.
  - `benchmark.py`: `ForecastingBenchmarkEngine` coordinating multi-model benchmark evaluation across Prophet, LightGBM, and PyTorch LSTM.
  - `schemas.py` & `backend/app/schemas/evaluation.py`: Structured Pydantic contracts for `EvaluationMetricResult`, `EvaluationPoint`, `ModelTestEvaluation`, `EvaluationRequest`, and `BenchmarkResult`.
- **Zero-Target Handling Strategy**:
  - Transparently excludes zero actual target observations from MAPE calculation (`|y_i| <= 1e-7`) with explicit tracking in `zero_target_count`.
- **Strict Model Selection Separation Invariant**:
  - Holdout test set is NEVER used for model selection, hyperparameter tuning, or fitting (`test_set_used_for_selection = False`).
  - Phase 4 validation winner (`lightgbm`) is independently preserved; test metrics are reported purely for academic evidence.
- **REST API Endpoints (`backend/app/api/v1/endpoints/evaluation.py`)**:
  - `POST /api/v1/evaluation/run`: Runs test-set benchmark evaluation on specified series and horizon.
  - `GET /api/v1/evaluation/health`: Subsystem health and dependency verification.
  - `GET /api/v1/evaluation/sample`: Fast sample demonstration benchmark on synthetic retail dataset.
  - `GET /api/v1/evaluation/metrics`: Quantitative metric mathematical formulas and invariant declarations.
- **Frontend Verification Component (`frontend/src/components/EvaluationPanel.tsx`)**:
  - Minimal development verification panel displaying test benchmarks, Phase 4 validation winner badge, and test predictions vs actuals.
- **Holdout Test Benchmark Verification on Synthetic 50K Retail Dataset**:
  - `scripts/verify_phase5_benchmark.py`: Evaluated 14-day holdout test set (2024-08-01 to 2024-08-14, 37 total holdout days):
    - LightGBM: Test MAE = 1.7051, Test RMSE = 2.7491, Test MAPE = 7.40% (Phase 4 Validation Winner: Val MAE = 1.8736)
    - Prophet: Test MAE = 2.3206, Test RMSE = 2.7221, Test MAPE = 11.12% (Val MAE = 2.3711)
    - PyTorch LSTM: Test MAE = 2.6528, Test RMSE = 3.4797, Test MAPE = 12.66% (Val MAE = 1.9655)
  - Independent manual recalculation of MAE, RMSE, and MAPE from raw predictions strictly confirmed exact match to 4 decimal places.
- **Comprehensive Automated Test Suite**:
  - Added 24 new Phase 5 unit tests across `test_evaluation_metrics.py`, `test_forecast_evaluator.py`, `test_benchmark_engine.py`, and `test_evaluation_api.py`.
  - Expanded total test suite from 79 to **103 passing tests** (100% pass rate, 0 regressions).

---

## [0.5.0-alpha] - Phase 4: Forecasting Agent (2026-09-13)

### Added
- **Model-Agnostic Common Interface (`BaseForecastModel`)**:
  - Abstract base contract in `backend/app/forecasting/base.py` defining standard lifecycle: `fit`, `predict`, `evaluate_validation`, `compute_residual_prediction_interval`, `get_model_metadata`, `save_model`, `load_model`.
  - Enforces uniform behavior across statistical, machine learning, and deep learning architectures without engine-specific branching.
- **Prophet Forecasting Adapter (`ProphetForecaster`)**:
  - Implemented in `backend/app/forecasting/prophet_model.py` fitting non-linear trends with yearly, weekly, and daily seasonalities.
  - Native Bayesian uncertainty intervals via Stan backend; retail demand non-negative clamping; JSON serialization.
- **LightGBM Forecasting Adapter (`LightGBMForecaster`)**:
  - Implemented in `backend/app/forecasting/lightgbm_model.py` incorporating calendar, lags (`lag_1`, `lag_7`, `lag_14`, `lag_28`), and rolling statistics.
  - **Zero-Leakage Multi-Step Recursive Forecasting**: Dynamically computes future lag and rolling features from previous model predictions without peeking into ground truth future targets.
  - Empirical validation-residual prediction intervals; binary checkpoint serialization via `joblib`.
- **PyTorch LSTM Forecasting Adapter (`LSTMForecaster`)**:
  - Implemented in `backend/app/forecasting/lstm_model.py` using `torch.nn` (`LSTM` -> `Dropout` -> `Linear`).
  - **Strict Train-Only Scaler Isolation**: Standard normalizer mean and std parameters are fitted solely on historical training observations.
  - Deterministic random seed initialization; validation-based early stopping with configurable patience; PyTorch checkpoint serialization (`.pt`).
- **Generic Forecasting Agent (`GenericForecastingAgent`)**:
  - Universal coordinator in `backend/app/forecasting/agent.py`:
    - Filters time series by entity and product identifier.
    - Minimum history validation (rejects series with < 15 observations with `INSUFFICIENT_HISTORY` status).
    - Splits historical series chronologically into Train (fitting) and Validation (ranking).
    - Competes candidate models in parallel with **Model Failure Isolation** (one candidate crash does not halt the pipeline).
    - Ranks candidates dynamically by chosen validation metric (**MAE**, **RMSE**, **MAPE**).
    - Refits winning model on combined (Train + Validation) partition prior to future projection.
    - Projects multi-step future horizon points with transparent uncertainty bounds.
    - Persists trained winner artifact to `models/saved/<model_name>/`.
- **API v1 Forecasting Endpoints (`backend/app/api/v1/endpoints/forecasting.py`)**:
  - `POST /api/v1/forecast/run`: Full multi-model competition, validation ranking, and future point forecast with prediction bounds.
  - `POST /api/v1/forecast/train`: Endpoint for candidate model training and checkpoint serialization.
  - `GET /api/v1/forecast/models`: Discovers supported models, feature requirements, and minimum history rules.
  - `GET /api/v1/forecast/config`: Exposes default system hyperparameters and selection criteria.
  - `GET /api/v1/forecast/health`: Checks operational status of Prophet, LightGBM, PyTorch (with version and CUDA availability), and local storage.
  - `GET /api/v1/forecast/sample`: Fast one-click demonstration forecast on benchmark dataset.
- **Minimal Development Verification UI (`ForecastingPanel.tsx`)**:
  - Integrated into Next.js dashboard shell with entity/product selectors, horizon dropdown, candidate model toggles, validation metric picker, and execution trigger.
  - Renders candidate validation ranking table, interactive SVG time-series forecast chart with prediction interval polygons, and detailed forecast point table.
- **Comprehensive Automated Test Suites**:
  - `tests/unit/test_forecast_models.py`: 4 tests covering BaseForecastModel contract, Prophet, LightGBM, and PyTorch LSTM lifecycles, prediction intervals, metadata, and persistence.
  - `tests/unit/test_forecasting_agent.py`: 5 tests covering multi-model competition, dynamic ranking (MAE/RMSE/MAPE), insufficient history handling, failure isolation, and deterministic reproducibility.
  - `tests/unit/test_forecasting_leakage.py`: 3 tests verifying train-only scaler isolation, holdout test partition protection, and recursive forecasting safety.
  - `tests/unit/test_forecasting_api.py`: 6 tests verifying all REST endpoints, health checks, model catalogs, sample execution, and Pydantic request validation.
  - Expanded total automated test suite from 61 to **79 passing tests** (100% pass rate in 10.74s).

---

## [0.4.0-alpha] - Phase 3: Data Processing Agent (2026-09-13)

### Added
- **Generic Business Data Processor (`GenericBusinessDataProcessor`)**:
  - Implemented universal preprocessing coordinator in `backend/app/data_processing/processing/processor.py` operating directly on canonical `BusinessTimeSeriesRecord` records.
  - End-to-end 11-step pipeline: cleaning -> duplicate resolution -> invalid value rectification -> missing value imputation -> chronological sorting -> gap integrity -> outlier profiling -> leakage-safe feature engineering -> quality score re-evaluation -> temporal splitting -> machine-readable audit report.
- **Data Processing Submodules (`backend/app/data_processing/processing/`)**:
  - `cleaning.py` (`DataCleaner`): Type normalization, string whitespace trimming, datetime standardization with ISO parsing, and numeric coercion.
  - `duplicates.py` (`DuplicateHandler`): Dynamic composite business key (`date` + `entity_id` + `product_id`), exact duplicate detection, conflicting record detection, and deterministic resolution policy (`KEEP_FIRST`, `KEEP_LAST`, `FLAG`).
  - `invalid_values.py` (`InvalidValueHandler`): Rectifies non-positive prices, negative inventory, negative target demand, and clamps promotion/holiday indicators to binary {0, 1}.
  - `missing_values.py` (`MissingValueHandler`): Imputes missing numeric values via entity-grouped medians or forward-fill; handles categoricals via mode or explicit "Unknown". Strict target safety: missing target values are never fabricated from future observations (dropped by default).
  - `outliers.py` (`OutlierDetector`): Group-aware IQR and Z-Score outlier detection. **Default behavior is DETECT & FLAG** (`target_outlier = 1`) to safeguard genuine promotional and holiday demand signals from deletion.
  - `time_series.py` (`TimeSeriesAnalyzer`): Enforces strict chronological sorting, infers observation frequency ('D', 'W'), and computes `TimeSeriesIntegrityReport` (missing periods, largest gap, continuity percentage).
  - `feature_engineering.py` (`FeatureEngineer`): Generates calendar features (`year`, `month`, `quarter`, `day_of_week`, `is_weekend`, etc.), strictly historical lag features (`lag_1`, `lag_7`, `lag_14`, `lag_28`), and rolling window features (`rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28`, `rolling_std_7`, `rolling_std_28`) using `shift(1).rolling(w)` to eliminate lookahead leakage.
  - `splitting.py` (`TimeSeriesSplitter`): Pure data utility for chronological walk-forward splitting (70% train, 15% val, 15% test) without random shuffling or premature model training.
  - `audit.py` (`AuditTrailTracker`): Thread-safe accumulator of `ProcessingAuditEntry` records logging every transformation, row count affected, before/after states, and justifications.
- **Pydantic Processing Schemas (`backend/app/schemas/processing.py`)**:
  - Defined `DataProcessingConfig`, `ProcessingAuditEntry`, `TimeSeriesIntegrityReport`, `OutlierSummary`, `TemporalSplitMetadata`, and `DataProcessingResult`.
- **API v1 Processing Endpoints (`backend/app/api/v1/endpoints/processing.py`)**:
  - `POST /api/v1/process/file`: Ingest & process uploaded CSV.
  - `POST /api/v1/process/local`: Process local server-side CSV.
  - `GET /api/v1/process/config`: Return default `DataProcessingConfig`.
  - `GET /api/v1/process/sample-summary`: Execute processing pipeline on committed retail sample.
- **Dataset Processing Script & Outputs**:
  - Created `scripts/process_retail_dataset.py` processing 50,000 synthetic rows in 0.91s.
  - Generated full processed dataset: `data/processed/synthetic/retail_processed.csv` (12.54 MB, Git ignored).
  - Generated committed sample dataset: `data/sample/retail_processed_sample.csv` (18.79 KB, Git tracked).
- **Automated Unit Tests**:
  - Added `tests/unit/test_data_processor.py` (11 tests).
  - Added `tests/unit/test_feature_engineering.py` (4 tests).
  - Added `tests/unit/test_temporal_splitting.py` (2 tests).
  - Added `tests/unit/test_processing_api.py` (5 tests).
  - Expanded total automated test suite from 39 to **61 passing tests** (100% pass rate).

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
