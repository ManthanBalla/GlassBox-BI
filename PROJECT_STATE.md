# GlassBox-BI — Project State & Governance Tracker

This document serves as the single source of truth for current project progress, architectural decisions, running instructions, environment configuration, and test status.

---

## 1. Current Phase

**Phase 8 — Multi-Agent Orchestration**
- **Status**: Completed
- **Phase Date**: September 2026
- **Version**: `0.9.0-alpha`
- **Next Phase**: Phase 9 — Feedback & Self-Correction Loop

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

### Phase 6 — Explainability Agent
- [x] **Explainability Architecture & Contracts (`backend/app/explainability/`)**:
  - `base.py`: Model-agnostic `BaseExplainer` abstract interface with local/global contracts and contribution ranking.
  - `schemas.py`: Pydantic models for `LocalExplanationRequest`, `GlobalExplanationRequest`, `ExplanationResult`, `FeatureContribution`, `GlobalFeatureImportance`, `ExplanationFidelity`, `ExplanationAuditTrail`.
  - `feature_adapter.py`: Dynamic feature reconstruction and zero-leakage training background sampling.
  - `validation.py`: Model fitting assertions, background sample size checks, and `MODEL_COMPATIBILITY` matrix.
  - `agent.py`: `ExplainabilityAgent` coordinator dispatching model-specific strategies, verifying model immutability, and assembling audit trails.
- [x] **Model-Specific Explainability Implementations**:
  - **LightGBM**: Native `shap.TreeExplainer` providing exact additive Shapley attributions across tabular calendar, lag, rolling, and business regressors. `LIMEExplainer` generating local linear surrogates with reproducible random seeds.
  - **PyTorch LSTM**: Model-agnostic sequence attribution explaining the univariate lookback target sequence (`lag_1` to `lag_{lookback}`). Zero fabrication of tabular features.
  - **Prophet**: `ProphetComponentExplainer` providing exact additive decomposition into `trend`, `weekly_seasonality`, `yearly_seasonality`, and `holiday_effects`. Formally reported as `component_based`.
- [x] **Explanation Fidelity & Quality Metrics**:
  - Additive reconstruction error: $|\hat{y} - (\text{base\_value} + \sum \phi_i)|$.
  - Normalized fidelity score: $\max(0, 1 - \text{error} / (|\hat{y}| + 10^{-6}))$.
  - LIME local surrogate $R^2$ goodness-of-fit.
- [x] **Model Immutability & Zero Data Leakage**:
  - Pre- and post-explanation model state snapshots verify model weights, trees, historical buffers, and scalers are 100% immutable.
  - Background reference distributions are sampled strictly from historical training partitions (`train_df`).
- [x] **REST API Endpoints (`backend/app/api/v1/endpoints/explainability.py`)**:
  - `GET /api/v1/explainability/health`: Subsystem health and dependency verification.
  - `GET /api/v1/explainability/methods`: Explicit model-to-method compatibility matrix.
  - `GET /api/v1/explainability/config`: Default configuration and fidelity thresholds.
  - `POST /api/v1/explainability/local`: Instance-level local feature attribution.
  - `POST /api/v1/explainability/global`: Dataset-level global feature importance ranking.
  - `GET /api/v1/explainability/sample`: Fast pre-computed sample explanation.
- [x] **Minimal Development Frontend Component (`ExplainabilityPanel.tsx`)**:
  - Model and method selectors, Local vs Global mode toggle, positive/negative impact breakdown with colored bars, ranked feature table, fidelity metrics card, and audit drawer.
- [x] **Automated Testing Suite**:
  - **134 tests passing** (31 new Phase 6 unit tests across base contracts, SHAP, LIME, Prophet, model immutability regression, and APIs). Zero failures, zero regressions.

### Phase 7 — Decision Intelligence Agent
- [x] **Deterministic Prescriptive Architecture (`backend/app/decision_intelligence/`)**:
  - `base.py`: Model-independent `BaseDecisionEngine` defining the prescriptive decision contract.
  - `schemas.py`: Pydantic schemas for `BusinessContext`, `DecisionPriority`, `BusinessActionCategory`, `TradeOff`, `RecommendationItem`, `DecisionResult`, `ScenarioRequest`, `ScenarioResult`, `DecisionAuditRecord`.
  - `rules.py`: Deterministic suite of 8 retail business rules (`RetailDecisionRuleEngine`):
    - `RULE_1_HIGH_DEMAND_LOW_INVENTORY`: High priority replenishment scale-up.
    - `RULE_2_HIGH_STOCKOUT_CRITICAL`: Critical emergency replenishment when days-of-supply < lead time.
    - `RULE_3_LOW_DEMAND_HIGH_INVENTORY`: Order curtailment and clearance planning when inventory is excess.
    - `RULE_4_DEMAND_INCREASE_PROMOTION`: Promotion sustaining and inventory burn rate monitoring.
    - `RULE_5_PRICE_DRIVEN_DEMAND_CHANGE`: Price review recommendation with mandatory human sign-off.
    - `RULE_6_HIGH_FORECAST_UNCERTAINTY`: Conservative buffering and confidence downgrade when intervals are wide.
    - `RULE_7_LOW_EXPLANATION_FIDELITY`: Confidence downgrade and human review alert when XAI fidelity is low.
    - `RULE_MISSING_INVENTORY_PLANNING`: Strict zero-fabrication safety rule directing review when context is incomplete.
  - `scoring.py`: Methodological calibration separating Forecast Uncertainty (relative interval spread) from Decision Confidence (calibrated reliability metric), plus rule-based Recommendation Score (0-100).
  - `validation.py`: Strict contract validation enforcing non-empty rationales, traceable evidence lists, valid ranges, and mandatory human review flags.
  - `agent.py`: `DecisionIntelligenceAgent` orchestrating decision evaluation, scenario simulations, and audit logging.
- [x] **Lightweight Deterministic What-If Scenario Analysis**:
  - `run_scenario()` evaluates operational shocks (inventory shifts, promo toggles, price elasticity $\epsilon = -1.5$) without retraining forecasting models.
- [x] **REST API Endpoints (`backend/app/api/v1/endpoints/decisions.py`)**:
  - `GET /api/v1/decisions/health`: Subsystem health, engine parameters, and supported categories.
  - `GET /api/v1/decisions/rules`: Rule catalog with trigger criteria and default priorities.
  - `GET /api/v1/decisions/config`: Default thresholds, coverage multipliers, and holding cost rates.
  - `GET /api/v1/decisions/sample`: Instantaneous sample decision for fast UI evaluation.
  - `POST /api/v1/decisions/run`: Full decision evaluation from forecast, XAI, and context.
  - `POST /api/v1/decisions/scenario`: Interactive what-if scenario simulation.
- [x] **Minimal Development Frontend Component (`DecisionIntelligencePanel.tsx`)**:
  - Operational context inputs, action presets, primary recommendation banner with priority badges, confidence/score meters, traceable evidence list, explicit trade-offs card, and interactive what-if scenario simulator.
- [x] **Automated Testing Suite**:
  - **158 tests passing** (24 new Phase 7 unit tests covering schemas, all 8 retail rules, scoring, agent, scenarios, and REST APIs). Zero failures, zero regressions.
- [x] **Architectural Principles & Safety**:
  - "The Phase 7 Decision Intelligence Agent is a deterministic recommendation engine. It does not autonomously execute business actions."
  - Benchmarked for retail demand forecasting; architecturally extensible to SME cash-flow management. Zero LLM dependency.

### Phase 8 — Multi-Agent Orchestration
- [x] **Orchestration Module (`backend/app/orchestration/`)**:
  - `base.py`: Abstract `BaseOrchestrator` defining deterministic workflow execution and state inspection interface.
  - `schemas.py`: Pydantic schemas for `PipelineStage`, `WorkflowStatus`, `StageStatus`, `StageExecutionResult`, `OrchestrationRequest`, `OrchestrationAuditRecord`, `OrchestrationResult`, and `OrchestrationState`.
  - `registry.py`: `AgentRegistry` providing explicit mapping of `PipelineStage` to agent implementations (`GenericBusinessDataProcessor`, `GenericForecastingAgent`, `FormalForecastEvaluator`, `ExplainabilityAgent`, `DecisionIntelligenceAgent`).
  - `graph.py`: `WorkflowGraph` canonical pipeline sequence (`DATA_PROCESSING` → `FORECASTING` → `EVALUATION` → `EXPLAINABILITY` → `DECISION_INTELLIGENCE`), dependency definitions, and blocking failure policies.
  - `validation.py`: `WorkflowValidator` fail-early validation for dataset paths, horizons, model candidates, and business parameters.
  - `audit.py`: `WorkflowAuditTracker` recording workflow-level audit trail, timestamps, and stage performance.
  - `executor.py`: `SequentialWorkflowExecutor` executing stages strictly in order, enforcing error isolation, preventing automatic retries, and assembling `OrchestrationResult`.
  - `agent.py`: `MultiAgentOrchestrator` high-level coordinator with workflow state caching.
- [x] **Sequential Data Flow & Stage Contracts**:
  - Raw Input $\longrightarrow$ Data Processing Result $\longrightarrow$ Processed Dataset $\longrightarrow$ Forecast Request $\longrightarrow$ Forecast Result $\longrightarrow$ Evaluation Result $\longrightarrow$ Explanation Request $\longrightarrow$ Explanation Result $\longrightarrow$ Decision Request $\longrightarrow$ Decision Result $\longrightarrow$ Unified Orchestration Result.
- [x] **Error Isolation & Blocking Failure Policy**:
  - `DATA_PROCESSING` or `FORECASTING` failure halts downstream stages immediately (`WorkflowStatus.FAILED`).
  - `EVALUATION` failure preserves forecast and allows downstream execution without fabricating metrics.
  - `EXPLAINABILITY` failure preserves forecast and allows decisions with missing explanation warnings, confidence penalties, and mandatory human review.
  - `DECISION_INTELLIGENCE` failure preserves data, forecast, evaluation, and explanation results (`WorkflowStatus.PARTIAL`).
- [x] **Strict Non-Functional & Boundary Invariants**:
  - **Zero Feedback / Self-Correction**: If a stage fails or emits warnings, Phase 8 records the outcome and halts/continues per policy. It strictly does NOT trigger automatic retraining, reforecasting, or parameter adjustments (Phase 9 boundary).
  - **Zero LLM Dependency**: Pure deterministic Python orchestration.
  - **Deterministic Repeatability**: Identical inputs yield bitwise identical execution sequences, predictions, and recommendations.
- [x] **REST API Endpoints (`backend/app/api/v1/endpoints/orchestration.py`)**:
  - `GET /api/v1/orchestration/health`: Subsystem health and registered stage catalog.
  - `GET /api/v1/orchestration/stages`: Canonical pipeline stages, dependencies, and blocking failure policies.
  - `GET /api/v1/orchestration/sample`: Sample pre-configured workflow execution with synthetic retail dataset.
  - `POST /api/v1/orchestration/run`: End-to-end multi-agent pipeline execution.
  - `GET /api/v1/orchestration/{workflow_id}`: Workflow state retrieval and stage audit inspection.
- [x] **Minimal Development Frontend Component (`OrchestrationPanel.tsx`)**:
  - Interactive pipeline run trigger, real-time stage progress timeline with status icons and execution durations, selected model and forecast overview, explanation summary, decision recommendation cards, and full audit drawer.
- [x] **Automated Testing Suite**:
  - **189 total tests passing** (30 new Phase 8 unit tests across schemas, registry, graph, executor, error isolation, API, and boundary invariants). Zero failures, zero regressions.
- [x] **Architectural Principles & Safety**:
  - "Phase 8 coordinates the existing agents but does not implement feedback-driven self-correction."

---

## 3. Pending Phases

| Phase | Description | Status |
|---|---|---|
| **Phase 8** | Multi-Agent Orchestration | **Completed** |
| **Phase 9** | Feedback & Self-Correction Loop | **Next Recommended Phase** |
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
  - `tests/unit/test_forecasting_*.py`: 18 tests passing (candidate models, hyperparameter tuning, leakage safety)
  - `tests/unit/test_evaluation_*.py`: 24 tests passing (formal evaluation metrics, holdout benchmark engine, evaluation APIs)
  - `tests/unit/test_explainability_*.py`: 31 tests passing (SHAP, LIME, Prophet component decomposition, model immutability)
  - `tests/unit/test_decision_*.py`: 24 tests passing (deterministic retail rules, scoring, scenario simulations, decision APIs)
  - **Total**: **158/158 passing** (100% pass rate across Phase 0 through Phase 7)
- **Frontend Build**: `npm run build` — **Passing** (0 TypeScript errors)

