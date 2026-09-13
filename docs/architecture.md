# GlassBox-BI — System Architecture Blueprint

## 1. Executive Summary & Vision

**GlassBox-BI** is an Explainable Artificial Intelligence (XAI) and Decision Intelligence platform engineered for transparent business forecasting. 

Traditional BI and machine learning forecasting pipelines often function as "black boxes" — producing numeric predictions without explaining *why* numbers move, *what drivers* cause the variation, or *what actionable decisions* leaders should make. GlassBox-BI bridges this trust deficit by pairing predictive modeling with explainability layers and collaborative agentic workflows, providing understandable, auditable, and actionable business insights.

---

## 2. High-Level Architecture

The platform is structured into modular layers, maintaining strict separation of concerns:

```mermaid
flowchart TD
    subgraph Client["Presentation Layer (Frontend)"]
        UI["Interactive BI Dashboard (KPIs, Forecasts, SHAP Visualizations, Action Plans)"]
    end

    subgraph API["API & Gateway Layer (FastAPI)"]
        Routes["REST Endpoints / WebSocket Streaming"]
        Core["Settings, Auth, & Logging"]
    end

    subgraph Orchestration["Agent & Workflow Orchestration"]
        OrchEngine["Pipeline Orchestrator & State Graph"]
        subgraph Agents["Specialized Agents"]
            DA["Data Processing Agent"]
            FA["Forecasting Agent"]
            EA["Explainability Agent"]
            DIA["Decision Intelligence Agent"]
        end
    end

    subgraph Engines["Computational Engines"]
        DataMod["Data Processing Engine (Cleaning, Validation, Engineering)"]
        ForecastMod["Forecasting Engine (Baselines, ML Time-Series)"]
        ExplainMod["Explainability Engine (SHAP, Feature Attribution, Decomposition)"]
        DecIntelMod["Decision Intelligence Engine (What-If Simulations, Recommendations)"]
    end

    subgraph Storage["Storage & Persistence"]
        DB[(Metadata & Logs DB)]
        DataStore[(Datasets: Raw / Processed)]
        ModelStore[(Model Artifacts / Checkpoints)]
    end

    UI <--> Routes
    Routes <--> OrchEngine
    OrchEngine --> DA & FA & EA & DIA
    DA --> DataMod
    FA --> ForecastMod
    EA --> ExplainMod
    DIA --> DecIntelMod
    DataMod <--> DataStore
    ForecastMod <--> ModelStore
    OrchEngine <--> DB
```

---

## 3. Core Module Boundaries & Responsibilities

### 3.1 Data Processing Module (`backend/app/data_processing`)
- **Ingestion**: Supports CSV, tabular formats, and API data streams.
- **Cleansing & Validation**: Automatic schema verification, missing value imputation, outlier handling, and timestamp alignment.
- **Feature Engineering**: Lag features, rolling statistics, calendar seasonality indicators, and external regressors.

### 3.2 Forecasting Module (`backend/app/forecasting`)
- **Predictive Modeling**: Combines statistical baselines (e.g., ARIMA/ETS) and modern machine learning models (e.g., LightGBM/XGBoost, Prophet).
- **Evaluation**: Computes standardized time-series metrics (MAE, RMSE, MAPE, WAPE).
- **Artifacts**: Serializes and tracks model artifacts and metadata.

### 3.3 Explainability Module (`backend/app/explainability`)
- **Glass-Box Principle**: Ensures every forecast is accompanied by interpretability metadata.
- **Feature Attribution**: Global and local feature attributions using SHAP (SHapley Additive exPlanations) and surrogate models.
- **Time-Series Decomposition**: Separates signals into trend, seasonality, cyclical effects, and residual noise.

### 3.4 Decision Intelligence Module (`backend/app/decision_intelligence`)
- **Scenario Simulation**: "What-if" analysis allowing business users to simulate driver adjustments (e.g., price increase, marketing spend shift).
- **Prescriptive Insights**: Translates forecast gaps and key drivers into human-readable strategic recommendations.
- **Risk Quantification**: Confidence bounds and scenario volatility metrics.

### 3.5 Agent Layer & Multi-Agent Orchestration (`backend/app/agents` & `backend/app/orchestration`)
- **Role-Based Agents**:
  - *Data Processing Agent*: Audits dataset quality and recommends transformations.
  - *Forecasting Agent*: Selects optimal models, tunes parameters, and validates performance.
  - *Explainability Agent*: Formulates narrative explanations around feature attributions.
  - *Decision Intelligence Agent*: Evaluates simulated scenarios and drafts executive decision memos.
- **Orchestration**: Manages the multi-agent graph, state transitions, human-in-the-loop approvals, and self-correction loops.

### 3.6 API Layer (`backend/app/api`)
- FastAPI-powered asynchronous REST API.
- Endpoints for dataset ingestion, forecast triggers, explainability queries, scenario simulations, and agent run status.

### 3.7 Frontend Layer (`frontend/`)
- Responsive business dashboard presenting forecast curves, interactive attribution charts (waterfall, beeswarm), scenario sliders, and AI agent reasoning summaries.

---

## 4. Architectural Principles for Collaborative Student Development

1. **Modularity First**: Modules communicate through explicit Python interfaces and typed Pydantic models. No direct hidden dependencies between engines.
2. **Minimal Initial Complexity**: Avoid over-engineering. Build lightweight, verified components before adding complex agentic graphs or heavy deep learning models.
3. **Reproducibility**: Environment variables are strictly centralized (`.env.example`), and dependencies are pinned in `requirements.txt`.
4. **Transparent Governance**: Every phase is tracked via `PROJECT_STATE.md` and `CHANGELOG.md` with continuous test coverage in `tests/`.
