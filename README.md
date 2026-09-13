# GlassBox-BI

> **Explainable AI (XAI) & Decision Intelligence for Business Forecasting**

[![Current Phase](https://img.shields.io/badge/Phase-0%20%7C%20Foundation-blue.svg)](./PROJECT_STATE.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 📖 Overview

**GlassBox-BI** transforms black-box machine learning predictions into transparent, auditable, and actionable business intelligence.

In real-world business forecasting, knowing a projected number is only half the battle. Decision-makers need to understand:
- **Why** the forecast moved in a specific direction.
- **Which internal and external drivers** contributed most heavily to the outcome.
- **What actions and simulated scenarios** yield the best strategic path forward.

GlassBox-BI combines robust time-series forecasting, glass-box explainability techniques (SHAP, decomposition), prescriptive decision intelligence, and collaborative AI agents within a unified, interactive platform.

---

## 🏛️ System Architecture

The project is structured into clear, decoupled layers:

```
GlassBox-BI/
├── backend/                        # Backend Application Layer
│   ├── app/
│   │   ├── api/                    # REST API endpoints & routers
│   │   ├── core/                   # Configuration, settings & utilities
│   │   ├── data_processing/        # Data ingestion, cleaning & validation
│   │   ├── forecasting/            # Statistical & ML forecasting models
│   │   ├── explainability/         # XAI engine (SHAP, feature attributions)
│   │   ├── decision_intelligence/  # Scenario simulation & recommendations
│   │   ├── agents/                 # Specialized autonomous agents
│   │   ├── orchestration/          # Multi-agent workflow orchestrator
│   │   └── main.py                 # FastAPI application entrypoint
│   └── requirements.txt            # Backend dependencies
├── frontend/                       # Frontend Dashboard (Interactive BI UI)
├── data/                           # Local data store
│   ├── raw/                        # Untouched raw business datasets
│   └── processed/                  # Validated, preprocessed datasets
├── models/                         # Serialized models and checkpoints
│   ├── checkpoints/
│   └── saved/
├── tests/                          # Test Suite
│   ├── unit/                       # Unit tests for domain modules
│   └── integration/                # End-to-end and API integration tests
├── docs/                           # Architecture and development docs
│   ├── architecture.md             # In-depth architectural blueprint
│   └── development_phases.md       # Phased implementation roadmap
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Git ignore rules
├── CHANGELOG.md                    # Project change history
├── PROJECT_STATE.md                # Living status and governance tracker
└── README.md                       # Project documentation entry point
```

For detailed architectural specifications and dataflow diagrams, see [`docs/architecture.md`](docs/architecture.md).

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/ManthanBalla/GlassBox-BI.git
cd GlassBox-BI

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

# Install Phase 0 dependencies
pip install -r backend/requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Adjust parameters (such as `BACKEND_PORT` or API keys) as needed.

### 4. Run Verification Tests
```bash
pytest tests/
```

### 5. Start the Backend Development Server
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
API documentation will be available at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

---

## 🗺️ Phased Development Roadmap

GlassBox-BI follows a disciplined, phase-by-phase development lifecycle:

| Phase | Focus | Status |
|---|---|---|
| **Phase 0** | **Project Foundation & Governance** | ✅ Completed |
| **Phase 1** | **Application Skeleton** | ⏳ Next |
| **Phase 2** | **Dataset Ingestion** | ⏳ Pending |
| **Phase 3** | **Data Processing Agent** | ⏳ Pending |
| **Phase 4** | **Forecasting Agent** | ⏳ Pending |
| **Phase 5** | **Forecast Evaluation** | ⏳ Pending |
| **Phase 6** | **Explainability Agent** | ⏳ Pending |
| **Phase 7** | **Decision Intelligence Agent** | ⏳ Pending |
| **Phase 8** | **Multi-Agent Orchestration** | ⏳ Pending |
| **Phase 9** | **Feedback & Self-Correction Loop** | ⏳ Pending |
| **Phase 10** | **Dashboard** | ⏳ Pending |
| **Phase 11** | **MLflow, Testing & Deployment** | ⏳ Pending |
| **Phase 12** | **Final Integration & Validation** | ⏳ Pending |

Detailed milestone specifications are documented in [`docs/development_phases.md`](docs/development_phases.md).

---

## 📋 Governance & Tracking

All project decisions, state transitions, and releases are tracked in:
- [`PROJECT_STATE.md`](PROJECT_STATE.md): Current phase, architectural decisions, testing status, and active instructions.
- [`CHANGELOG.md`](CHANGELOG.md): Detailed record of changes per release/phase.

---

## 👥 Authors & Academic Context

Developed as a Collaborative Major Project in Artificial Intelligence & Data Science.
- **Repository**: [ManthanBalla/GlassBox-BI](https://github.com/ManthanBalla/GlassBox-BI)
