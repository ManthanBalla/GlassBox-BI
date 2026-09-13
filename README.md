# GlassBox-BI

> **Explainable AI (XAI) & Decision Intelligence for Business Forecasting**

[![Current Phase](https://img.shields.io/badge/Phase-1%20%7C%20App%20Skeleton-blue.svg)](./PROJECT_STATE.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Frontend](https://img.shields.io/badge/Next.js-16%20%7C%20TypeScript-black.svg)](https://nextjs.org/)

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

The project is structured into clean, decoupled layers:

```
GlassBox-BI/
├── backend/                        # FastAPI Backend Application
│   ├── app/
│   │   ├── api/                    # Versioned REST APIs (/api/v1/)
│   │   ├── core/                   # Config, settings, logging & CORS
│   │   ├── schemas/                # Pydantic v2 domain & contract schemas
│   │   ├── services/               # Business logic & domain services
│   │   ├── data_processing/        # Data ingestion & cleansing boundary
│   │   ├── forecasting/            # Forecasting models boundary
│   │   ├── explainability/         # XAI & SHAP attribution boundary
│   │   ├── decision_intelligence/  # Scenario simulation boundary
│   │   ├── agents/                 # Autonomous agents boundary
│   │   ├── orchestration/          # Multi-agent orchestrator boundary
│   │   └── main.py                 # Application entrypoint & health checks
│   └── requirements.txt            # Backend dependencies
├── frontend/                       # Next.js 16 + TypeScript Dashboard
│   ├── src/
│   │   ├── app/                    # Next.js App Router (Layout & Dashboard)
│   │   └── components/             # Sidebar, Header, BackendStatus, Contracts
│   ├── public/                     # Static assets
│   └── package.json                # Frontend dependencies & scripts
├── data/                           # Local data storage
│   ├── raw/                        # Raw uploaded business datasets
│   └── processed/                  # Cleaned, validated datasets
├── models/                         # Serialized models and checkpoints
│   ├── checkpoints/
│   └── saved/
├── tests/                          # Automated test suites
│   ├── unit/                       # Unit tests (API, schemas, foundation)
│   └── integration/                # Integration test suites
├── docs/                           # Documentation
│   ├── architecture.md             # System architecture blueprint
│   └── development_phases.md       # 13-Phase development roadmap
├── .env.example                    # Global environment variables template
├── .gitignore                      # Git ignore rules
├── CHANGELOG.md                    # Project change history
├── PROJECT_STATE.md                # Single source of truth for project state
└── README.md                       # Project documentation entry point
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js v18+ & npm
- Git

### 2. Backend Setup & Run
```bash
# Clone the repository
git clone https://github.com/ManthanBalla/GlassBox-BI.git
cd GlassBox-BI

# Create and activate Python virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# or source .venv/bin/activate  (Linux/macOS)

# Install backend dependencies
pip install -r backend/requirements.txt

# Copy environment configuration
cp .env.example .env

# Run backend test suite
pytest tests/

# Launch FastAPI development server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend endpoints will be available at:
- Root Health Check: `http://127.0.0.1:8000/health`
- API v1 Health Check: `http://127.0.0.1:8000/api/v1/health`
- Contract Discovery: `http://127.0.0.1:8000/api/v1/contracts/specs`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### 3. Frontend Setup & Run
```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Next.js development server
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser to view the interactive GlassBox-BI dashboard shell with live backend health monitoring.

---

## 🗺️ Phased Development Roadmap

GlassBox-BI follows a disciplined 13-phase development strategy:

| Phase | Focus | Status |
|---|---|---|
| **Phase 0** | **Project Foundation & Governance** | ✅ Completed |
| **Phase 1** | **Application Skeleton** | ✅ Completed |
| **Phase 2** | **Dataset Ingestion** | ⏳ Next |
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
