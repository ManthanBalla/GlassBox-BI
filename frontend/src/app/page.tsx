"use client";

import React from "react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import BackendStatus from "@/components/BackendStatus";
import ArchitectureCard from "@/components/ArchitectureCard";
import ContractsViewer from "@/components/ContractsViewer";
import ForecastingPanel from "@/components/ForecastingPanel";
import EvaluationPanel from "@/components/EvaluationPanel";
import ExplainabilityPanel from "@/components/ExplainabilityPanel";

export default function Home() {
  return (
    <div className="flex min-h-screen bg-[#090d16] text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Sidebar Navigation */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Header />

        <main className="flex-1 p-8 max-w-7xl w-full mx-auto space-y-8">
          {/* Welcome Banner */}
          <div className="glass-panel rounded-2xl p-6 relative overflow-hidden border border-slate-800">
            <div className="absolute -right-10 -bottom-10 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none"></div>
            <div className="relative z-10">
              <span className="text-xs font-bold uppercase tracking-widest text-cyan-400 bg-cyan-950/80 px-2.5 py-1 rounded-md border border-cyan-800">
                Phase 1 Active
              </span>
              <h3 className="text-xl sm:text-2xl font-black text-white mt-3 tracking-tight">
                GlassBox-BI Platform Shell
              </h3>
              <p className="text-slate-300 text-sm mt-1.5 max-w-3xl leading-relaxed">
                Welcome to the operational foundation of GlassBox-BI. This shell establishes the frontend-to-backend 
                communication highway and standardizes the inter-module contracts. Future phases will introduce autonomous 
                data processing, time-series forecasting, glass-box SHAP explainability, and prescriptive decision intelligence.
              </p>
            </div>
          </div>

          {/* Backend Connection & Health Status */}
          <BackendStatus />

          {/* Architecture Module Blueprint Cards */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-bold text-white tracking-tight">
                  Planned Platform Engines
                </h3>
                <p className="text-xs text-slate-400">
                  Architectural modules ready for incremental implementation.
                </p>
              </div>
              <span className="text-xs text-slate-400 font-mono">13-Phase Lifecycle</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <ArchitectureCard
                title="Dataset Ingestion"
                phase="Phase 2"
                icon="📊"
                accent="cyan"
                description="Ingestion, parsing, and automated frequency detection for business time-series datasets."
              />
              <ArchitectureCard
                title="Forecasting Agent"
                phase="Phase 4"
                icon="📈"
                accent="blue"
                description="Autonomous modeling agent combining statistical baselines (ARIMA/ETS) and ML (LightGBM/Prophet)."
              />
              <ArchitectureCard
                title="Explainability (XAI)"
                phase="Phase 6"
                icon="🔍"
                accent="purple"
                description="SHAP attributions and trend-seasonality signal decomposition for total forecasting transparency."
              />
              <ArchitectureCard
                title="Decision Intelligence"
                phase="Phase 7"
                icon="💡"
                accent="emerald"
                description="'What-if' simulation engine generating prescriptive business actions and risk scores."
              />
            </div>
          </div>

          {/* Active Contract Explorer */}
          <ContractsViewer />

          {/* Phase 4 Development Verification: Forecasting Agent */}
          <ForecastingPanel />

          {/* Phase 5 Development Verification: Holdout Test-Set Benchmark */}
          <EvaluationPanel />

          {/* Phase 6 Development Verification: Explainability Agent (SHAP / LIME / Component) */}
          <ExplainabilityPanel />

          {/* Development Notice */}
          <div className="text-center py-6 border-t border-slate-900 text-xs text-slate-400">
            <p>GlassBox-BI · Collaborative Major Project in AI & Data Science · Phase 1 Application Skeleton</p>
          </div>
        </main>
      </div>
    </div>
  );
}
