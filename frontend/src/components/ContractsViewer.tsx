"use client";

import React, { useState } from "react";

interface ContractSpec {
  name: string;
  phase: string;
  description: string;
  fields: { name: string; type: string; desc: string }[];
}

const contractSpecs: ContractSpec[] = [
  {
    name: "DatasetMetadata",
    phase: "Phase 2 — Dataset Ingestion",
    description: "Catalog contract representing uploaded tabular time-series datasets.",
    fields: [
      { name: "dataset_id", type: "string", desc: "Unique identifier" },
      { name: "name", type: "string", desc: "Human readable name" },
      { name: "row_count", type: "int", desc: "Number of rows in dataset" },
      { name: "timestamp_column", type: "string", desc: "Primary datetime column" },
      { name: "target_column", type: "string", desc: "Target variable to project" },
    ],
  },
  {
    name: "ForecastRequest",
    phase: "Phase 4 — Forecasting Agent",
    description: "Input payload to initiate time-series predictive modeling.",
    fields: [
      { name: "dataset_id", type: "string", desc: "Target dataset ID" },
      { name: "target_column", type: "string", desc: "Column to forecast" },
      { name: "horizon", type: "int", desc: "Number of forward steps (1-365)" },
      { name: "model_preference", type: "string", desc: "'auto', 'statistical', or 'ml'" },
    ],
  },
  {
    name: "ForecastResult",
    phase: "Phase 4 & 5 — Forecasting & Evaluation",
    description: "Standardized predictive output with error metrics and confidence bounds.",
    fields: [
      { name: "forecast_id", type: "string", desc: "Unique run ID" },
      { name: "model_name", type: "string", desc: "Selected model algorithm" },
      { name: "predictions", type: "list[point]", desc: "Generated time-series forecast" },
      { name: "metrics", type: "dict[float]", desc: "MAE, RMSE, MAPE" },
    ],
  },
  {
    name: "ExplanationResult",
    phase: "Phase 6 — Explainability Agent",
    description: "Glass-box interpretability data including SHAP attributions and decomposition.",
    fields: [
      { name: "explanation_id", type: "string", desc: "Unique explanation ID" },
      { name: "method", type: "string", desc: "'shap', 'decomposition'" },
      { name: "feature_attributions", type: "dict[float]", desc: "Impact scores per driver" },
      { name: "narrative_summary", type: "string", desc: "Plain language interpretation" },
    ],
  },
  {
    name: "RecommendationResult",
    phase: "Phase 7 — Decision Intelligence Agent",
    description: "Prescriptive insights, 'what-if' simulations, and ranked business action items.",
    fields: [
      { name: "recommendation_id", type: "string", desc: "Unique recommendation ID" },
      { name: "scenario_name", type: "string", desc: "Simulation scenario" },
      { name: "action_items", type: "list[string]", desc: "Ranked strategic levers" },
      { name: "risk_assessment", type: "string", desc: "Risk and uncertainty score" },
    ],
  },
];

export default function ContractsViewer() {
  const [activeTab, setActiveTab] = useState(0);
  const activeContract = contractSpecs[activeTab];

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6">
        <div>
          <h3 className="text-lg font-bold text-white tracking-tight">
            Phase 1 Module Contracts (Pydantic v2)
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Inter-module communication contracts established for upcoming phase development.
          </p>
        </div>
        <span className="text-[11px] font-medium text-cyan-400 bg-cyan-950/80 px-2.5 py-1 rounded-lg border border-cyan-800/80 shrink-0 self-start sm:self-auto">
          Schema Contracts Only — Zero Business Logic
        </span>
      </div>

      {/* Contract Tabs */}
      <div className="flex flex-wrap gap-2 pb-4 border-b border-slate-800/80">
        {contractSpecs.map((contract, idx) => (
          <button
            key={contract.name}
            onClick={() => setActiveTab(idx)}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
              activeTab === idx
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 border border-transparent"
            }`}
          >
            {contract.name}
          </button>
        ))}
      </div>

      {/* Contract Details */}
      <div className="mt-5">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-bold text-slate-200 font-mono">
            class {activeContract.name}(BaseModel)
          </span>
          <span className="text-[11px] text-slate-400 font-medium">
            {activeContract.phase}
          </span>
        </div>
        <p className="text-xs text-slate-400 mb-4">{activeContract.description}</p>

        <div className="overflow-x-auto rounded-xl border border-slate-800/80">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/80 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800/80">
              <tr>
                <th className="py-2.5 px-4 font-semibold">Field Name</th>
                <th className="py-2.5 px-4 font-semibold">Type</th>
                <th className="py-2.5 px-4 font-semibold">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
              {activeContract.fields.map((f) => (
                <tr key={f.name} className="hover:bg-slate-850/40 transition-colors">
                  <td className="py-2 px-4 text-cyan-300 font-semibold">{f.name}</td>
                  <td className="py-2 px-4 text-purple-300">{f.type}</td>
                  <td className="py-2 px-4 text-slate-400 font-sans text-xs">{f.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
