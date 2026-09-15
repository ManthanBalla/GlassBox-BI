"use client";

import React, { useState, useEffect } from "react";

interface StageSummaryItem {
  status: string;
  duration_ms: number;
  warnings: string[];
  errors: string[];
}

interface OrchestrationResult {
  workflow_id: string;
  status: string;
  stage_summary: Record<string, StageSummaryItem>;
  data_processing?: Record<string, any> | null;
  forecast?: {
    model_name: string;
    horizon: number;
    predictions: Array<{
      date: string;
      predicted_value: number;
      lower_bound?: number | null;
      upper_bound?: number | null;
    }>;
    metrics: Record<string, number>;
  } | null;
  evaluation?: {
    phase4_selected_model: string;
    horizon: number;
    models: Array<{
      model_name: string;
      mae?: number | null;
      rmse?: number | null;
      mape?: number | null;
      status: string;
    }>;
  } | null;
  explanation?: {
    method: string;
    top_positive_features?: string[];
    top_negative_features?: string[];
    fidelity?: {
      fidelity_score: number;
      explanation_status: string;
      reconstruction_error: number;
    } | null;
  } | null;
  decisions?: {
    primary_recommendation?: {
      action: string;
      priority: string;
      confidence: number;
      risk: string;
      rationale: string;
    } | null;
    recommendations: Array<{
      action: string;
      priority: string;
      confidence: number;
    }>;
    audit_trail?: {
      decision_confidence: number;
    } | null;
    requires_human_review: boolean;
  } | null;
  warnings: string[];
  errors: string[];
  audit?: {
    workflow_id: string;
    total_duration_ms: number;
    stage_sequence: string[];
    selected_model?: string | null;
    final_status: string;
  } | null;
  execution_summary?: Record<string, any>;
}

const STAGES = [
  { id: "DATA_PROCESSING", label: "Data Processing", phase: "Phase 3", icon: "🧹" },
  { id: "FORECASTING", label: "Forecasting", phase: "Phase 4", icon: "📈" },
  { id: "EVALUATION", label: "Evaluation", phase: "Phase 5", icon: "🎯" },
  { id: "EXPLAINABILITY", label: "Explainability", phase: "Phase 6", icon: "🔍" },
  { id: "DECISION_INTELLIGENCE", label: "Decision Intelligence", phase: "Phase 7", icon: "💡" },
];

export default function OrchestrationPanel() {
  const [horizon, setHorizon] = useState<number>(14);
  const [candidateModels, setCandidateModels] = useState<string[]>(["lightgbm", "prophet", "lstm"]);
  const [runEvaluation, setRunEvaluation] = useState<boolean>(true);
  const [runExplainability, setRunExplainability] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<string>("summary");

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<OrchestrationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSample();
  }, []);

  const fetchSample = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch("http://localhost:8000/api/v1/orchestration/sample");
      if (!res.ok) throw new Error(`Backend returned HTTP ${res.status}`);
      const data: OrchestrationResult = await res.json();
      setResult(data);
    } catch (err: any) {
      console.warn("Could not fetch sample orchestration:", err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunPipeline = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const payload = {
        entity_id: "STORE_001",
        product_id: "PROD_001",
        horizon,
        selection_metric: "MAE",
        confidence_level: 0.80,
        candidate_models: candidateModels,
        run_evaluation: runEvaluation,
        evaluation_horizon: horizon,
        run_explainability: runExplainability,
        explanation_method: "auto",
        business_context: {
          current_inventory: 85.0,
          reorder_point: 120.0,
          lead_time_days: 7,
          current_price: 24.99,
          promotion_active: true,
        },
      };

      const res = await fetch("http://localhost:8000/api/v1/orchestration/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned error ${res.status}`);
      }

      const data: OrchestrationResult = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to execute orchestration pipeline");
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusBadge = (status?: string) => {
    switch (status) {
      case "COMPLETED":
      case "SUCCESS":
        return "bg-emerald-950/80 text-emerald-300 border-emerald-700/60";
      case "RUNNING":
        return "bg-cyan-950/80 text-cyan-300 border-cyan-700/60 animate-pulse";
      case "PARTIAL":
        return "bg-amber-950/80 text-amber-300 border-amber-700/60";
      case "FAILED":
        return "bg-rose-950/80 text-rose-300 border-rose-700/60";
      case "SKIPPED":
        return "bg-slate-900 text-slate-400 border-slate-700";
      default:
        return "bg-slate-900 text-slate-500 border-slate-800";
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-6 relative overflow-hidden">
      <div className="absolute -right-16 -top-16 w-64 h-64 bg-cyan-600/10 rounded-full blur-3xl pointer-events-none"></div>

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-widest text-cyan-400 bg-cyan-950/80 px-2.5 py-0.5 rounded border border-cyan-800/60">
              Phase 8 Active
            </span>
            <span className="text-xs text-slate-500 font-mono">Sequential Multi-Agent Engine</span>
          </div>
          <h3 className="text-lg font-bold text-white mt-1 tracking-tight flex items-center gap-2">
            <span>⚙️ Multi-Agent Orchestration Engine</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Coordinates Phase 3 through 7 agents in an auditable, deterministic workflow without automatic retries.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchSample}
            disabled={isLoading}
            className="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-900/80 hover:bg-slate-800 text-xs font-medium text-slate-300 transition-colors disabled:opacity-50"
          >
            Load Sample
          </button>
          <button
            onClick={handleRunPipeline}
            disabled={isLoading}
            className="px-4 py-1.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-xs font-semibold text-slate-950 transition-all shadow-md shadow-cyan-950/50 disabled:opacity-50 flex items-center gap-1.5"
          >
            {isLoading ? (
              <>
                <div className="w-3 h-3 border-2 border-slate-950 border-t-transparent rounded-full animate-spin"></div>
                <span>Executing Pipeline...</span>
              </>
            ) : (
              <>
                <span>▶ Run Multi-Agent Pipeline</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Configuration Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-950/50 p-3 rounded-xl border border-slate-800/60 text-xs">
        <div>
          <label className="text-slate-400 block font-medium mb-1">Forecast Horizon</label>
          <select
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value={7}>7 Days (Short-term)</option>
            <option value={14}>14 Days (Standard)</option>
            <option value={28}>28 Days (Monthly)</option>
          </select>
        </div>

        <div>
          <label className="text-slate-400 block font-medium mb-1">Model Selection</label>
          <div className="flex gap-2 pt-1">
            {["lightgbm", "prophet", "lstm"].map((m) => (
              <label key={m} className="flex items-center gap-1 cursor-pointer text-slate-300">
                <input
                  type="checkbox"
                  checked={candidateModels.includes(m)}
                  onChange={(e) => {
                    if (e.target.checked) setCandidateModels([...candidateModels, m]);
                    else if (candidateModels.length > 1) setCandidateModels(candidateModels.filter((c) => c !== m));
                  }}
                  className="rounded border-slate-700 text-cyan-500 focus:ring-0"
                />
                <span className="uppercase text-[10px] font-mono">{m}</span>
              </label>
            ))}
          </div>
        </div>

        <div>
          <label className="text-slate-400 block font-medium mb-1">Phase 5 Evaluation</label>
          <label className="flex items-center gap-1.5 pt-1 text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={runEvaluation}
              onChange={(e) => setRunEvaluation(e.target.checked)}
              className="rounded border-slate-700 text-cyan-500 focus:ring-0"
            />
            <span>Test-Set Benchmark</span>
          </label>
        </div>

        <div>
          <label className="text-slate-400 block font-medium mb-1">Phase 6 Explainability</label>
          <label className="flex items-center gap-1.5 pt-1 text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={runExplainability}
              onChange={(e) => setRunExplainability(e.target.checked)}
              className="rounded border-slate-700 text-cyan-500 focus:ring-0"
            />
            <span>SHAP / LIME Attribution</span>
          </label>
        </div>
      </div>

      {/* Error alert */}
      {error && (
        <div className="bg-rose-950/50 border border-rose-800 text-rose-300 text-xs p-3 rounded-xl flex items-start gap-2">
          <span>⚠️</span>
          <div>
            <strong>Orchestration Pipeline Error:</strong> {error}
          </div>
        </div>
      )}

      {/* Sequential Pipeline Stage Timeline */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Sequential Execution Pipeline</span>
          {result && (
            <span className="text-xs font-mono text-slate-500">
              Workflow ID: <span className="text-cyan-400">{result.workflow_id}</span>
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-5 gap-2">
          {STAGES.map((st, index) => {
            const stageResult = result?.stage_summary?.[st.id];
            const status = stageResult?.status || "PENDING";
            const duration = stageResult?.duration_ms || 0;

            return (
              <div
                key={st.id}
                className="relative bg-slate-900/90 border border-slate-800 rounded-xl p-3 flex flex-col justify-between space-y-2 hover:border-slate-700 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono text-slate-500">0{index + 1}</span>
                  <span
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded border uppercase ${getStatusBadge(
                      status
                    )}`}
                  >
                    {status}
                  </span>
                </div>

                <div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-sm">{st.icon}</span>
                    <span className="text-xs font-bold text-white tracking-tight">{st.label}</span>
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono block mt-0.5">{st.phase}</span>
                </div>

                <div className="text-[10px] text-slate-500 border-t border-slate-800/60 pt-1.5 flex justify-between items-center">
                  <span>Duration:</span>
                  <span className="font-mono text-slate-300">{duration > 0 ? `${duration.toFixed(1)}ms` : "—"}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Result Inspection Section */}
      {result && (
        <div className="space-y-4 pt-2">
          {/* Top Key Result Metric Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/80">
              <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">Workflow Status</span>
              <div className="flex items-center gap-2 mt-1">
                <span className={`text-xs font-bold px-2 py-0.5 rounded border ${getStatusBadge(result.status)}`}>
                  {result.status}
                </span>
                <span className="text-[11px] text-slate-400 font-mono">
                  {result.audit?.total_duration_ms ? `${result.audit.total_duration_ms}ms` : ""}
                </span>
              </div>
            </div>

            <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/80">
              <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">Winning Forecaster</span>
              <span className="text-sm font-bold text-cyan-400 mt-1 block uppercase font-mono">
                {result.forecast?.model_name || "N/A"}
              </span>
              <span className="text-[10px] text-slate-500">
                {result.forecast?.horizon ? `${result.forecast.horizon}-day projection` : ""}
              </span>
            </div>

            <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/80">
              <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">Formal Test MAE</span>
              <span className="text-sm font-bold text-emerald-400 mt-1 block font-mono">
                {(() => {
                  const winner = result.evaluation?.phase4_selected_model;
                  const modelEval = result.evaluation?.models?.find((m) => m.model_name === winner);
                  return modelEval?.mae ? modelEval.mae.toFixed(4) : "Evaluated";
                })()}
              </span>
              <span className="text-[10px] text-slate-500">Protected test set</span>
            </div>

            <div className="bg-slate-950/70 p-3 rounded-xl border border-slate-800/80">
              <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">Decision Confidence</span>
              <span className="text-sm font-bold text-purple-400 mt-1 block font-mono">
                {result.decisions?.audit_trail?.decision_confidence
                  ? `${(result.decisions.audit_trail.decision_confidence * 100).toFixed(1)}%`
                  : "N/A"}
              </span>
              <span className="text-[10px] text-slate-500">
                {result.decisions?.requires_human_review ? "⚠️ Human Review Req." : "Auto-Verified"}
              </span>
            </div>
          </div>

          {/* Primary Recommendation Banner */}
          {result.decisions?.primary_recommendation && (
            <div className="p-4 rounded-xl border border-cyan-800/60 bg-gradient-to-r from-cyan-950/40 via-slate-900 to-slate-950 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400 bg-cyan-950 px-2 py-0.5 rounded border border-cyan-800/80">
                  Primary Business Recommendation ({result.decisions.primary_recommendation.priority})
                </span>
                <h4 className="text-sm font-bold text-white mt-1.5 tracking-tight">
                  {result.decisions.primary_recommendation.action}
                </h4>
                <p className="text-xs text-slate-300 mt-1 max-w-3xl leading-relaxed">
                  {result.decisions.primary_recommendation.rationale}
                </p>
              </div>

              <div className="text-right whitespace-nowrap">
                <span className="text-[10px] text-slate-400 block">Confidence Score</span>
                <span className="text-lg font-black text-cyan-300 font-mono">
                  {(result.decisions.primary_recommendation.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          )}

          {/* Warnings List */}
          {result.warnings && result.warnings.length > 0 && (
            <div className="bg-amber-950/30 border border-amber-800/50 rounded-xl p-3 text-xs text-amber-300 space-y-1">
              <span className="font-bold flex items-center gap-1">
                <span>⚠️</span> Operational Audit Warnings ({result.warnings.length})
              </span>
              <ul className="list-disc list-inside space-y-0.5 text-amber-200/90 text-[11px]">
                {result.warnings.slice(0, 3).map((w, idx) => (
                  <li key={idx}>{w}</li>
                ))}
                {result.warnings.length > 3 && <li>...and {result.warnings.length - 3} more warnings logged in audit.</li>}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
