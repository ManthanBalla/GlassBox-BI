"use client";

import React, { useState } from "react";

interface EvaluationPoint {
  date: string;
  actual: number;
  prediction: number;
  lower_bound?: number | null;
  upper_bound?: number | null;
  absolute_error: number;
  percentage_error?: number | null;
}

interface ModelTestEvaluation {
  model_name: string;
  model_type: string;
  status: string;
  mae?: number | null;
  rmse?: number | null;
  mape?: number | null;
  test_start_date?: string | null;
  test_end_date?: string | null;
  horizon: number;
  predictions: EvaluationPoint[];
  evaluation_duration_seconds: number;
  is_validation_winner: boolean;
  validation_metrics?: Record<string, number> | null;
  error_message?: string | null;
}

interface BenchmarkResponse {
  evaluation_id: string;
  dataset_id: string;
  entity_id: string;
  product_id?: string | null;
  status: string;
  horizon: number;
  test_period: {
    start: string;
    end: string;
    total_available_days: number;
    evaluated_days: number;
  };
  training_period: {
    train_start: string;
    train_end: string;
    val_start: string;
    val_end: string;
    history_end: string;
  };
  models: ModelTestEvaluation[];
  phase4_selected_model: string;
  phase4_selection_metric: string;
  phase4_validation_score?: number | null;
  selection_source: string;
  test_set_used_for_selection: boolean;
  test_winner_for_reporting_only?: string | null;
  benchmark_summary_markdown?: string | null;
  warnings?: string[];
}

export default function EvaluationPanel() {
  const [entityId, setEntityId] = useState("STORE_001");
  const [productId, setProductId] = useState("PROD_001");
  const [horizon, setHorizon] = useState(14);
  const [selectedModels, setSelectedModels] = useState<string[]>([
    "prophet",
    "lightgbm",
    "lstm",
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const toggleModel = (model: string) => {
    if (selectedModels.includes(model)) {
      if (selectedModels.length > 1) {
        setSelectedModels(selectedModels.filter((m) => m !== model));
      }
    } else {
      setSelectedModels([...selectedModels, model]);
    }
  };

  const handleRunBenchmark = async () => {
    setIsLoading(true);
    setError(null);

    const payload = {
      entity_id: entityId.trim(),
      product_id: productId.trim(),
      horizon: Number(horizon),
      candidate_models: selectedModels,
      random_seed: 42,
    };

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${apiUrl}/api/v1/evaluation/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with status ${res.status}`);
      }

      const data: BenchmarkResponse = await res.json();
      setBenchmarkResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred during benchmark.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800/80 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-emerald-400 bg-emerald-950/80 px-2.5 py-0.5 rounded border border-emerald-800">
              Phase 5
            </span>
            <h2 className="text-lg font-bold text-white tracking-tight">
              Formal Forecasting Evaluation & Test Benchmark
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Rigorous quantitative holdout test-set evaluation across Prophet, LightGBM, and PyTorch LSTM.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-cyan-950/60 text-cyan-300 border border-cyan-800/60">
            Protected Test Set (0% Used for Selection)
          </span>
        </div>
      </div>

      {/* Protocol Notice Banner */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 text-xs text-slate-300 space-y-1.5">
        <div className="flex items-center gap-2 font-semibold text-cyan-400">
          <span>🛡️ Strict Academic Evaluation Protocol</span>
        </div>
        <p className="text-slate-400 leading-relaxed">
          Model selection was conducted exclusively in Phase 4 using the chronological <strong>Validation Partition</strong> (70/15/15 split). 
          Test-set metrics (MAE, RMSE, MAPE) are calculated against untouched holdout test targets as an independent scientific benchmark.
        </p>
      </div>

      {/* Configuration Form */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 bg-slate-950/50 p-4 rounded-xl border border-slate-800/60">
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
            Entity ID
          </label>
          <input
            type="text"
            value={entityId}
            onChange={(e) => setEntityId(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500 font-mono"
            placeholder="e.g. STORE_001"
          />
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
            Product ID
          </label>
          <input
            type="text"
            value={productId}
            onChange={(e) => setProductId(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500 font-mono"
            placeholder="e.g. PROD_001"
          />
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
            Evaluation Horizon
          </label>
          <select
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value={7}>7 Days (Short-Term)</option>
            <option value={14}>14 Days (Standard Benchmark)</option>
            <option value={28}>28 Days (Full Test Window)</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
            Candidate Models
          </label>
          <div className="flex items-center gap-2 pt-1.5">
            {["prophet", "lightgbm", "lstm"].map((model) => (
              <label
                key={model}
                className={`flex items-center gap-1 text-xs cursor-pointer px-2.5 py-1 rounded-md border transition-all ${
                  selectedModels.includes(model)
                    ? "bg-cyan-950/80 border-cyan-700 text-cyan-300"
                    : "bg-slate-900 border-slate-800 text-slate-500"
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedModels.includes(model)}
                  onChange={() => toggleModel(model)}
                  className="hidden"
                />
                <span className="capitalize">{model}</span>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Action Button */}
      <div className="flex items-center justify-end gap-3">
        {error && <span className="text-xs text-rose-400 mr-auto font-mono">⚠️ {error}</span>}
        <button
          onClick={handleRunBenchmark}
          disabled={isLoading}
          className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white font-semibold text-xs tracking-wider uppercase transition-all shadow-lg shadow-cyan-950/50 disabled:opacity-50 flex items-center gap-2"
        >
          {isLoading ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              <span>Evaluating Test Set...</span>
            </>
          ) : (
            <>
              <span>⚡</span>
              <span>Execute Holdout Benchmark</span>
            </>
          )}
        </button>
      </div>

      {/* Results Section */}
      {benchmarkResult && (
        <div className="space-y-6 pt-2 border-t border-slate-800">
          {/* Phase 4 Selection Status & Separation Guarantee */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4">
              <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">
                Phase 4 Validation Winner
              </div>
              <div className="text-xl font-black text-emerald-400 uppercase mt-1 flex items-center gap-2">
                <span>⭐ {benchmarkResult.phase4_selected_model}</span>
              </div>
              <div className="text-xs text-slate-400 mt-1">
                Selected via Validation {benchmarkResult.phase4_selection_metric} (
                {benchmarkResult.phase4_validation_score?.toFixed(4) || "N/A"})
              </div>
            </div>

            <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4">
              <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">
                Test Set Usage Invariant
              </div>
              <div className="text-lg font-bold text-cyan-300 mt-1 flex items-center gap-2">
                <span>🔒 Strictly Protected</span>
              </div>
              <div className="text-xs text-slate-400 mt-1">
                Test Set Used for Selection:{" "}
                <strong className="text-rose-400">
                  {benchmarkResult.test_set_used_for_selection ? "TRUE" : "FALSE"}
                </strong>
              </div>
            </div>

            <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4">
              <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">
                Evaluated Test Period
              </div>
              <div className="text-sm font-semibold text-white mt-1 font-mono">
                {benchmarkResult.test_period.start} → {benchmarkResult.test_period.end}
              </div>
              <div className="text-xs text-slate-400 mt-1">
                {benchmarkResult.horizon} Days Evaluated ({benchmarkResult.test_period.total_available_days} Days Holdout)
              </div>
            </div>
          </div>

          {/* Benchmark Comparison Table */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                Formal Test Benchmark Comparison
              </h3>
              <span className="text-xs text-slate-400">Zero-safe MAPE calculation</span>
            </div>

            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-900/90 text-slate-400 font-semibold border-b border-slate-800">
                    <th className="py-3 px-4">Candidate Model</th>
                    <th className="py-3 px-4">Test MAE</th>
                    <th className="py-3 px-4">Test RMSE</th>
                    <th className="py-3 px-4">Test MAPE</th>
                    <th className="py-3 px-4">Val MAE (Phase 4)</th>
                    <th className="py-3 px-4">Phase 4 Selected?</th>
                    <th className="py-3 px-4">Execution Time</th>
                    <th className="py-3 px-4">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
                  {benchmarkResult.models.map((m) => {
                    const isWinner = m.model_name === benchmarkResult.phase4_selected_model;
                    return (
                      <tr
                        key={m.model_name}
                        className={isWinner ? "bg-emerald-950/20" : "hover:bg-slate-900/40"}
                      >
                        <td className="py-3 px-4 font-mono font-bold capitalize text-white flex items-center gap-2">
                          {isWinner && <span className="text-emerald-400">⭐</span>}
                          <span>{m.model_name}</span>
                        </td>
                        <td className="py-3 px-4 font-mono text-cyan-300 font-semibold">
                          {m.mae !== undefined && m.mae !== null ? m.mae.toFixed(4) : "N/A"}
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-300">
                          {m.rmse !== undefined && m.rmse !== null ? m.rmse.toFixed(4) : "N/A"}
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-300">
                          {m.mape !== undefined && m.mape !== null ? `${m.mape.toFixed(2)}%` : "N/A"}
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-400">
                          {m.validation_metrics?.MAE !== undefined
                            ? m.validation_metrics.MAE.toFixed(4)
                            : "N/A"}
                        </td>
                        <td className="py-3 px-4">
                          {isWinner ? (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                              Selected Winner
                            </span>
                          ) : (
                            <span className="text-slate-500">Candidate</span>
                          )}
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-400">
                          {m.evaluation_duration_seconds.toFixed(2)}s
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium ${
                              m.status === "SUCCESS"
                                ? "bg-emerald-950/60 text-emerald-300 border border-emerald-800"
                                : "bg-rose-950/60 text-rose-300 border border-rose-800"
                            }`}
                          >
                            {m.status}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Visual Test-Set Comparison: Actual Test Data vs Model Predictions */}
          {benchmarkResult.models.length > 0 && benchmarkResult.models[0].predictions.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                  Test-Set Point Comparison (Actual vs Forecast)
                </h3>
                <div className="flex items-center gap-3 text-xs">
                  <span className="flex items-center gap-1.5 text-emerald-400 font-semibold">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
                    ACTUAL TEST DATA
                  </span>
                  <span className="flex items-center gap-1.5 text-cyan-400 font-semibold">
                    <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
                    MODEL PREDICTIONS
                  </span>
                </div>
              </div>

              <div className="overflow-x-auto rounded-xl border border-slate-800 max-h-72 overflow-y-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead className="sticky top-0 bg-slate-900 text-slate-400 font-semibold border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-4">Date</th>
                      <th className="py-2.5 px-4 text-emerald-400 font-bold">Actual Target</th>
                      {benchmarkResult.models.map((m) => (
                        <th key={m.model_name} className="py-2.5 px-4 capitalize text-cyan-300">
                          {m.model_name} Pred
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 bg-slate-950/60 font-mono">
                    {benchmarkResult.models[0].predictions.map((pt, idx) => (
                      <tr key={pt.date} className="hover:bg-slate-900/40">
                        <td className="py-2 px-4 text-slate-300">{pt.date}</td>
                        <td className="py-2 px-4 text-emerald-400 font-bold">{pt.actual.toFixed(2)}</td>
                        {benchmarkResult.models.map((m) => {
                          const predPt = m.predictions[idx];
                          return (
                            <td key={m.model_name} className="py-2 px-4 text-slate-200">
                              {predPt ? predPt.prediction.toFixed(2) : "N/A"}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
