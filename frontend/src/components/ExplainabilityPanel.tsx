"use client";

import React, { useState } from "react";

interface FeatureContribution {
  feature: string;
  value?: number | null;
  contribution: number;
  shap_value?: number | null;
  absolute_contribution: number;
  direction: string;
  rank: number;
}

interface GlobalFeatureImportance {
  feature: string;
  importance_score: number;
  rank: number;
  normalized_importance?: number | null;
}

interface ExplanationFidelity {
  fidelity_score: number;
  reconstruction_error: number;
  surrogate_r2?: number | null;
  base_value?: number | null;
  reconstructed_prediction?: number | null;
  actual_prediction?: number | null;
  explanation_status: string;
  method_notes?: string | null;
}

interface ExplanationAuditTrail {
  explanation_id: string;
  model_name: string;
  model_type?: string | null;
  entity_id?: string | null;
  product_id?: string | null;
  prediction_date?: string | null;
  prediction: number;
  explanation_method: string;
  explanation_type: string;
  feature_count: number;
  random_seed?: number | null;
  sample_size?: number | null;
  duration_seconds: number;
  timestamp: string;
}

interface ExplanationResult {
  explanation_id: string;
  model_name: string;
  method: string;
  explanation_type: string;
  prediction: number;
  prediction_date?: string | null;
  base_value?: number | null;
  features: FeatureContribution[];
  top_positive_contributors: FeatureContribution[];
  top_negative_contributors: FeatureContribution[];
  global_importance: GlobalFeatureImportance[];
  fidelity: ExplanationFidelity;
  audit_trail: ExplanationAuditTrail;
  warnings: string[];
}

export default function ExplainabilityPanel() {
  const [modelName, setModelName] = useState<string>("lightgbm");
  const [method, setMethod] = useState<string>("shap");
  const [explType, setExplType] = useState<"local" | "global">("local");
  const [backgroundSamples, setBackgroundSamples] = useState<number>(30);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ExplanationResult | null>(null);
  const [showAudit, setShowAudit] = useState<boolean>(false);

  // Sync default method when model changes
  const handleModelChange = (m: string) => {
    setModelName(m);
    if (m === "prophet") {
      setMethod("component_based");
    } else if (method === "component_based") {
      setMethod("shap");
    }
  };

  const handleRunExplanation = async () => {
    setLoading(true);
    setError(null);

    const endpoint =
      explType === "local"
        ? "http://127.0.0.1:8000/api/v1/explainability/local"
        : "http://127.0.0.1:8000/api/v1/explainability/global";

    const payload =
      explType === "local"
        ? {
            entity_id: "STORE_001",
            product_id: "PROD_001",
            model_name: modelName,
            method: method,
            background_samples: backgroundSamples,
            random_seed: 42,
          }
        : {
            entity_id: "STORE_001",
            product_id: "PROD_001",
            model_name: modelName,
            method: method,
            sample_size: backgroundSamples,
            top_k: 15,
            random_seed: 42,
          };

    try {
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: HTTP ${resp.status}`);
      }

      const data: ExplanationResult = await resp.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to generate explanation");
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch("http://127.0.0.1:8000/api/v1/explainability/sample");
      if (!resp.ok) {
        throw new Error(`Failed to load sample explanation: HTTP ${resp.status}`);
      }
      const data: ExplanationResult = await resp.json();
      setResult(data);
      setModelName(data.model_name);
      setMethod(data.method);
      setExplType("local");
    } catch (err: any) {
      setError(err.message || "Failed to load sample");
    } finally {
      setLoading(false);
    }
  };

  // Compute max absolute contribution for bar scaling
  const maxContrib = result?.features
    ? Math.max(...result.features.map((f) => Math.abs(f.contribution)), 0.01)
    : 1.0;

  const maxGlobalScore = result?.global_importance
    ? Math.max(...result.global_importance.map((g) => g.importance_score), 0.01)
    : 1.0;

  return (
    <div className="glass-panel rounded-2xl p-6 border border-purple-900/50 bg-slate-950/40 relative overflow-hidden">
      <div className="absolute -left-10 -bottom-10 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl pointer-events-none"></div>

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6 relative z-10">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-widest text-purple-400 bg-purple-950/80 px-2.5 py-0.5 rounded-md border border-purple-800">
              Phase 6 Active
            </span>
            <span className="text-xs text-slate-400 font-mono">XAI Verification Engine</span>
          </div>
          <h3 className="text-xl font-bold text-white tracking-tight mt-1.5 flex items-center gap-2">
            <span>Explainability Agent</span>
            <span className="text-xs px-2 py-0.5 rounded font-mono bg-purple-900/40 text-purple-300 border border-purple-700/50">
              SHAP · LIME · Component Decomposition
            </span>
          </h3>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Inspect instance-level local attributions and global feature importance rankings
            with mathematical reconstruction fidelity and zero test-set leakage.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleLoadSample}
            disabled={loading}
            className="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-750 text-xs font-semibold text-slate-300 hover:text-white transition disabled:opacity-50"
          >
            Load Sample
          </button>
          <button
            onClick={handleRunExplanation}
            disabled={loading}
            className="px-4 py-1.5 rounded-lg bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-xs font-bold text-white shadow-lg shadow-purple-900/40 transition disabled:opacity-50 flex items-center gap-1.5"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-3.5 w-3.5 text-white" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                <span>Attributing...</span>
              </>
            ) : (
              <span>Explain Forecast</span>
            )}
          </button>
        </div>
      </div>

      {/* Control Panel */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6 relative z-10 bg-slate-900/60 p-3.5 rounded-xl border border-slate-800/80">
        <div>
          <label className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
            Model
          </label>
          <select
            value={modelName}
            onChange={(e) => handleModelChange(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-md px-2.5 py-1 text-xs focus:border-purple-500 focus:outline-none"
          >
            <option value="lightgbm">LightGBM (GBDT)</option>
            <option value="lstm">PyTorch LSTM</option>
            <option value="prophet">Prophet (GAM)</option>
          </select>
        </div>

        <div>
          <label className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
            Method
          </label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-md px-2.5 py-1 text-xs focus:border-purple-500 focus:outline-none"
          >
            {modelName === "prophet" ? (
              <option value="component_based">Component Decomposition</option>
            ) : (
              <>
                <option value="shap">SHAP (Attribution)</option>
                <option value="lime">LIME (Surrogate)</option>
              </>
            )}
          </select>
        </div>

        <div>
          <label className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
            Scope
          </label>
          <div className="flex rounded-md bg-slate-950 p-0.5 border border-slate-700">
            <button
              onClick={() => setExplType("local")}
              className={`flex-1 py-0.5 text-xs font-semibold rounded ${
                explType === "local" ? "bg-purple-600 text-white" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Local
            </button>
            <button
              onClick={() => setExplType("global")}
              className={`flex-1 py-0.5 text-xs font-semibold rounded ${
                explType === "global" ? "bg-purple-600 text-white" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Global
            </button>
          </div>
        </div>

        <div>
          <label className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
            Background Samples
          </label>
          <input
            type="number"
            min={10}
            max={100}
            value={backgroundSamples}
            onChange={(e) => setBackgroundSamples(parseInt(e.target.value) || 30)}
            className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-md px-2.5 py-1 text-xs focus:border-purple-500 focus:outline-none"
          />
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-3 mb-4 rounded-lg bg-red-950/60 border border-red-800/80 text-red-200 text-xs flex items-center gap-2">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* Results View */}
      {result && (
        <div className="space-y-5 relative z-10">
          {/* Summary Metric Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                Point Forecast
              </span>
              <div className="text-xl font-black text-white mt-1">
                {result.prediction.toFixed(2)}{" "}
                <span className="text-xs font-normal text-slate-400">units</span>
              </div>
              <span className="text-[10px] text-slate-500 font-mono">
                {result.prediction_date || "Reference Period"}
              </span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                Base Value
              </span>
              <div className="text-xl font-black text-purple-300 mt-1">
                {result.base_value !== null && result.base_value !== undefined
                  ? result.base_value.toFixed(2)
                  : "0.00"}
              </div>
              <span className="text-[10px] text-slate-500">Expected / Prior</span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                Fidelity Score
              </span>
              <div className="text-xl font-black text-emerald-400 mt-1">
                {(result.fidelity.fidelity_score * 100).toFixed(1)}%
              </div>
              <span className="text-[10px] text-emerald-500/80 font-mono">
                {result.fidelity.explanation_status}
              </span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-xl">
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                Reconstruction Error
              </span>
              <div className="text-xl font-black text-slate-300 mt-1 font-mono">
                {result.fidelity.reconstruction_error.toFixed(4)}
              </div>
              <span className="text-[10px] text-slate-500 font-mono">
                {result.audit_trail.duration_seconds.toFixed(2)}s runtime
              </span>
            </div>
          </div>

          {/* Local Explanation View */}
          {result.explanation_type === "local" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Positive vs Negative Contributors */}
              <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                    Feature Impact Breakdown
                  </h4>
                  <span className="text-[10px] font-mono text-slate-400">
                    Method: {result.method.toUpperCase()}
                  </span>
                </div>

                {/* Top Positive Drivers */}
                <div>
                  <div className="text-[11px] font-bold text-emerald-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <span>▲</span>
                    <span>Top Positive Contributors (Pushes Forecast Higher)</span>
                  </div>
                  <div className="space-y-2">
                    {result.top_positive_contributors.slice(0, 4).map((f) => (
                      <div key={f.feature} className="text-xs">
                        <div className="flex justify-between text-slate-300 mb-0.5">
                          <span className="font-mono text-slate-200">{f.feature}</span>
                          <span className="font-mono font-bold text-emerald-400">
                            +{f.contribution.toFixed(4)}
                          </span>
                        </div>
                        <div className="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-emerald-500 h-1.5 rounded-full"
                            style={{
                              width: `${Math.min(100, (Math.abs(f.contribution) / maxContrib) * 100)}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                    {result.top_positive_contributors.length === 0 && (
                      <p className="text-xs text-slate-500 italic">No positive contributors.</p>
                    )}
                  </div>
                </div>

                {/* Top Negative Drivers */}
                <div>
                  <div className="text-[11px] font-bold text-rose-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <span>▼</span>
                    <span>Top Negative Contributors (Pulls Forecast Lower)</span>
                  </div>
                  <div className="space-y-2">
                    {result.top_negative_contributors.slice(0, 4).map((f) => (
                      <div key={f.feature} className="text-xs">
                        <div className="flex justify-between text-slate-300 mb-0.5">
                          <span className="font-mono text-slate-200">{f.feature}</span>
                          <span className="font-mono font-bold text-rose-400">
                            {f.contribution.toFixed(4)}
                          </span>
                        </div>
                        <div className="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-rose-500 h-1.5 rounded-full"
                            style={{
                              width: `${Math.min(100, (Math.abs(f.contribution) / maxContrib) * 100)}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                    {result.top_negative_contributors.length === 0 && (
                      <p className="text-xs text-slate-500 italic">No negative contributors.</p>
                    )}
                  </div>
                </div>
              </div>

              {/* Ranked Attribution Table */}
              <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                    Ranked Feature Attribution
                  </h4>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {result.features.length} Features Attributed
                  </span>
                </div>

                <div className="max-h-72 overflow-y-auto pr-1">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-[10px] uppercase font-bold text-slate-400">
                        <th className="py-1 px-2">#</th>
                        <th className="py-1 px-2">Feature</th>
                        <th className="py-1 px-2 text-right">Value</th>
                        <th className="py-1 px-2 text-right">Impact</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-850">
                      {result.features.map((f) => (
                        <tr key={f.feature} className="hover:bg-slate-800/40">
                          <td className="py-1 px-2 text-slate-500 font-mono">{f.rank}</td>
                          <td className="py-1 px-2 font-mono text-slate-200">{f.feature}</td>
                          <td className="py-1 px-2 text-right font-mono text-slate-400">
                            {f.value !== null && f.value !== undefined ? f.value.toFixed(2) : "—"}
                          </td>
                          <td
                            className={`py-1 px-2 text-right font-mono font-bold ${
                              f.contribution > 0
                                ? "text-emerald-400"
                                : f.contribution < 0
                                ? "text-rose-400"
                                : "text-slate-400"
                            }`}
                          >
                            {f.contribution > 0 ? `+${f.contribution.toFixed(4)}` : f.contribution.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Global Feature Importance View */}
          {result.explanation_type === "global" && (
            <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                  Global Feature Importance Ranking (Dataset Scope)
                </h4>
                <span className="text-[10px] text-slate-400 font-mono">
                  Method: {result.method.toUpperCase()}
                </span>
              </div>

              <div className="space-y-2.5">
                {result.global_importance.map((gf) => (
                  <div key={gf.feature} className="text-xs">
                    <div className="flex justify-between items-center mb-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-bold text-slate-400 w-5">
                          {gf.rank}.
                        </span>
                        <span className="font-mono text-slate-200">{gf.feature}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="font-mono text-purple-300 font-bold">
                          {gf.importance_score.toFixed(4)}
                        </span>
                        {gf.normalized_importance !== null && gf.normalized_importance !== undefined && (
                          <span className="text-[10px] font-mono text-slate-400 w-12 text-right">
                            {(gf.normalized_importance * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-purple-600 to-indigo-500 h-1.5 rounded-full"
                        style={{
                          width: `${Math.min(100, (gf.importance_score / maxGlobalScore) * 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Audit Trail Drawer Toggle */}
          <div className="pt-2 border-t border-slate-900 flex justify-between items-center text-xs">
            <span className="text-slate-500 font-mono">
              ID: {result.explanation_id}
            </span>
            <button
              onClick={() => setShowAudit(!showAudit)}
              className="text-purple-400 hover:text-purple-300 font-semibold"
            >
              {showAudit ? "Hide Audit Details" : "View Audit Payload"}
            </button>
          </div>

          {showAudit && (
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs font-mono text-slate-300 space-y-1">
              <div><strong>Explanation ID:</strong> {result.audit_trail.explanation_id}</div>
              <div><strong>Model Architecture:</strong> {result.audit_trail.model_name} ({result.audit_trail.model_type})</div>
              <div><strong>Method Used:</strong> {result.audit_trail.explanation_method}</div>
              <div><strong>Reproducibility Seed:</strong> {result.audit_trail.random_seed}</div>
              <div><strong>Reference Samples:</strong> {result.audit_trail.sample_size}</div>
              <div><strong>Methodology Notes:</strong> {result.fidelity.method_notes}</div>
              <div><strong>Generated At:</strong> {result.audit_trail.timestamp}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
