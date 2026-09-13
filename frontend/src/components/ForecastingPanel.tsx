"use client";

import React, { useState } from "react";

interface ForecastPoint {
  date: string;
  prediction: number;
  predicted_value?: number;
  lower_bound?: number | null;
  upper_bound?: number | null;
}

interface ModelScore {
  model_name: string;
  model_type: string;
  mae: number;
  rmse: number;
  mape: number;
  training_duration: number;
  status: string;
  error_message?: string | null;
}

interface ForecastResponse {
  forecast_id: string;
  model_name: string;
  entity_id: string;
  product_id?: string | null;
  horizon: number;
  frequency: string;
  status: string;
  uncertainty_method?: string | null;
  confidence_level?: number | null;
  predictions: ForecastPoint[];
  selection_result?: {
    selected_model: string;
    selection_metric: string;
    selection_reason: string;
    candidate_rankings: ModelScore[];
  } | null;
  warnings?: string[];
}

export default function ForecastingPanel() {
  const [entityId, setEntityId] = useState("STORE_001");
  const [productId, setProductId] = useState("PROD_001");
  const [horizon, setHorizon] = useState(14);
  const [selectedModels, setSelectedModels] = useState<string[]>([
    "prophet",
    "lightgbm",
    "lstm",
  ]);
  const [metric, setMetric] = useState("MAE");
  const [isLoading, setIsLoading] = useState(false);
  const [forecastResult, setForecastResult] = useState<ForecastResponse | null>(null);
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

  const handleRunForecast = async () => {
    setIsLoading(true);
    setError(null);

    const payload = {
      entity_id: entityId.trim(),
      product_id: productId.trim(),
      horizon: Number(horizon),
      candidate_models: selectedModels,
      selection_metric: metric,
      confidence_level: 0.8,
    };

    try {
      const res = await fetch("http://localhost:8000/api/v1/forecast/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({ detail: "Network error" }));
        throw new Error(errJson.detail || `Server returned status ${res.status}`);
      }

      const data: ForecastResponse = await res.json();
      setForecastResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to execute forecast request.");
    } finally {
      setIsLoading(false);
    }
  };

  // Helper to render SVG Chart
  const renderChart = (predictions: ForecastPoint[]) => {
    if (!predictions || predictions.length === 0) return null;

    const width = 720;
    const height = 260;
    const padding = { top: 25, right: 30, bottom: 40, left: 55 };

    const allValues = predictions.flatMap((p) => {
      const pred = p.prediction ?? p.predicted_value ?? 0;
      const vals = [pred];
      if (p.lower_bound != null) vals.push(p.lower_bound);
      if (p.upper_bound != null) vals.push(p.upper_bound);
      return vals;
    });

    const minVal = Math.max(0, Math.floor(Math.min(...allValues) * 0.9));
    const maxVal = Math.ceil(Math.max(...allValues) * 1.1) || 10;
    const valRange = maxVal - minVal || 1;

    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const getX = (idx: number) =>
      padding.left + (idx / Math.max(1, predictions.length - 1)) * chartW;
    const getY = (val: number) =>
      padding.top + chartH - ((val - minVal) / valRange) * chartH;

    // Points for prediction line
    const linePath = predictions
      .map((p, i) => `${i === 0 ? "M" : "L"} ${getX(i)} ${getY(p.prediction ?? p.predicted_value ?? 0)}`)
      .join(" ");

    // Polygon for uncertainty interval
    const hasIntervals = predictions.some(
      (p) => p.lower_bound != null && p.upper_bound != null
    );
    let bandPath = "";
    if (hasIntervals) {
      const upperPoints = predictions.map(
        (p, i) => `${getX(i)},${getY(p.upper_bound ?? (p.prediction ?? p.predicted_value ?? 0))}`
      );
      const lowerPoints = [...predictions]
        .reverse()
        .map(
          (p, i) =>
            `${getX(predictions.length - 1 - i)},${getY(
              p.lower_bound ?? (p.prediction ?? p.predicted_value ?? 0)
            )}`
        );
      bandPath = `M ${upperPoints.join(" L ")} L ${lowerPoints.join(" L ")} Z`;
    }

    return (
      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto min-w-[500px] select-none"
        >
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const y = padding.top + chartH * ratio;
            const val = Math.round(maxVal - ratio * valRange);
            return (
              <g key={ratio}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={width - padding.right}
                  y2={y}
                  stroke="#1e293b"
                  strokeDasharray="4 4"
                />
                <text
                  x={padding.left - 10}
                  y={y + 4}
                  fill="#64748b"
                  fontSize="10"
                  textAnchor="end"
                  fontFamily="monospace"
                >
                  {val}
                </text>
              </g>
            );
          })}

          {/* Uncertainty Band */}
          {hasIntervals && (
            <path
              d={bandPath}
              fill="rgba(6, 182, 212, 0.15)"
              stroke="rgba(6, 182, 212, 0.4)"
              strokeDasharray="2 2"
            />
          )}

          {/* Point Forecast Line */}
          <path
            d={linePath}
            fill="none"
            stroke="#06b6d4"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Data Points */}
          {predictions.map((p, i) => (
            <circle
              key={p.date}
              cx={getX(i)}
              cy={getY(p.prediction ?? p.predicted_value ?? 0)}
              r="4"
              fill="#0891b2"
              stroke="#e0f2fe"
              strokeWidth="1.5"
            />
          ))}

          {/* X Axis Labels */}
          {predictions.map((p, i) => {
            if (predictions.length > 10 && i % 2 !== 0) return null;
            return (
              <text
                key={`label-${p.date}`}
                x={getX(i)}
                y={height - 12}
                fill="#94a3b8"
                fontSize="9"
                textAnchor="middle"
                fontFamily="monospace"
              >
                {p.date.slice(5)}
              </text>
            );
          })}
        </svg>
      </div>
    );
  };

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-widest text-cyan-400 bg-cyan-950/80 px-2.5 py-0.5 rounded border border-cyan-800">
              Phase 4 Live Verification
            </span>
            <span className="text-xs font-medium text-slate-400">
              Model-Agnostic Forecasting Agent
            </span>
          </div>
          <h2 className="text-xl font-bold text-white mt-1">
            Time-Series Multi-Model Competition & Forecasting
          </h2>
        </div>
        <button
          id="run-forecast-btn"
          onClick={handleRunForecast}
          disabled={isLoading}
          className={`px-5 py-2.5 rounded-xl font-semibold text-sm transition-all duration-200 flex items-center gap-2 ${
            isLoading
              ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700"
              : "bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/20 active:scale-95"
          }`}
        >
          {isLoading ? (
            <>
              <svg className="animate-spin h-4 w-4 text-slate-400" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
              </svg>
              <span>Evaluating Candidates...</span>
            </>
          ) : (
            <>
              <span>⚡</span> Run Forecasting Agent
            </>
          )}
        </button>
      </div>

      {/* Interactive Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 p-4 rounded-xl bg-slate-950/50 border border-slate-800/80">
        {/* Series: Entity */}
        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1.5">
            Entity ID (e.g. Store)
          </label>
          <input
            id="forecast-entity-id"
            type="text"
            value={entityId}
            onChange={(e) => setEntityId(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-100 focus:outline-none focus:border-cyan-500 font-mono"
            placeholder="STORE_001"
          />
        </div>

        {/* Series: Product */}
        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1.5">
            Product ID (Optional)
          </label>
          <input
            id="forecast-product-id"
            type="text"
            value={productId}
            onChange={(e) => setProductId(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-100 focus:outline-none focus:border-cyan-500 font-mono"
            placeholder="PROD_001"
          />
        </div>

        {/* Horizon Selector */}
        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1.5">
            Forecast Horizon (Days)
          </label>
          <select
            id="forecast-horizon-select"
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-100 focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value={7}>7 Days</option>
            <option value={14}>14 Days (Standard)</option>
            <option value={28}>28 Days (Monthly)</option>
          </select>
        </div>

        {/* Validation Metric */}
        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1.5">
            Selection Metric (Validation)
          </label>
          <select
            id="forecast-metric-select"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-100 focus:outline-none focus:border-cyan-500 font-mono"
          >
            <option value="MAE">MAE (Mean Absolute Error)</option>
            <option value="RMSE">RMSE (Root Mean Squared Error)</option>
            <option value="MAPE">MAPE (Mean Absolute % Error)</option>
          </select>
        </div>
      </div>

      {/* Candidate Model Checkboxes */}
      <div className="flex flex-wrap items-center gap-4 text-xs">
        <span className="text-slate-400 font-medium">Candidate Models:</span>
        {["prophet", "lightgbm", "lstm"].map((model) => (
          <label
            key={model}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border cursor-pointer select-none transition-colors ${
              selectedModels.includes(model)
                ? "bg-cyan-950/40 border-cyan-700/80 text-cyan-300 font-semibold"
                : "bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200"
            }`}
          >
            <input
              type="checkbox"
              checked={selectedModels.includes(model)}
              onChange={() => toggleModel(model)}
              className="accent-cyan-500 rounded"
            />
            <span className="capitalize">{model}</span>
          </label>
        ))}
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-950/30 border border-rose-800/80 text-rose-300 text-sm">
          <strong>Forecast Request Failed:</strong> {error}
        </div>
      )}

      {/* Forecast Results View */}
      {forecastResult && (
        <div className="space-y-6 pt-4 border-t border-slate-800">
          {/* Winner Overview Card */}
          <div className="p-4 rounded-xl bg-gradient-to-r from-cyan-950/40 to-blue-950/30 border border-cyan-800/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded text-[11px] font-bold uppercase tracking-wider bg-emerald-950/80 text-emerald-400 border border-emerald-800">
                  {forecastResult.status}
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  ID: {forecastResult.forecast_id}
                </span>
              </div>
              <h3 className="text-lg font-bold text-white mt-1">
                Selected Model:{" "}
                <span className="text-cyan-400 uppercase font-mono">
                  {forecastResult.model_name}
                </span>
              </h3>
              <p className="text-xs text-slate-300 mt-0.5">
                {forecastResult.selection_result?.selection_reason ||
                  "Selected via internal validation score."}
              </p>
            </div>
            <div className="text-right text-xs text-slate-400 font-mono space-y-1">
              <div>Horizon: {forecastResult.horizon} Periods ({forecastResult.frequency})</div>
              <div>Uncertainty: {forecastResult.uncertainty_method || "None"} ({(forecastResult.confidence_level || 0.8) * 100}%)</div>
            </div>
          </div>

          {/* Model Ranking Table */}
          {forecastResult.selection_result?.candidate_rankings && (
            <div>
              <h4 className="text-sm font-semibold text-slate-300 mb-2">
                Candidate Validation Comparison (Holdout Test Excluded)
              </h4>
              <div className="overflow-x-auto rounded-xl border border-slate-800">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-900/80 text-slate-400 font-mono uppercase text-[10px] border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-4">Rank</th>
                      <th className="py-2.5 px-4">Candidate Model</th>
                      <th className="py-2.5 px-4">Architecture</th>
                      <th className="py-2.5 px-4">MAE</th>
                      <th className="py-2.5 px-4">RMSE</th>
                      <th className="py-2.5 px-4">MAPE (%)</th>
                      <th className="py-2.5 px-4">Train Time (s)</th>
                      <th className="py-2.5 px-4">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                    {forecastResult.selection_result.candidate_rankings.map(
                      (cand, idx) => {
                        const isWinner =
                          cand.model_name ===
                          forecastResult.selection_result?.selected_model;
                        return (
                          <tr
                            key={cand.model_name}
                            className={
                              isWinner
                                ? "bg-cyan-950/20 text-cyan-200 font-medium"
                                : "hover:bg-slate-900/30"
                            }
                          >
                            <td className="py-2.5 px-4">
                              {idx + 1} {isWinner ? "🏆" : ""}
                            </td>
                            <td className="py-2.5 px-4 font-semibold uppercase">
                              {cand.model_name}
                            </td>
                            <td className="py-2.5 px-4 text-slate-400">
                              {cand.model_type}
                            </td>
                            <td className="py-2.5 px-4">
                              {cand.mae > 0 ? cand.mae.toFixed(3) : "—"}
                            </td>
                            <td className="py-2.5 px-4">
                              {cand.rmse > 0 ? cand.rmse.toFixed(3) : "—"}
                            </td>
                            <td className="py-2.5 px-4">
                              {cand.mape > 0 ? cand.mape.toFixed(2) + "%" : "—"}
                            </td>
                            <td className="py-2.5 px-4">
                              {cand.training_duration.toFixed(2)}s
                            </td>
                            <td className="py-2.5 px-4">
                              <span
                                className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold ${
                                  cand.status === "success"
                                    ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800"
                                    : "bg-rose-950/80 text-rose-400 border border-rose-800"
                                }`}
                              >
                                {cand.status}
                              </span>
                            </td>
                          </tr>
                        );
                      }
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Forecast Chart & Uncertainty Interval */}
          <div className="p-4 rounded-xl bg-slate-950/40 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-300">
                Future Forecast Trajectory & Uncertainty Interval ({(forecastResult.confidence_level || 0.8) * 100}%)
              </h4>
              <div className="flex items-center gap-4 text-xs font-mono">
                <span className="flex items-center gap-1.5 text-cyan-400">
                  <span className="w-3 h-0.5 bg-cyan-400 inline-block"></span> Point Forecast
                </span>
                <span className="flex items-center gap-1.5 text-cyan-200/70">
                  <span className="w-3 h-2 bg-cyan-500/20 border border-cyan-500/40 inline-block"></span> Prediction Interval
                </span>
              </div>
            </div>
            {renderChart(forecastResult.predictions)}
          </div>

          {/* Predictions Data Table */}
          <div>
            <h4 className="text-sm font-semibold text-slate-300 mb-2">
              Forecast Points & Prediction Bounds
            </h4>
            <div className="max-h-64 overflow-y-auto rounded-xl border border-slate-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-900/80 text-slate-400 font-mono uppercase text-[10px] border-b border-slate-800 sticky top-0">
                  <tr>
                    <th className="py-2 px-4">Period</th>
                    <th className="py-2 px-4">Date</th>
                    <th className="py-2 px-4">Point Prediction</th>
                    <th className="py-2 px-4">Lower Bound</th>
                    <th className="py-2 px-4">Upper Bound</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                  {forecastResult.predictions.map((pt, i) => (
                    <tr key={pt.date} className="hover:bg-slate-900/30">
                      <td className="py-2 px-4 text-slate-500">t+{i + 1}</td>
                      <td className="py-2 px-4 font-semibold text-slate-200">
                        {pt.date}
                      </td>
                      <td className="py-2 px-4 text-cyan-400 font-bold">
                        {(pt.prediction ?? pt.predicted_value ?? 0).toFixed(2)}
                      </td>
                      <td className="py-2 px-4 text-slate-400">
                        {pt.lower_bound != null ? pt.lower_bound.toFixed(2) : "—"}
                      </td>
                      <td className="py-2 px-4 text-slate-400">
                        {pt.upper_bound != null ? pt.upper_bound.toFixed(2) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
