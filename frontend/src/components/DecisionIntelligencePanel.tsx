"use client";

import React, { useState, useEffect } from "react";

interface TradeOff {
  benefit: string;
  trade_off: string;
  quantified_impact?: string | null;
}

interface RecommendationItem {
  recommendation_id: string;
  action: string;
  category: string;
  priority: string;
  rationale: string;
  evidence: string[];
  expected_impact?: string | null;
  risk: string;
  confidence: number;
  recommendation_score: number;
  supporting_forecast: Record<string, any>;
  supporting_features: string[];
  trade_offs?: TradeOff | null;
  assumptions: string[];
  warnings: string[];
  requires_human_review: boolean;
  rule_id: string;
}

interface DecisionResult {
  decision_id: string;
  entity_id?: string | null;
  product_id?: string | null;
  forecast_id?: string | null;
  primary_recommendation?: RecommendationItem | null;
  recommendations: RecommendationItem[];
  context_summary: Record<string, any>;
  forecast_summary: Record<string, any>;
  explanation_summary: Record<string, any>;
  rules_evaluated: string[];
  rules_triggered: string[];
  warnings: string[];
  requires_human_review: boolean;
  audit_trail: {
    decision_confidence: number;
    recommendation_score: number;
    execution_duration_ms: number;
    primary_rule_id?: string | null;
  };
}

interface ScenarioResult {
  scenario_name: string;
  delta_summary: Record<string, any>;
  scenario_decision: DecisionResult;
}

export default function DecisionIntelligencePanel() {
  const [currentInventory, setCurrentInventory] = useState<string>("85");
  const [reorderPoint, setReorderPoint] = useState<string>("140");
  const [safetyStock, setSafetyStock] = useState<string>("40");
  const [leadTimeDays, setLeadTimeDays] = useState<string>("7");
  const [currentPrice, setCurrentPrice] = useState<string>("24.99");
  const [promotionActive, setPromotionActive] = useState<boolean>(true);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isScenarioLoading, setIsScenarioLoading] = useState<boolean>(false);
  const [result, setResult] = useState<DecisionResult | null>(null);
  const [scenarioResult, setScenarioResult] = useState<ScenarioResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load sample decision on mount
  useEffect(() => {
    fetchSample();
  }, []);

  const fetchSample = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch("http://localhost:8000/api/v1/decisions/sample");
      if (!res.ok) throw new Error(`Backend returned status ${res.status}`);
      const data: DecisionResult = await res.json();
      setResult(data);
    } catch (err: any) {
      console.warn("Could not fetch decision sample from backend:", err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunDecision = async () => {
    setIsLoading(true);
    setError(null);
    setScenarioResult(null);

    const payload = {
      entity_id: "STORE_001",
      product_id: "PROD_001",
      business_context: {
        current_inventory: currentInventory !== "" ? parseFloat(currentInventory) : null,
        reorder_point: reorderPoint !== "" ? parseFloat(reorderPoint) : null,
        safety_stock: safetyStock !== "" ? parseFloat(safetyStock) : null,
        lead_time_days: leadTimeDays !== "" ? parseInt(leadTimeDays) : null,
        current_price: currentPrice !== "" ? parseFloat(currentPrice) : null,
        promotion_active: promotionActive,
      },
    };

    try {
      const res = await fetch("http://localhost:8000/api/v1/decisions/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      const data: DecisionResult = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to execute decision intelligence analysis.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunScenario = async (deltaType: "inventory_cut" | "price_cut" | "promo_toggle") => {
    if (!result) return;
    setIsScenarioLoading(true);
    setError(null);

    let scenarioName = "Scenario Simulation";
    let inventoryDelta: number | null = null;
    let priceDeltaPercent: number | null = null;
    let promoOverride: boolean | null = null;

    if (deltaType === "inventory_cut") {
      scenarioName = "Inventory Cut (-50 Units)";
      inventoryDelta = -50.0;
    } else if (deltaType === "price_cut") {
      scenarioName = "15% Promotional Price Cut";
      priceDeltaPercent = -15.0;
    } else if (deltaType === "promo_toggle") {
      scenarioName = `Toggle Promotion to ${!promotionActive ? "ON" : "OFF"}`;
      promoOverride = !promotionActive;
    }

    const payload = {
      scenario_name: scenarioName,
      base_request: {
        entity_id: "STORE_001",
        product_id: "PROD_001",
        business_context: {
          current_inventory: currentInventory !== "" ? parseFloat(currentInventory) : null,
          reorder_point: reorderPoint !== "" ? parseFloat(reorderPoint) : null,
          safety_stock: safetyStock !== "" ? parseFloat(safetyStock) : null,
          lead_time_days: leadTimeDays !== "" ? parseInt(leadTimeDays) : null,
          current_price: currentPrice !== "" ? parseFloat(currentPrice) : null,
          promotion_active: promotionActive,
        },
      },
      inventory_delta: inventoryDelta,
      price_delta_percent: priceDeltaPercent,
      promotion_override: promoOverride,
    };

    try {
      const res = await fetch("http://localhost:8000/api/v1/decisions/scenario", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      const data: ScenarioResult = await res.json();
      setScenarioResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to execute scenario simulation.");
    } finally {
      setIsScenarioLoading(false);
    }
  };

  const getPriorityBadgeClass = (priority: string) => {
    switch (priority) {
      case "CRITICAL":
        return "bg-rose-500/20 text-rose-400 border-rose-500/40";
      case "HIGH":
        return "bg-amber-500/20 text-amber-400 border-amber-500/40";
      case "MEDIUM":
        return "bg-cyan-500/20 text-cyan-400 border-cyan-500/40";
      default:
        return "bg-slate-500/20 text-slate-300 border-slate-500/40";
    }
  };

  const applyPreset = (preset: "stockout" | "overstock" | "missing_inv") => {
    if (preset === "stockout") {
      setCurrentInventory("25");
      setReorderPoint("120");
      setSafetyStock("35");
      setLeadTimeDays("7");
      setPromotionActive(true);
    } else if (preset === "overstock") {
      setCurrentInventory("600");
      setReorderPoint("100");
      setSafetyStock("30");
      setLeadTimeDays("7");
      setPromotionActive(false);
    } else if (preset === "missing_inv") {
      setCurrentInventory("");
      setReorderPoint("");
      setSafetyStock("");
      setLeadTimeDays("7");
      setPromotionActive(false);
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-6 relative overflow-hidden">
      <div className="absolute top-0 right-0 w-80 h-80 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none"></div>

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="text-xs font-bold uppercase tracking-widest text-emerald-400 bg-emerald-950/80 px-2.5 py-0.5 rounded-md border border-emerald-800">
              Phase 7 Active
            </span>
            <span className="text-xs text-slate-400 font-mono">
              Deterministic Decision Engine · Zero LLM Dependency
            </span>
          </div>
          <h3 className="text-xl font-bold text-white mt-1.5 tracking-tight flex items-center gap-2">
            <span>💡</span> Decision Intelligence Agent
          </h3>
          <p className="text-xs text-slate-300 mt-1 max-w-3xl leading-relaxed">
            Translates forecasting uncertainty, Phase 6 XAI feature evidence, and operational business context
            into actionable business recommendations with explicit trade-offs and zero autonomous execution.
          </p>
        </div>

        {/* Action Presets */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-slate-400 font-mono">Presets:</span>
          <button
            onClick={() => applyPreset("stockout")}
            className="text-xs px-2.5 py-1 rounded bg-slate-800/80 hover:bg-slate-700 text-rose-300 border border-slate-700 transition"
          >
            Critical Stockout
          </button>
          <button
            onClick={() => applyPreset("overstock")}
            className="text-xs px-2.5 py-1 rounded bg-slate-800/80 hover:bg-slate-700 text-cyan-300 border border-slate-700 transition"
          >
            Overstock
          </button>
          <button
            onClick={() => applyPreset("missing_inv")}
            className="text-xs px-2.5 py-1 rounded bg-slate-800/80 hover:bg-slate-700 text-amber-300 border border-slate-700 transition"
          >
            Missing Inventory
          </button>
        </div>
      </div>

      {/* Interactive Business Context Controls */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 p-4 bg-slate-900/60 rounded-xl border border-slate-800 text-xs">
        <div>
          <label className="text-slate-400 block mb-1">Current Inventory</label>
          <input
            type="number"
            value={currentInventory}
            placeholder="e.g. 85 (Empty = None)"
            onChange={(e) => setCurrentInventory(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-white font-mono focus:border-emerald-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="text-slate-400 block mb-1">Reorder Point</label>
          <input
            type="number"
            value={reorderPoint}
            placeholder="e.g. 140"
            onChange={(e) => setReorderPoint(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-white font-mono focus:border-emerald-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="text-slate-400 block mb-1">Safety Stock</label>
          <input
            type="number"
            value={safetyStock}
            placeholder="e.g. 40"
            onChange={(e) => setSafetyStock(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-white font-mono focus:border-emerald-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="text-slate-400 block mb-1">Lead Time (Days)</label>
          <input
            type="number"
            value={leadTimeDays}
            placeholder="e.g. 7"
            onChange={(e) => setLeadTimeDays(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-white font-mono focus:border-emerald-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="text-slate-400 block mb-1">Unit Price ($)</label>
          <input
            type="number"
            step="0.01"
            value={currentPrice}
            placeholder="e.g. 24.99"
            onChange={(e) => setCurrentPrice(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-white font-mono focus:border-emerald-500 focus:outline-none"
          />
        </div>
        <div className="flex flex-col justify-between">
          <label className="text-slate-400 block mb-1">Promotion Active</label>
          <button
            type="button"
            onClick={() => setPromotionActive(!promotionActive)}
            className={`w-full py-1.5 px-2.5 rounded font-bold transition border ${
              promotionActive
                ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/40"
                : "bg-slate-800 text-slate-400 border-slate-700"
            }`}
          >
            {promotionActive ? "PROMO ON" : "PROMO OFF"}
          </button>
        </div>
      </div>

      {/* Execution Button */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleRunDecision}
          disabled={isLoading}
          className="px-5 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-sm shadow-lg shadow-emerald-950/40 transition disabled:opacity-50 flex items-center gap-2"
        >
          {isLoading ? (
            <>
              <span className="animate-spin inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full"></span>
              Evaluating Rules...
            </>
          ) : (
            <>
              <span>⚡</span> Run Decision Intelligence
            </>
          )}
        </button>

        <span className="text-xs text-slate-400">
          Evaluates 8 deterministic business rules against forecast trends and XAI feature attributions.
        </span>
      </div>

      {error && (
        <div className="p-3.5 rounded-lg bg-rose-950/60 border border-rose-800 text-rose-300 text-xs flex items-center gap-2">
          <span>⚠️</span> {error}
        </div>
      )}

      {/* Main Results Display */}
      {result && result.primary_recommendation && (
        <div className="space-y-6 pt-2">
          {/* Primary Recommendation Banner */}
          <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-700 relative overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
              <div className="flex items-center gap-2.5">
                <span
                  className={`text-xs font-bold px-2.5 py-0.5 rounded border uppercase ${getPriorityBadgeClass(
                    result.primary_recommendation.priority
                  )}`}
                >
                  {result.primary_recommendation.priority} PRIORITY
                </span>
                <span className="text-xs font-mono bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700">
                  {result.primary_recommendation.category}
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  Rule: {result.primary_recommendation.rule_id}
                </span>
              </div>

              {result.primary_recommendation.requires_human_review && (
                <span className="text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 px-2.5 py-0.5 rounded flex items-center gap-1.5 animate-pulse">
                  <span>👤</span> HUMAN REVIEW REQUIRED
                </span>
              )}
            </div>

            <h4 className="text-lg font-bold text-white tracking-tight">
              {result.primary_recommendation.action}
            </h4>
            <p className="text-xs text-slate-300 mt-2 leading-relaxed max-w-4xl">
              {result.primary_recommendation.rationale}
            </p>
          </div>

          {/* Key Metric Meters */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-xs text-slate-400 block font-medium">Decision Confidence</span>
              <div className="flex items-baseline gap-1 mt-1">
                <span className="text-2xl font-bold text-white font-mono">
                  {(result.primary_recommendation.confidence * 100).toFixed(1)}%
                </span>
                <span className="text-[10px] text-slate-400">Calibrated</span>
              </div>
              <span className="text-[11px] text-slate-400 mt-1 block">
                Derived from validation metrics & context penalties.
              </span>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-xs text-slate-400 block font-medium">Recommendation Score</span>
              <div className="flex items-baseline gap-1 mt-1">
                <span className="text-2xl font-bold text-emerald-400 font-mono">
                  {result.primary_recommendation.recommendation_score.toFixed(1)}
                </span>
                <span className="text-[10px] text-slate-400">/ 100</span>
              </div>
              <span className="text-[11px] text-slate-400 mt-1 block">
                Rule-based decision strength index (not a probability).
              </span>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-xs text-slate-400 block font-medium">Forecast Demand Total</span>
              <div className="flex items-baseline gap-1 mt-1">
                <span className="text-2xl font-bold text-white font-mono">
                  {result.forecast_summary?.total_demand ?? "N/A"}
                </span>
                <span className="text-[10px] text-slate-400">units</span>
              </div>
              <span className="text-[11px] text-slate-400 mt-1 block">
                Trend: {result.forecast_summary?.trend_direction ?? "flat"} (slope:{" "}
                {((result.forecast_summary?.trend_slope || 0) * 100).toFixed(1)}%)
              </span>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-xs text-slate-400 block font-medium">Days of Supply Coverage</span>
              <div className="flex items-baseline gap-1 mt-1">
                <span className="text-2xl font-bold text-white font-mono">
                  {result.context_summary?.days_of_supply !== null
                    ? `${result.context_summary?.days_of_supply}d`
                    : "Unknown"}
                </span>
                <span className="text-[10px] text-slate-400">
                  vs {result.context_summary?.lead_time_days ?? 7}d lead time
                </span>
              </div>
              <span className="text-[11px] text-slate-400 mt-1 block">
                Status: {result.context_summary?.coverage_status ?? "N/A"}
              </span>
            </div>
          </div>

          {/* Traceable Evidence & Trade-Offs Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Traceable Evidence */}
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2.5">
              <h5 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <span>🔍</span> Traceable Decision Evidence
              </h5>
              <ul className="space-y-1.5 text-xs text-slate-300">
                {result.primary_recommendation.evidence.map((ev, idx) => (
                  <li key={idx} className="flex items-start gap-2 bg-slate-950/60 p-2 rounded border border-slate-800/80">
                    <span className="text-emerald-400 mt-0.5">•</span>
                    <span>{ev}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Explicit Trade-Offs */}
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2.5">
              <h5 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <span>⚖️</span> Explicit Operational Trade-Offs
              </h5>
              {result.primary_recommendation.trade_offs ? (
                <div className="space-y-2 text-xs">
                  <div className="bg-emerald-950/40 border border-emerald-800/60 p-2.5 rounded">
                    <strong className="text-emerald-400 block mb-0.5">Primary Benefit:</strong>
                    <span className="text-slate-200">
                      {result.primary_recommendation.trade_offs.benefit}
                    </span>
                  </div>
                  <div className="bg-amber-950/40 border border-amber-800/60 p-2.5 rounded">
                    <strong className="text-amber-400 block mb-0.5">Associated Trade-Off / Cost:</strong>
                    <span className="text-slate-200">
                      {result.primary_recommendation.trade_offs.trade_off}
                    </span>
                  </div>
                  {result.primary_recommendation.trade_offs.quantified_impact && (
                    <div className="bg-slate-950 p-2 rounded border border-slate-800 text-slate-300 font-mono text-[11px]">
                      Impact: {result.primary_recommendation.trade_offs.quantified_impact}
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-xs text-slate-400">No explicit financial trade-offs flagged.</p>
              )}
            </div>
          </div>

          {/* Warnings & Assumptions */}
          {(result.warnings.length > 0 || result.primary_recommendation.assumptions.length > 0) && (
            <div className="p-3.5 rounded-xl bg-slate-900/40 border border-slate-800 text-xs space-y-1.5">
              {result.warnings.map((w, idx) => (
                <div key={idx} className="text-amber-400 flex items-center gap-1.5 font-medium">
                  <span>⚠️</span> Warning: {w}
                </div>
              ))}
              {result.primary_recommendation.assumptions.map((a, idx) => (
                <div key={idx} className="text-slate-400 flex items-center gap-1.5">
                  <span>ℹ️</span> Assumption: {a}
                </div>
              ))}
            </div>
          )}

          {/* Interactive What-If Scenario Simulator */}
          <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-700 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h5 className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
                  <span>🔮</span> Deterministic What-If Scenario Simulation
                </h5>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Simulate operational shocks without retraining forecasting models.
                </p>
              </div>

              <div className="flex items-center gap-2 flex-wrap">
                <button
                  onClick={() => handleRunScenario("inventory_cut")}
                  disabled={isScenarioLoading}
                  className="text-xs px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-rose-300 border border-slate-700 font-semibold transition disabled:opacity-50"
                >
                  -50 Inventory Shock
                </button>
                <button
                  onClick={() => handleRunScenario("price_cut")}
                  disabled={isScenarioLoading}
                  className="text-xs px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 font-semibold transition disabled:opacity-50"
                >
                  -15% Price Discount
                </button>
                <button
                  onClick={() => handleRunScenario("promo_toggle")}
                  disabled={isScenarioLoading}
                  className="text-xs px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-emerald-300 border border-slate-700 font-semibold transition disabled:opacity-50"
                >
                  Toggle Promo
                </button>
              </div>
            </div>

            {/* Scenario Comparative Output */}
            {scenarioResult && (
              <div className="p-3.5 rounded-lg bg-slate-950 border border-cyan-900/60 text-xs space-y-2 mt-2">
                <div className="flex items-center justify-between text-cyan-300 font-bold border-b border-slate-800 pb-1.5">
                  <span>Scenario: {scenarioResult.scenario_name}</span>
                  <span className="text-[11px] text-slate-400 font-mono">
                    Simulation ID: {scenarioResult.scenario_decision.decision_id}
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                  <div className="p-2.5 rounded bg-slate-900/60 border border-slate-800">
                    <strong className="text-slate-400 block mb-1">Baseline Action:</strong>
                    <div className="text-white font-medium">
                      {scenarioResult.delta_summary?.primary_action_change?.baseline_action}
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">
                      Priority: {scenarioResult.delta_summary?.primary_action_change?.baseline_priority} | Score:{" "}
                      {scenarioResult.delta_summary?.primary_action_change?.baseline_score}
                    </span>
                  </div>

                  <div className="p-2.5 rounded bg-cyan-950/30 border border-cyan-800/50">
                    <strong className="text-cyan-400 block mb-1">Scenario Recommended Action:</strong>
                    <div className="text-cyan-100 font-medium">
                      {scenarioResult.delta_summary?.primary_action_change?.scenario_action}
                    </div>
                    <span className="text-[10px] text-cyan-300 font-mono">
                      Priority: {scenarioResult.delta_summary?.primary_action_change?.scenario_priority} | Score:{" "}
                      {scenarioResult.delta_summary?.primary_action_change?.scenario_score}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
