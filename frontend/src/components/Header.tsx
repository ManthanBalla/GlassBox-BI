"use client";

import React from "react";

export default function Header() {
  return (
    <header className="glass-panel border-b border-slate-800/80 px-8 py-5 flex items-center justify-between">
      <div>
        <div className="flex items-center space-x-3">
          <h2 className="text-2xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-white via-slate-100 to-cyan-300">
            GlassBox-BI
          </h2>
          <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            Application Skeleton
          </span>
        </div>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          Multi-Agent Explainable AI Framework for Business Analytics, Forecasting, and Decision Intelligence
        </p>
      </div>

      <div className="flex items-center space-x-4">
        <div className="hidden sm:flex items-center space-x-2 bg-slate-900/80 border border-slate-800 px-3.5 py-1.5 rounded-lg text-xs text-slate-300">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>FastAPI + Next.js</span>
        </div>
      </div>
    </header>
  );
}
