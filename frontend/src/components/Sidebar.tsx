"use client";

import React from "react";

interface NavItem {
  id: string;
  name: string;
  badge?: string;
  icon: React.ReactNode;
  active?: boolean;
}

const navItems: NavItem[] = [
  {
    id: "dashboard",
    name: "Dashboard",
    active: true,
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
      </svg>
    ),
  },
  {
    id: "dataset",
    name: "Dataset",
    badge: "Phase 2",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 7v10c0 2 1.5 3 3.5 3h9c2 0 3.5-1 3.5-3V7M4 7c0-2 1.5-3 3.5-3h9c2 0 3.5 1 3.5 3M4 7h16m-8 4v6m-4-3h8" />
      </svg>
    ),
  },
  {
    id: "forecasting",
    name: "Forecasting",
    badge: "Phase 4",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
  },
  {
    id: "explainability",
    name: "Explainability",
    badge: "Phase 6",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    id: "decisions",
    name: "Decisions",
    badge: "Phase 7",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
  },
  {
    id: "agent_activity",
    name: "Agent Activity",
    badge: "Phase 8",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
  },
];

export default function Sidebar() {
  return (
    <aside className="w-64 glass-panel border-r border-slate-800/80 flex flex-col justify-between p-5 shrink-0 min-h-screen">
      <div>
        {/* Brand Logo & Title */}
        <div className="flex items-center space-x-3 mb-8 px-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-purple-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 font-bold text-white text-xl">
            GB
          </div>
          <div>
            <h1 className="font-bold text-lg text-white tracking-tight">GlassBox-BI</h1>
            <p className="text-[11px] text-cyan-400 font-medium tracking-wide">EXPLAINABLE AI PLATFORM</p>
          </div>
        </div>

        {/* Section Title */}
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-3 mb-3">
          Platform Navigation
        </p>

        {/* Navigation Items */}
        <nav className="space-y-1.5">
          {navItems.map((item) => (
            <div
              key={item.id}
              className={`flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 cursor-pointer ${
                item.active
                  ? "bg-gradient-to-r from-cyan-500/20 to-blue-600/10 text-cyan-300 border border-cyan-500/30 shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <div className="flex items-center space-x-3">
                <span className={item.active ? "text-cyan-400" : "text-slate-400"}>
                  {item.icon}
                </span>
                <span>{item.name}</span>
              </div>
              {item.badge && (
                <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-slate-800/90 text-slate-400 border border-slate-700/60">
                  {item.badge}
                </span>
              )}
            </div>
          ))}
        </nav>
      </div>

      {/* Footer Info */}
      <div className="pt-4 border-t border-slate-800/80 px-2 text-xs text-slate-400">
        <div className="flex items-center justify-between">
          <span className="text-slate-300 font-medium">Architecture</span>
          <span className="px-2 py-0.5 text-[10px] font-semibold rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
            Phase 1 Active
          </span>
        </div>
        <p className="mt-2 text-[11px] text-slate-400">GlassBox-BI v0.2.0-alpha</p>
      </div>
    </aside>
  );
}
