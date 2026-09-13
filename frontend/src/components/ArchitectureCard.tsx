"use client";

import React from "react";

interface CardProps {
  title: string;
  phase: string;
  description: string;
  icon: string;
  accent: "cyan" | "blue" | "purple" | "emerald";
}

const accentMap = {
  cyan: "from-cyan-500/10 to-transparent border-cyan-500/20 text-cyan-400",
  blue: "from-blue-500/10 to-transparent border-blue-500/20 text-blue-400",
  purple: "from-purple-500/10 to-transparent border-purple-500/20 text-purple-400",
  emerald: "from-emerald-500/10 to-transparent border-emerald-500/20 text-emerald-400",
};

export default function ArchitectureCard({ title, phase, description, icon, accent }: CardProps) {
  return (
    <div className={`glass-panel glass-panel-hover rounded-2xl p-5 border bg-gradient-to-b ${accentMap[accent]} flex flex-col justify-between`}>
      <div>
        <div className="flex items-center justify-between mb-3">
          <span className="text-2xl">{icon}</span>
          <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-slate-900/80 border border-slate-800 text-slate-300">
            {phase}
          </span>
        </div>
        <h4 className="font-bold text-slate-100 text-base mb-1.5">{title}</h4>
        <p className="text-xs text-slate-400 leading-relaxed">{description}</p>
      </div>
      <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-400">
        <span>Contract Defined</span>
        <span className="text-cyan-400">Pydantic v2</span>
      </div>
    </div>
  );
}
