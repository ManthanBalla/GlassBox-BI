"use client";

import React, { useEffect, useState, useCallback } from "react";

interface HealthData {
  status: string;
  app_name: string;
  version: string;
  phase: string;
  timestamp: string;
}

type ConnectionState = "CHECKING" | "CONNECTED" | "UNAVAILABLE";

export default function BackendStatus() {
  const [connectionState, setConnectionState] = useState<ConnectionState>("CHECKING");
  const [healthData, setHealthData] = useState<HealthData | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [lastChecked, setLastChecked] = useState<string | null>(null);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  const checkHealth = useCallback(async () => {
    setConnectionState("CHECKING");
    const startTime = performance.now();

    try {
      const response = await fetch(`${apiUrl}/health`, {
        method: "GET",
        headers: {
          "Accept": "application/json",
        },
        cache: "no-store",
      });

      const endTime = performance.now();
      setLatencyMs(Math.round(endTime - startTime));

      if (response.ok) {
        const data: HealthData = await response.json();
        setHealthData(data);
        setConnectionState("CONNECTED");
      } else {
        setHealthData(null);
        setConnectionState("UNAVAILABLE");
      }
    } catch {
      // Graceful error handling: never expose raw stack traces or internal backend errors
      setHealthData(null);
      setConnectionState("UNAVAILABLE");
    } finally {
      setLastChecked(new Date().toLocaleTimeString());
    }
  }, [apiUrl]);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  return (
    <div className="glass-panel glass-panel-hover rounded-2xl p-6 mb-8 border border-slate-800">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <h3 className="text-lg font-bold text-white tracking-tight">
              Backend Communication Service
            </h3>
            
            {/* Status Pill */}
            {connectionState === "CHECKING" && (
              <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
                <span className="w-2 h-2 mr-2 rounded-full bg-amber-400 animate-ping"></span>
                Checking Connection...
              </span>
            )}
            {connectionState === "CONNECTED" && (
              <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                <span className="w-2 h-2 mr-2 rounded-full bg-emerald-400 animate-pulse"></span>
                Backend Status: Connected
              </span>
            )}
            {connectionState === "UNAVAILABLE" && (
              <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/30">
                <span className="w-2 h-2 mr-2 rounded-full bg-rose-400"></span>
                Backend Status: Unavailable
              </span>
            )}
          </div>

          <p className="text-xs text-slate-400 mt-1.5">
            Connecting to FastAPI REST API via endpoint:{" "}
            <code className="px-1.5 py-0.5 rounded bg-slate-900 text-cyan-300 font-mono text-[11px] border border-slate-800">
              {apiUrl}/health
            </code>
          </p>
        </div>

        {/* Retry Button */}
        <button
          onClick={checkHealth}
          disabled={connectionState === "CHECKING"}
          className="px-4 py-2 text-xs font-semibold text-white bg-slate-800 hover:bg-slate-700 disabled:opacity-50 transition-colors rounded-xl border border-slate-700 flex items-center justify-center space-x-2 shrink-0 self-start sm:self-auto cursor-pointer"
        >
          <svg
            className={`w-3.5 h-3.5 ${connectionState === "CHECKING" ? "animate-spin" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
          <span>Refresh Status</span>
        </button>
      </div>

      {/* Diagnostics / Connection Details */}
      <div className="mt-5 pt-4 border-t border-slate-800/80">
        {connectionState === "CONNECTED" && healthData && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
            <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800/60">
              <span className="text-slate-400 block text-[11px]">Service Name</span>
              <span className="font-semibold text-slate-200 mt-0.5 block">{healthData.app_name}</span>
            </div>
            <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800/60">
              <span className="text-slate-400 block text-[11px]">API Version</span>
              <span className="font-semibold text-cyan-300 mt-0.5 block">v{healthData.version}</span>
            </div>
            <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800/60">
              <span className="text-slate-400 block text-[11px]">Roundtrip Latency</span>
              <span className="font-semibold text-emerald-400 mt-0.5 block">{latencyMs} ms</span>
            </div>
            <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800/60">
              <span className="text-slate-400 block text-[11px]">Last Verified</span>
              <span className="font-semibold text-slate-300 mt-0.5 block">{lastChecked}</span>
            </div>
          </div>
        )}

        {connectionState === "UNAVAILABLE" && (
          <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 flex items-start space-x-3">
            <svg className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div>
              <p className="font-semibold">Unable to establish connection to backend service.</p>
              <p className="text-rose-400/80 mt-1">
                Ensure the FastAPI backend is running via{" "}
                <code className="bg-rose-950 px-1.5 py-0.5 rounded text-rose-200">
                  uvicorn backend.app.main:app --port 8000
                </code>.
              </p>
            </div>
          </div>
        )}

        {connectionState === "CHECKING" && (
          <div className="text-xs text-slate-400 italic">Testing connectivity to backend server...</div>
        )}
      </div>
    </div>
  );
}
