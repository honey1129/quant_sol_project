import { formatClock, formatDateTime } from "../lib/format";
import { StatusBadge } from "./StatusBadge";
import type { DataSource, ExchangeName, StrategyStatus, ThemeMode } from "../types";

interface TopNavProps {
  productName: string;
  strategyName: string;
  exchange: ExchangeName;
  status: StrategyStatus;
  updatedAt: string;
  dataSource: DataSource;
  now: Date;
  theme: ThemeMode;
  onThemeToggle: () => void;
}

function getStatusTone(status: StrategyStatus) {
  if (status === "Running") {
    return "emerald";
  }
  if (status === "Paused") {
    return "amber";
  }
  return "rose";
}

function getDataSourceTone(dataSource: DataSource) {
  if (dataSource === "live") {
    return "emerald";
  }
  if (dataSource === "hybrid") {
    return "amber";
  }
  return "violet";
}

export function TopNav({
  productName,
  strategyName,
  exchange,
  status,
  updatedAt,
  dataSource,
  now,
  theme,
  onThemeToggle,
}: TopNavProps) {
  return (
    <header className="sticky top-4 z-30 mb-6 rounded-3xl border border-white/10 bg-white/75 px-5 py-4 shadow-terminal backdrop-blur-xl dark:bg-slate-950/80">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-5">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-sky-400">{productName}</p>
            <h1 className="mt-1 text-2xl font-semibold text-slate-950 dark:text-white">{strategyName}</h1>
          </div>
          <div className="flex flex-wrap gap-2">
            <StatusBadge label={exchange} tone="sky" />
            <StatusBadge label={status} tone={getStatusTone(status)} />
            <StatusBadge label={dataSource === "live" ? "Live API" : dataSource === "hybrid" ? "Hybrid Data" : "Mock Fallback"} tone={getDataSourceTone(dataSource)} />
          </div>
        </div>

        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <div className="rounded-2xl border border-white/10 bg-slate-950/[0.03] px-4 py-2 text-right dark:bg-white/5">
            <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500 dark:text-slate-400">Current Time</p>
            <p className="mt-1 font-mono text-sm font-medium text-slate-900 dark:text-slate-100">{formatClock(now)}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/[0.03] px-4 py-2 text-right dark:bg-white/5">
            <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500 dark:text-slate-400">Last Update</p>
            <p className="mt-1 font-mono text-sm font-medium text-slate-900 dark:text-slate-100">{formatDateTime(updatedAt)}</p>
          </div>
          <button
            type="button"
            onClick={onThemeToggle}
            className="rounded-2xl border border-white/10 bg-slate-950/[0.03] px-4 py-3 text-sm font-semibold text-slate-900 transition hover:border-sky-400/40 hover:text-sky-500 dark:bg-white/5 dark:text-white dark:hover:text-sky-300"
          >
            Theme: {theme === "dark" ? "Dark" : "Light"}
          </button>
        </div>
      </div>
    </header>
  );
}
