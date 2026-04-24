import type { StrategySignal } from "../types";
import { formatCountdown, formatDateTime } from "../lib/format";
import { getSignalDirectionLabel, getSignalSourceLabel } from "../lib/uiText";
import { StatusBadge } from "./StatusBadge";

interface SignalPanelProps {
  signal: StrategySignal;
  now: Date;
}

function getSignalTone(direction: StrategySignal["direction"]) {
  if (direction === "Long") {
    return "emerald";
  }
  if (direction === "Short") {
    return "rose";
  }
  return "amber";
}

export function SignalPanel({ signal, now }: SignalPanelProps) {
  return (
    <section className="terminal-panel">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="terminal-kicker">信号面板</p>
          <h2 className="terminal-title">策略信号概览</h2>
        </div>
        <StatusBadge label={getSignalDirectionLabel(signal.direction)} tone={getSignalTone(signal.direction)} />
      </div>

      <div className="mt-6 rounded-2xl border border-white/10 bg-slate-950/[0.03] p-5 dark:bg-white/[0.03]">
        <div className="flex items-end justify-between gap-4">
          <div>
            <p className="text-sm text-slate-400">信号强度</p>
            <p className="mt-2 font-mono text-4xl font-semibold text-slate-950 dark:text-white">{signal.score}</p>
          </div>
          <p className="text-xs tracking-[0.14em] text-slate-500">评分 / 100</p>
        </div>

        <div className="mt-4 h-3 overflow-hidden rounded-full bg-slate-300 dark:bg-slate-800/90">
          <div
            className="h-full rounded-full bg-gradient-to-r from-sky-500 via-cyan-400 to-emerald-400"
            style={{ width: `${signal.score}%` }}
          />
        </div>
      </div>

      <div className="mt-5 flex flex-wrap gap-2">
        {signal.sources.map((source) => (
          <span
            key={source}
            className="rounded-full border border-sky-400/20 bg-sky-500/10 px-3 py-1 text-xs font-medium text-sky-300"
          >
            {getSignalSourceLabel(source)}
          </span>
        ))}
      </div>

      <div className="mt-6 grid gap-3 md:grid-cols-2">
        <div className="rounded-2xl border border-white/10 bg-slate-950/[0.03] p-4 dark:bg-white/[0.03]">
          <p className="text-xs tracking-[0.14em] text-slate-500">最近触发</p>
          <p className="mt-2 font-mono text-sm text-slate-900 dark:text-slate-100">{formatDateTime(signal.lastTriggeredAt)}</p>
        </div>
        <div className="rounded-2xl border border-white/10 bg-slate-950/[0.03] p-4 dark:bg-white/[0.03]">
          <p className="text-xs tracking-[0.14em] text-slate-500">距离下次运行</p>
          <p className="mt-2 font-mono text-sm text-slate-900 dark:text-slate-100">{formatCountdown(signal.nextRunAt, now)}</p>
        </div>
      </div>
    </section>
  );
}
