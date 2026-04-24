import type { RiskSnapshot } from "../types";
import { formatNumber, formatPercent } from "../lib/format";
import { getConnectionStatusLabel } from "../lib/uiText";
import { StatusBadge } from "./StatusBadge";

interface RiskPanelProps {
  risk: RiskSnapshot;
}

function toneFromConnection(status: RiskSnapshot["apiStatus"] | RiskSnapshot["wsStatus"]) {
  if (status === "Connected") {
    return "emerald";
  }
  if (status === "Lagging" || status === "Degraded") {
    return "amber";
  }
  return "rose";
}

export function RiskPanel({ risk }: RiskPanelProps) {
  return (
    <section className="terminal-panel">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="terminal-kicker">风控台</p>
          <h2 className="terminal-title">风控与连接状态</h2>
        </div>
        <StatusBadge label={risk.riskTriggered ? "已触发" : "正常"} tone={risk.riskTriggered ? "rose" : "emerald"} />
      </div>

      <div className="mt-6 space-y-4">
        <div>
          <div className="mb-2 flex items-center justify-between text-sm text-slate-400">
            <span>保证金使用率</span>
            <span className="font-mono text-slate-950 dark:text-white">{formatPercent(risk.marginUsagePct, 1, false)}</span>
          </div>
          <div className="h-2.5 overflow-hidden rounded-full bg-slate-300 dark:bg-slate-800">
            <div className="h-full rounded-full bg-gradient-to-r from-sky-500 to-cyan-400" style={{ width: `${risk.marginUsagePct}%` }} />
          </div>
        </div>

        <div>
          <div className="mb-2 flex items-center justify-between text-sm text-slate-400">
            <span>当日亏损阈值消耗</span>
            <span className="font-mono text-slate-950 dark:text-white">
              {formatPercent(risk.dailyLossUsedPct, 1, false)} / {formatPercent(risk.dailyLossLimitPct, 1, false)}
            </span>
          </div>
          <div className="h-2.5 overflow-hidden rounded-full bg-slate-300 dark:bg-slate-800">
            <div className="h-full rounded-full bg-gradient-to-r from-amber-500 to-rose-500" style={{ width: `${(risk.dailyLossUsedPct / risk.dailyLossLimitPct) * 100}%` }} />
          </div>
        </div>
      </div>

      <div className="mt-6 grid gap-3 md:grid-cols-2">
        <div className="rounded-2xl border border-white/10 bg-slate-950/[0.03] p-4 dark:bg-white/[0.03]">
          <p className="text-xs tracking-[0.14em] text-slate-500">当前杠杆</p>
          <p className="mt-2 font-mono text-lg text-slate-950 dark:text-white">{formatNumber(risk.currentLeverage, 1)}x</p>
        </div>
        <div className="rounded-2xl border border-white/10 bg-slate-950/[0.03] p-4 dark:bg-white/[0.03]">
          <p className="text-xs tracking-[0.14em] text-slate-500">单笔最大亏损</p>
          <p className="mt-2 font-mono text-lg text-slate-950 dark:text-white">{formatPercent(risk.maxLossPerTradePct, 1, false)}</p>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap gap-2">
        <StatusBadge label={`API ${getConnectionStatusLabel(risk.apiStatus)}`} tone={toneFromConnection(risk.apiStatus)} />
        <StatusBadge label={`WS ${getConnectionStatusLabel(risk.wsStatus)}`} tone={toneFromConnection(risk.wsStatus)} />
      </div>
    </section>
  );
}
