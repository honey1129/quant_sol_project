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

function safeBarWidth(value: number | null | undefined, fallback = 0) {
  const numeric = Number.isFinite(Number(value)) ? Number(value) : fallback;
  return `${Math.max(0, Math.min(100, numeric))}%`;
}

export function RiskPanel({ risk }: RiskPanelProps) {
  const dailyLossLimit = Number.isFinite(Number(risk.dailyLossLimitPct)) ? Number(risk.dailyLossLimitPct) : 10;
  const dailyLossWidth = dailyLossLimit > 0 ? (risk.dailyLossUsedPct / dailyLossLimit) * 100 : 0;

  return (
    <section className="terminal-panel">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="panel-kicker">风险控制</p>
          <h2 className="panel-title">风控与连接状态</h2>
        </div>
        <StatusBadge label={risk.riskTriggered ? "已触发" : "正常"} tone={risk.riskTriggered ? "rose" : "emerald"} />
      </div>

      <div className="mt-6 space-y-4">
        <div className="risk-meter">
          <div className="risk-meter-head">
            <span>风险暴露</span>
            <strong>{formatPercent(risk.marginUsagePct, 1, false)}</strong>
          </div>
          <div className="risk-meter-track">
            <div className="risk-meter-fill risk-meter-fill-amber" style={{ width: safeBarWidth(risk.marginUsagePct) }} />
          </div>
        </div>

        <div className="risk-meter">
          <div className="risk-meter-head">
            <span>日内亏损使用率</span>
            <strong>{formatPercent(risk.dailyLossUsedPct, 1, false)}</strong>
          </div>
          <div className="risk-meter-track">
            <div className="risk-meter-fill risk-meter-fill-violet" style={{ width: safeBarWidth(dailyLossWidth) }} />
          </div>
        </div>

        <div className="risk-meter">
          <div className="risk-meter-head">
            <span>单笔风险阈值</span>
            <strong>{formatPercent(risk.maxLossPerTradePct, 1, false)}</strong>
          </div>
          <div className="risk-meter-track">
            <div className="risk-meter-fill risk-meter-fill-emerald" style={{ width: safeBarWidth(risk.maxLossPerTradePct) }} />
          </div>
        </div>
      </div>

      <div className="mt-6 grid gap-3 md:grid-cols-2">
        <div className="panel-stat-card">
          <p>当前杠杆</p>
          <strong>{formatNumber(risk.currentLeverage, 1)}x</strong>
        </div>
        <div className="panel-stat-card">
          <p>连接状态</p>
          <strong>{getConnectionStatusLabel(risk.apiStatus)}</strong>
        </div>
      </div>

      <div className="mt-5 flex flex-wrap gap-2">
        <StatusBadge label={`API ${getConnectionStatusLabel(risk.apiStatus)}`} tone={toneFromConnection(risk.apiStatus)} />
        <StatusBadge label={`WS ${getConnectionStatusLabel(risk.wsStatus)}`} tone={toneFromConnection(risk.wsStatus)} />
      </div>
    </section>
  );
}
