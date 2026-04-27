import type { CSSProperties } from "react";
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
  const gaugeStyle = {
    "--score": `${signal.score}%`,
  } as CSSProperties;

  return (
    <section className="terminal-panel">
      <div className="flex items-start justify-between gap-4">
        <h2 className="panel-title">策略信号概览</h2>
        <StatusBadge label={getSignalDirectionLabel(signal.direction)} tone={getSignalTone(signal.direction)} pulse />
      </div>

      <div className="signal-ring-wrap mt-6">
        <div className="signal-ring" style={gaugeStyle}>
          <div className="signal-ring-core">
            <span className="signal-ring-score">{signal.score}</span>
            <span className="signal-ring-label">/100</span>
          </div>
        </div>
        <div className="flex-1">
          <p className="text-sm text-slate-400">强势信号</p>
          <p className="mt-2 text-2xl font-semibold text-slate-50">{getSignalDirectionLabel(signal.direction)}</p>
          <p className="mt-3 text-sm leading-7 text-slate-400">
            当前评分基于模型主信号、资金流与趋势确认因子综合推导，适合快速判断是否进入执行窗口。
          </p>
        </div>
      </div>

      <div className="mt-5 flex flex-wrap gap-2">
        {signal.sources.map((source) => (
          <span key={source} className="panel-pill">
            {getSignalSourceLabel(source)}
          </span>
        ))}
      </div>

      <div className="mt-6 grid gap-3">
        <div className="panel-stat-card">
          <p>最近触发</p>
          <strong>{formatDateTime(signal.lastTriggeredAt)}</strong>
        </div>
        <div className="panel-stat-card">
          <p>距离下次运行</p>
          <strong>{formatCountdown(signal.nextRunAt, now)}</strong>
        </div>
      </div>
    </section>
  );
}
