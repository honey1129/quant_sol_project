import { formatClock } from "../lib/format";
import type { StrategyParams } from "../types";

interface ParamsPanelProps {
  params: StrategyParams;
  savedAt: Date | null;
  saving?: boolean;
  restarting?: boolean;
  onChange: <K extends keyof StrategyParams>(key: K, value: StrategyParams[K]) => void;
  onSave: () => void | Promise<void>;
  onRestart: () => void | Promise<void>;
  onReset: () => void;
}

export function ParamsPanel({
  params,
  savedAt,
  saving = false,
  restarting = false,
  onChange,
  onSave,
  onRestart,
  onReset,
}: ParamsPanelProps) {
  return (
    <section className="terminal-panel">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="terminal-kicker">参数面板</p>
          <h2 className="terminal-title">策略参数</h2>
        </div>
        <div className="text-right text-xs text-slate-500">
          <p>实时快照，可在本地编辑后保存</p>
          <p>{savedAt ? `上次保存 ${formatClock(savedAt)} UTC+8` : "尚未保存"}</p>
        </div>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <label className="space-y-2">
          <span className="terminal-label">周期</span>
          <select
            value={params.timeframe}
            onChange={(event) => onChange("timeframe", event.target.value)}
            className="terminal-input"
          >
            {["5m", "15m", "1H", "4H"].map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
        </label>

        <label className="space-y-2">
          <span className="terminal-label">均线周期</span>
          <input
            className="terminal-input"
            type="number"
            value={params.maPeriod}
            onChange={(event) => onChange("maPeriod", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="terminal-label">RSI 周期</span>
          <input
            className="terminal-input"
            type="number"
            value={params.rsiPeriod}
            onChange={(event) => onChange("rsiPeriod", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="terminal-label">ATR 止损倍数</span>
          <input
            className="terminal-input"
            type="number"
            step="0.1"
            value={params.atrMultiplier}
            onChange={(event) => onChange("atrMultiplier", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="terminal-label">止损比例 %</span>
          <input
            className="terminal-input"
            type="number"
            step="0.1"
            value={params.stopLossPct}
            onChange={(event) => onChange("stopLossPct", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="terminal-label">止盈比例 %</span>
          <input
            className="terminal-input"
            type="number"
            step="0.1"
            value={params.takeProfitPct}
            onChange={(event) => onChange("takeProfitPct", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="terminal-label">仓位比例（目标资金）</span>
          <input
            className="terminal-input"
            type="number"
            step="0.1"
            value={params.positionSizePct}
            onChange={(event) => onChange("positionSizePct", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="terminal-label">最大杠杆</span>
          <input
            className="terminal-input"
            type="number"
            step="1"
            value={params.maxLeverage}
            onChange={(event) => onChange("maxLeverage", Number(event.target.value))}
          />
        </label>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        <button type="button" onClick={onSave} disabled={saving} className="terminal-button-primary disabled:cursor-not-allowed disabled:opacity-60">
          {saving ? "保存中..." : "保存参数"}
        </button>
        <button
          type="button"
          onClick={onRestart}
          disabled={saving || restarting}
          className="rounded-2xl border border-amber-400/20 bg-amber-500/10 px-4 py-3 text-sm font-semibold text-amber-300 transition hover:border-amber-300/40 hover:bg-amber-500/15 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {restarting ? "重启中..." : "重启策略"}
        </button>
        <button type="button" onClick={onReset} disabled={saving} className="terminal-button-secondary disabled:cursor-not-allowed disabled:opacity-60">
          重置
        </button>
      </div>

      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
        保存会把最新参数写入 <span className="font-mono">.env</span>。要让实盘进程生效，还需要重启策略守护进程。
      </p>
    </section>
  );
}
