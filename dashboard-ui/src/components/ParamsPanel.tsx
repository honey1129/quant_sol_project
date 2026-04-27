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
        <h2 className="panel-title">策略参数面板</h2>
        <div className="text-right text-xs text-slate-500">
          <p>支持在线调整</p>
          <p>{savedAt ? `上次保存 ${formatClock(savedAt)} UTC+8` : "尚未保存"}</p>
        </div>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <label className="space-y-2">
          <span className="panel-field-label">周期</span>
          <select
            value={params.timeframe}
            onChange={(event) => onChange("timeframe", event.target.value)}
            className="panel-input"
          >
            {["5m", "15m", "1H", "4H"].map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
        </label>

        <label className="space-y-2">
          <span className="panel-field-label">均线周期</span>
          <input
            className="panel-input"
            type="number"
            value={params.maPeriod}
            onChange={(event) => onChange("maPeriod", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="panel-field-label">RSI 周期</span>
          <input
            className="panel-input"
            type="number"
            value={params.rsiPeriod}
            onChange={(event) => onChange("rsiPeriod", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="panel-field-label">ATR 止损倍数</span>
          <input
            className="panel-input"
            type="number"
            step="0.1"
            value={params.atrMultiplier}
            onChange={(event) => onChange("atrMultiplier", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="panel-field-label">止损比例 %</span>
          <input
            className="panel-input"
            type="number"
            step="0.1"
            value={params.stopLossPct}
            onChange={(event) => onChange("stopLossPct", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="panel-field-label">止盈比例 %</span>
          <input
            className="panel-input"
            type="number"
            step="0.1"
            value={params.takeProfitPct}
            onChange={(event) => onChange("takeProfitPct", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="panel-field-label">仓位比例</span>
          <input
            className="panel-input"
            type="number"
            step="0.1"
            value={params.positionSizePct}
            onChange={(event) => onChange("positionSizePct", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="panel-field-label">最大杠杆</span>
          <input
            className="panel-input"
            type="number"
            step="1"
            value={params.maxLeverage}
            onChange={(event) => onChange("maxLeverage", Number(event.target.value))}
          />
        </label>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        <button type="button" onClick={onSave} disabled={saving} className="panel-button-primary disabled:cursor-not-allowed disabled:opacity-60">
          {saving ? "保存中..." : "保存参数"}
        </button>
        <button
          type="button"
          onClick={onRestart}
          disabled={saving || restarting}
          className="panel-button-warning disabled:cursor-not-allowed disabled:opacity-60"
        >
          {restarting ? "重启中..." : "重启策略"}
        </button>
        <button type="button" onClick={onReset} disabled={saving} className="panel-button-secondary disabled:cursor-not-allowed disabled:opacity-60">
          重置
        </button>
      </div>

      <p className="mt-4 text-xs leading-6 text-slate-500">
        参数写入 <span className="font-mono text-slate-300">.env</span> 后，记得重载实盘进程与 dashboard，确保新配置生效。
      </p>
    </section>
  );
}
