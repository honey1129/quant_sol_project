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
          <p className="terminal-kicker">Config Surface</p>
          <h2 className="terminal-title">Strategy Parameters</h2>
        </div>
        <div className="text-right text-xs text-slate-500">
          <p>Live snapshot with local edit buffer</p>
          <p>{savedAt ? `Last saved ${formatClock(savedAt)} UTC+8` : "Not saved yet"}</p>
        </div>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <label className="space-y-2">
          <span className="terminal-label">Timeframe</span>
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
          <span className="terminal-label">MA Period</span>
          <input
            className="terminal-input"
            type="number"
            value={params.maPeriod}
            onChange={(event) => onChange("maPeriod", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="terminal-label">RSI Period</span>
          <input
            className="terminal-input"
            type="number"
            value={params.rsiPeriod}
            onChange={(event) => onChange("rsiPeriod", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="terminal-label">ATR Stop Multiplier</span>
          <input
            className="terminal-input"
            type="number"
            step="0.1"
            value={params.atrMultiplier}
            onChange={(event) => onChange("atrMultiplier", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="terminal-label">Stop Loss %</span>
          <input
            className="terminal-input"
            type="number"
            step="0.1"
            value={params.stopLossPct}
            onChange={(event) => onChange("stopLossPct", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="terminal-label">Take Profit %</span>
          <input
            className="terminal-input"
            type="number"
            step="0.1"
            value={params.takeProfitPct}
            onChange={(event) => onChange("takeProfitPct", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="terminal-label">Position Size (Target Capital)</span>
          <input
            className="terminal-input"
            type="number"
            step="0.1"
            value={params.positionSizePct}
            onChange={(event) => onChange("positionSizePct", Number(event.target.value))}
          />
        </label>

        <label className="space-y-2">
          <span className="terminal-label">Max Leverage</span>
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
          {saving ? "Saving..." : "Save Parameters"}
        </button>
        <button
          type="button"
          onClick={onRestart}
          disabled={saving || restarting}
          className="rounded-2xl border border-amber-400/20 bg-amber-500/10 px-4 py-3 text-sm font-semibold text-amber-300 transition hover:border-amber-300/40 hover:bg-amber-500/15 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {restarting ? "Restarting..." : "Restart Strategy"}
        </button>
        <button type="button" onClick={onReset} disabled={saving} className="terminal-button-secondary disabled:cursor-not-allowed disabled:opacity-60">
          Reset
        </button>
      </div>

      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
        Saving writes the latest values into <span className="font-mono">.env</span>. Restart the strategy daemon to apply them to the live trading process.
      </p>
    </section>
  );
}
