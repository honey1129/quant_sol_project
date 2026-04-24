import type { PositionRow } from "../types";
import { formatOptionalCurrency } from "../lib/format";
import { StatusBadge } from "./StatusBadge";

interface PositionsTableProps {
  positions: PositionRow[];
}

export function PositionsTable({ positions }: PositionsTableProps) {
  return (
    <section className="terminal-panel">
      <div className="mb-5">
        <p className="terminal-kicker">Exposure</p>
        <h2 className="terminal-title">Current Positions</h2>
      </div>

      {positions.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-white/10 bg-slate-950/[0.03] px-4 py-10 text-center text-sm text-slate-500 dark:bg-white/[0.03] dark:text-slate-400">
          No active live positions right now.
        </div>
      ) : (
      <div className="overflow-x-auto">
        <table className="min-w-full table-auto text-left">
          <thead className="text-xs uppercase tracking-[0.2em] text-slate-500">
            <tr>
              {["Symbol", "Direction", "Entry Price", "Current Price", "Position Size", "Leverage", "Unrealized PnL", "Stop Loss", "Take Profit", "Holding Time"].map((head) => (
                <th key={head} className="px-3 py-3 font-medium">{head}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {positions.map((position) => (
              <tr key={position.symbol} className="border-t border-white/6 text-sm text-slate-700 dark:text-slate-200">
                <td className="px-3 py-4 font-semibold text-slate-950 dark:text-white">{position.symbol}</td>
                <td className="px-3 py-4">
                  <StatusBadge label={position.direction} tone={position.direction === "Long" ? "emerald" : "rose"} />
                </td>
                <td className="px-3 py-4 font-mono">{formatOptionalCurrency(position.entryPrice)}</td>
                <td className="px-3 py-4 font-mono">{formatOptionalCurrency(position.currentPrice)}</td>
                <td className="px-3 py-4">{position.positionSize}</td>
                <td className="px-3 py-4">{position.leverage}</td>
                <td className={`px-3 py-4 font-mono ${
                  position.unrealizedPnl === null
                    ? "text-slate-500 dark:text-slate-400"
                    : position.unrealizedPnl >= 0
                      ? "text-emerald-400"
                      : "text-rose-400"
                }`}>
                  {formatOptionalCurrency(position.unrealizedPnl)}
                </td>
                <td className="px-3 py-4 font-mono">{formatOptionalCurrency(position.stopLoss)}</td>
                <td className="px-3 py-4 font-mono">{formatOptionalCurrency(position.takeProfit)}</td>
                <td className="px-3 py-4">{position.holdingTime}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      )}
    </section>
  );
}
