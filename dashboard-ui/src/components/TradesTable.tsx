import type { TradeRow } from "../types";
import { formatDateTime, formatOptionalCurrency } from "../lib/format";
import { StatusBadge } from "./StatusBadge";

interface TradesTableProps {
  trades: TradeRow[];
}

function getStatusTone(status: TradeRow["status"]) {
  if (status === "Take Profit" || status === "Filled") {
    return "emerald";
  }
  if (status === "Stopped") {
    return "rose";
  }
  return "amber";
}

export function TradesTable({ trades }: TradesTableProps) {
  return (
    <section className="terminal-panel">
      <div className="mb-5">
        <p className="terminal-kicker">Execution Ledger</p>
        <h2 className="terminal-title">Historical Trades</h2>
      </div>

      {trades.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-white/10 bg-slate-950/[0.03] px-4 py-10 text-center text-sm text-slate-500 dark:bg-white/[0.03] dark:text-slate-400">
          No recent live execution records exposed by the API yet.
        </div>
      ) : (
      <div className="overflow-x-auto">
        <table className="min-w-full table-auto text-left">
          <thead className="text-xs uppercase tracking-[0.2em] text-slate-500">
            <tr>
              {["Time", "Symbol", "Side", "Entry", "Exit", "PnL", "Fee", "Slippage", "Reason", "Status"].map((head) => (
                <th key={head} className="px-3 py-3 font-medium">{head}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {trades.map((trade, index) => (
              <tr key={`${trade.symbol}-${index}`} className="border-t border-white/6 text-sm text-slate-700 dark:text-slate-200">
                <td className="px-3 py-4 font-mono">{formatDateTime(trade.time)}</td>
                <td className="px-3 py-4 font-semibold text-slate-950 dark:text-white">{trade.symbol}</td>
                <td className="px-3 py-4">
                  <StatusBadge label={trade.side} tone={trade.side === "Long" ? "emerald" : "rose"} />
                </td>
                <td className="px-3 py-4 font-mono">{formatOptionalCurrency(trade.entry)}</td>
                <td className="px-3 py-4 font-mono">{formatOptionalCurrency(trade.exit)}</td>
                <td className={`px-3 py-4 font-mono ${
                  trade.pnl === null
                    ? "text-slate-500 dark:text-slate-400"
                    : trade.pnl >= 0
                      ? "text-emerald-400"
                      : "text-rose-400"
                }`}>
                  {formatOptionalCurrency(trade.pnl)}
                </td>
                <td className="px-3 py-4 font-mono">{formatOptionalCurrency(trade.fee)}</td>
                <td className="px-3 py-4 font-mono">{formatOptionalCurrency(trade.slippage)}</td>
                <td className="max-w-[260px] px-3 py-4 text-slate-500 dark:text-slate-400">{trade.reason}</td>
                <td className="px-3 py-4">
                  <StatusBadge label={trade.status} tone={getStatusTone(trade.status)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      )}
    </section>
  );
}
