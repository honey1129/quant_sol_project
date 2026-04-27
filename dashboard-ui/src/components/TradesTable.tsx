import type { TradeRow } from "../types";
import { formatDateTime, formatOptionalCurrency } from "../lib/format";
import { getSignalDirectionLabel, getTradeStatusLabel } from "../lib/uiText";
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
      <div className="mb-5 flex items-center justify-between">
        <h2 className="panel-title">执行记录</h2>
        <span className="panel-chip">{trades.length} 笔</span>
      </div>

      {trades.length === 0 ? (
        <div className="empty-state-panel">
          暂无最近成交记录。
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="dashboard-table">
            <thead>
              <tr>
                <th>时间</th>
                <th>标的</th>
                <th>方向</th>
                <th className="text-right">开仓</th>
                <th className="text-right">平仓</th>
                <th className="text-right">盈亏</th>
                <th>原因</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((trade, index) => (
                <tr key={`${trade.symbol}-${index}`}>
                  <td className="font-mono text-slate-300">{formatDateTime(trade.time)}</td>
                  <td className="font-semibold text-slate-50">{trade.symbol}</td>
                  <td>
                    <StatusBadge label={getSignalDirectionLabel(trade.side)} tone={trade.side === "Long" ? "emerald" : "rose"} />
                  </td>
                  <td className="text-right font-mono">{formatOptionalCurrency(trade.entry)}</td>
                  <td className="text-right font-mono">{formatOptionalCurrency(trade.exit)}</td>
                  <td className={`text-right font-mono ${trade.pnl !== null && trade.pnl >= 0 ? "text-up" : "text-down"}`}>
                    {formatOptionalCurrency(trade.pnl)}
                  </td>
                  <td className="max-w-[280px] text-slate-400">{trade.reason}</td>
                  <td>
                    <StatusBadge label={getTradeStatusLabel(trade.status)} tone={getStatusTone(trade.status)} />
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
