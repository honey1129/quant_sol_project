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
        <div>
          <p className="panel-kicker">最近成交</p>
          <h2 className="panel-title">执行记录</h2>
        </div>
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
                {["时间", "标的", "方向", "开仓", "平仓", "盈亏", "原因", "状态"].map((head) => (
                  <th key={head}>{head}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {trades.map((trade, index) => (
                <tr key={`${trade.symbol}-${index}`}>
                  <td className="font-mono">{formatDateTime(trade.time)}</td>
                  <td className="font-semibold text-slate-50">{trade.symbol}</td>
                  <td>
                    <StatusBadge label={getSignalDirectionLabel(trade.side)} tone={trade.side === "Long" ? "emerald" : "rose"} />
                  </td>
                  <td className="font-mono">{formatOptionalCurrency(trade.entry)}</td>
                  <td className="font-mono">{formatOptionalCurrency(trade.exit)}</td>
                  <td className={trade.pnl !== null && trade.pnl >= 0 ? "text-emerald-300" : "text-rose-300"}>
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
