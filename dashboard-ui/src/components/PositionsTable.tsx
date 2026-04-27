import type { PositionRow } from "../types";
import { formatOptionalCurrency } from "../lib/format";
import { getSignalDirectionLabel } from "../lib/uiText";
import { StatusBadge } from "./StatusBadge";

interface PositionsTableProps {
  positions: PositionRow[];
}

export function PositionsTable({ positions }: PositionsTableProps) {
  return (
    <section className="terminal-panel">
      <div className="mb-5 flex items-center justify-between">
        <h2 className="panel-title">持仓明细</h2>
        <span className="panel-chip">{positions.length} 笔</span>
      </div>

      {positions.length === 0 ? (
        <div className="empty-state-panel">
          当前没有活跃持仓。
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="dashboard-table">
            <thead>
              <tr>
                <th>交易对</th>
                <th>方向</th>
                <th className="text-right">仓位(USDT)</th>
                <th className="text-right">开仓价</th>
                <th className="text-right">现价</th>
                <th className="text-right">浮盈亏</th>
                <th className="text-right">杠杆</th>
                <th className="text-right">时长</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((position) => (
                <tr key={`${position.symbol}-${position.direction}`}>
                  <td className="font-semibold text-slate-50">{position.symbol}</td>
                  <td>
                    <StatusBadge label={getSignalDirectionLabel(position.direction)} tone={position.direction === "Long" ? "emerald" : "rose"} />
                  </td>
                  <td className="text-right font-mono">{position.positionSize}</td>
                  <td className="text-right font-mono">{formatOptionalCurrency(position.entryPrice)}</td>
                  <td className="text-right font-mono">{formatOptionalCurrency(position.currentPrice)}</td>
                  <td className={`text-right font-mono ${position.unrealizedPnl !== null && position.unrealizedPnl >= 0 ? "text-up" : "text-down"}`}>
                    {formatOptionalCurrency(position.unrealizedPnl)}
                  </td>
                  <td className="text-right font-mono">{position.leverage}</td>
                  <td className="text-right text-slate-300">{position.holdingTime}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
