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
        <div>
          <p className="panel-kicker">当前持仓</p>
          <h2 className="panel-title">持仓明细</h2>
        </div>
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
                {["交易对", "方向", "仓位(USDT)", "开仓价", "现价", "浮盈亏", "杠杆", "时长"].map((head) => (
                  <th key={head}>{head}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {positions.map((position) => (
                <tr key={`${position.symbol}-${position.direction}`}>
                  <td className="font-semibold text-slate-50">{position.symbol}</td>
                  <td>
                    <StatusBadge label={getSignalDirectionLabel(position.direction)} tone={position.direction === "Long" ? "emerald" : "rose"} />
                  </td>
                  <td>{position.positionSize}</td>
                  <td className="font-mono">{formatOptionalCurrency(position.entryPrice)}</td>
                  <td className="font-mono">{formatOptionalCurrency(position.currentPrice)}</td>
                  <td className={position.unrealizedPnl !== null && position.unrealizedPnl >= 0 ? "text-emerald-300" : "text-rose-300"}>
                    {formatOptionalCurrency(position.unrealizedPnl)}
                  </td>
                  <td>{position.leverage}</td>
                  <td>{position.holdingTime}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
