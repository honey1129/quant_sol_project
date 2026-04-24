import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EquityPoint, TimeRange } from "../types";
import { formatAxisTime, formatDateTime, formatPercent } from "../lib/format";

interface DrawdownChartProps {
  data: EquityPoint[];
  range: TimeRange;
}

export function DrawdownChart({ data, range }: DrawdownChartProps) {
  const maxDrawdownPoint = data.reduce((worst, point) => {
    if (!worst || point.drawdown < worst.drawdown) {
      return point;
    }
    return worst;
  }, null as EquityPoint | null);

  return (
    <section className="terminal-panel">
      <div>
        <p className="terminal-kicker">风险画像</p>
        <h2 className="terminal-title">历史回撤</h2>
        <p className="terminal-subtitle">重点标出回撤深度，让资金压力在恶化前就能被看到。</p>
      </div>

      <div className="mt-6 h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="drawdownGradient" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.42} />
                <stop offset="100%" stopColor="#f43f5e" stopOpacity={0.04} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(148,163,184,0.14)" strokeDasharray="3 3" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(value) => formatAxisTime(value, range !== "1D")}
              stroke="#64748b"
              minTickGap={28}
            />
            <YAxis tickFormatter={(value) => `${Number(value).toFixed(1)}%`} stroke="#64748b" width={68} />
            <Tooltip
              contentStyle={{
                background: "rgba(2, 6, 23, 0.95)",
                border: "1px solid rgba(148, 163, 184, 0.15)",
                borderRadius: 16,
                color: "#e2e8f0",
              }}
              labelFormatter={(value) => formatDateTime(String(value))}
              formatter={(value: number) => [formatPercent(Number(value), 2, false), "回撤"]}
            />
            <ReferenceLine y={0} stroke="rgba(148,163,184,0.35)" />
            <Area type="monotone" dataKey="drawdown" stroke="#fb7185" fill="url(#drawdownGradient)" strokeWidth={2.2} />
            {maxDrawdownPoint ? (
              <ReferenceDot
                x={maxDrawdownPoint.timestamp}
                y={maxDrawdownPoint.drawdown}
                r={5}
                fill="#ffe4e6"
                stroke="#fb7185"
                label={{
                  value: `最大回撤 ${formatPercent(maxDrawdownPoint.drawdown, 2, false)}`,
                  position: "top",
                  fill: "#fecdd3",
                  fontSize: 12,
                }}
              />
            ) : null}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
