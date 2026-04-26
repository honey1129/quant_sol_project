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
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="panel-kicker">风险回撤分析</p>
          <h2 className="panel-title">回撤压力轨迹</h2>
          <p className="panel-subtitle">把资金谷底位置钉出来，帮助判断策略是否进入不舒适区间。</p>
        </div>
        <span className="panel-chip">风险</span>
      </div>

      <div className="chart-stage mt-6 h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="drawdownGradient" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#ff5f6d" stopOpacity={0.36} />
                <stop offset="100%" stopColor="#ff5f6d" stopOpacity={0.04} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(148,163,184,0.12)" strokeDasharray="4 4" vertical={false} />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(value) => formatAxisTime(value, range !== "1D")}
              stroke="#61708d"
              minTickGap={28}
            />
            <YAxis tickFormatter={(value) => `${Number(value).toFixed(1)}%`} stroke="#61708d" width={68} />
            <Tooltip
              contentStyle={{
                background: "rgba(10, 18, 34, 0.96)",
                border: "1px solid rgba(120, 144, 188, 0.22)",
                borderRadius: 18,
                color: "#d9e7ff",
              }}
              labelFormatter={(value) => formatDateTime(String(value))}
              formatter={(value: number) => [formatPercent(Number(value), 2, false), "回撤"]}
            />
            <ReferenceLine y={0} stroke="rgba(148,163,184,0.22)" />
            <Area type="monotone" dataKey="drawdown" stroke="#ff5f6d" fill="url(#drawdownGradient)" strokeWidth={2.4} />
            {maxDrawdownPoint ? (
              <ReferenceDot
                x={maxDrawdownPoint.timestamp}
                y={maxDrawdownPoint.drawdown}
                r={4.5}
                fill="#ffe4e6"
                stroke="#ff5f6d"
                label={{
                  value: `最大回撤 ${formatPercent(maxDrawdownPoint.drawdown, 2, false)}`,
                  position: "top",
                  fill: "#ffc2c9",
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
