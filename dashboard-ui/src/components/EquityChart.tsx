import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EquityPoint, TimeRange } from "../types";
import { formatAxisTime, formatCompact, formatCurrency, formatDateTime } from "../lib/format";
import { RangeTabs } from "./RangeTabs";

interface EquityChartProps {
  data: EquityPoint[];
  range: TimeRange;
  onRangeChange: (value: TimeRange) => void;
}

export function EquityChart({ data, range, onRangeChange }: EquityChartProps) {
  const compactAxis = range !== "1D";

  return (
    <section className="terminal-panel">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="terminal-kicker">Performance</p>
          <h2 className="terminal-title">Equity Curve vs Benchmark</h2>
          <p className="terminal-subtitle">Account equity is plotted against a normalized crypto benchmark to spot alpha drift fast.</p>
        </div>
        <RangeTabs value={range} onChange={onRangeChange} />
      </div>

      <div className="mt-6 h-[340px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="equityGradient" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#22c55e" stopOpacity={0.34} />
                <stop offset="100%" stopColor="#22c55e" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(148,163,184,0.15)" strokeDasharray="3 3" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(value) => formatAxisTime(value, compactAxis)}
              stroke="#64748b"
              minTickGap={28}
            />
            <YAxis tickFormatter={(value) => formatCompact(Number(value))} stroke="#64748b" width={86} />
            <Tooltip
              contentStyle={{
                background: "rgba(2, 6, 23, 0.95)",
                border: "1px solid rgba(148, 163, 184, 0.15)",
                borderRadius: 16,
                color: "#e2e8f0",
              }}
              labelFormatter={(value) => formatDateTime(String(value))}
              formatter={(value: number, name: string) => [formatCurrency(Number(value)), name === "equity" ? "Equity" : "Benchmark"]}
            />
            <Legend />
            <Area
              type="monotone"
              dataKey="equity"
              stroke="#22c55e"
              strokeWidth={2.4}
              fill="url(#equityGradient)"
              name="equity"
            />
            <Line
              type="monotone"
              dataKey="benchmark"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={false}
              strokeDasharray="5 5"
              name="benchmark"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
