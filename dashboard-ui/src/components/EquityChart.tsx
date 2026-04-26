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
  const latest = data[data.length - 1];

  return (
    <section className="terminal-panel">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div>
          <p className="panel-kicker">策略累计收益曲线</p>
          <h2 className="panel-title">资金曲线与基准对比</h2>
          <p className="panel-subtitle">
            统一观察账户权益、标准化基准与近期拐点，快速识别 Alpha 是否在扩张。
          </p>
        </div>
        <div className="flex flex-col items-start gap-3 xl:items-end">
          <RangeTabs value={range} onChange={onRangeChange} />
          <p className="text-xs text-slate-500">
            最新净值 <span className="font-mono text-slate-200">{latest ? formatCurrency(latest.equity) : "--"}</span>
          </p>
        </div>
      </div>

      <div className="chart-stage mt-6 h-[360px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="equityGradient" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#2f7cff" stopOpacity={0.42} />
                <stop offset="100%" stopColor="#2f7cff" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="benchmarkGradient" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#34d399" stopOpacity={0.18} />
                <stop offset="100%" stopColor="#34d399" stopOpacity={0.01} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(148,163,184,0.12)" strokeDasharray="4 4" vertical={false} />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(value) => formatAxisTime(value, compactAxis)}
              stroke="#61708d"
              minTickGap={28}
            />
            <YAxis tickFormatter={(value) => formatCompact(Number(value))} stroke="#61708d" width={86} />
            <Tooltip
              contentStyle={{
                background: "rgba(10, 18, 34, 0.96)",
                border: "1px solid rgba(120, 144, 188, 0.22)",
                borderRadius: 18,
                color: "#d9e7ff",
                boxShadow: "0 18px 48px rgba(1, 6, 17, 0.38)",
              }}
              labelFormatter={(value) => formatDateTime(String(value))}
              formatter={(value: number, name: string) => [formatCurrency(Number(value)), name]}
            />
            <Legend wrapperStyle={{ color: "#9fb2d7" }} />
            <Line
              type="monotone"
              dataKey="equity"
              stroke="#3b82f6"
              strokeWidth={2.8}
              dot={false}
              activeDot={{ r: 5, fill: "#8ec5ff", stroke: "#0f172a" }}
              name="累计收益率"
            />
            <Area
              type="monotone"
              dataKey="equity"
              fill="url(#equityGradient)"
              stroke="transparent"
              name="账户净值"
            />
            <Line
              type="monotone"
              dataKey="benchmark"
              stroke="#38bdf8"
              strokeWidth={1.8}
              strokeDasharray="6 6"
              dot={false}
              name="基准走势"
            />
            <Area
              type="monotone"
              dataKey="benchmark"
              fill="url(#benchmarkGradient)"
              stroke="transparent"
              name="基准阴影"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
