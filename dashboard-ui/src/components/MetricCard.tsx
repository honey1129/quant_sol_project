import type { CSSProperties } from "react";

interface MetricCardProps {
  label: string;
  value: string;
  change: string;
  helper: string;
  tone?: "neutral" | "positive" | "negative" | "highlight";
  index?: number;
}

const toneMap: Record<NonNullable<MetricCardProps["tone"]>, string> = {
  neutral: "metric-card-neutral",
  positive: "metric-card-positive",
  negative: "metric-card-negative",
  highlight: "metric-card-highlight",
};

export function MetricCard({
  label,
  value,
  change,
  helper,
  tone = "neutral",
  index = 0,
}: MetricCardProps) {
  return (
    <article
      className={`metric-card ${toneMap[tone]}`}
      style={{ "--card-delay": `${index * 0.06}s` } as CSSProperties}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="metric-card-label">{label}</p>
          <h3 className="metric-card-value">{value}</h3>
        </div>
        <span className="metric-card-orb" />
      </div>
      <div className="mt-5 flex items-end justify-between gap-4">
        <div>
          <p className="metric-card-change">{change}</p>
          <p className="metric-card-helper">{helper}</p>
        </div>
        <div className="metric-card-spark" aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
          <span />
        </div>
      </div>
    </article>
  );
}
