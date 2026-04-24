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
  neutral: "from-slate-900/90 to-slate-950/80 dark:from-slate-900/90 dark:to-slate-950/80",
  positive: "from-emerald-950/80 to-slate-950/90 dark:from-emerald-950/70 dark:to-slate-950/90",
  negative: "from-rose-950/80 to-slate-950/90 dark:from-rose-950/70 dark:to-slate-950/90",
  highlight: "from-sky-950/80 to-slate-950/90 dark:from-sky-950/70 dark:to-slate-950/90",
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
      className={`metric-card rounded-2xl border border-white/10 bg-gradient-to-br p-4 shadow-terminal transition-transform duration-300 hover:-translate-y-1 ${toneMap[tone]}`}
      style={{ "--card-delay": `${index * 0.12}s` } as CSSProperties}
    >
      <p className="text-[11px] tracking-[0.12em] text-slate-400">{label}</p>
      <h3 className="mt-3 font-mono text-2xl font-semibold text-slate-950 dark:text-white">{value}</h3>
      <p className="mt-2 text-sm font-medium text-slate-600 dark:text-slate-300">{change}</p>
      <p className="mt-4 text-xs text-slate-500 dark:text-slate-400">{helper}</p>
    </article>
  );
}
