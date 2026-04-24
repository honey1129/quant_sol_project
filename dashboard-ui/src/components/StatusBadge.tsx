import type { ReactNode } from "react";

type Tone = "emerald" | "amber" | "rose" | "sky" | "violet" | "slate";

interface StatusBadgeProps {
  label: ReactNode;
  tone?: Tone;
}

const toneMap: Record<Tone, string> = {
  emerald: "border-emerald-400/20 bg-emerald-500/10 text-emerald-300",
  amber: "border-amber-400/20 bg-amber-500/10 text-amber-300",
  rose: "border-rose-400/20 bg-rose-500/10 text-rose-300",
  sky: "border-sky-400/20 bg-sky-500/10 text-sky-300",
  violet: "border-violet-400/20 bg-violet-500/10 text-violet-300",
  slate: "border-slate-400/20 bg-slate-500/10 text-slate-300",
};

export function StatusBadge({ label, tone = "slate" }: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] ${toneMap[tone]}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {label}
    </span>
  );
}
