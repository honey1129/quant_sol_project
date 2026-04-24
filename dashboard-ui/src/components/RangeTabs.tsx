import type { TimeRange } from "../types";
import { getTimeRangeLabel } from "../lib/uiText";

interface RangeTabsProps {
  value: TimeRange;
  onChange: (value: TimeRange) => void;
}

const ranges: TimeRange[] = ["1D", "7D", "30D", "90D", "All"];

export function RangeTabs({ value, onChange }: RangeTabsProps) {
  return (
    <div className="inline-flex rounded-full border border-white/10 bg-slate-200/70 p-1 dark:bg-slate-950/70">
      {ranges.map((range) => {
        const active = range === value;
        return (
          <button
            key={range}
            type="button"
            onClick={() => onChange(range)}
            className={`rounded-full px-3 py-1.5 text-xs font-semibold tracking-[0.18em] transition ${
              active
                ? "bg-sky-500 text-white shadow-lg shadow-sky-500/20"
                : "text-slate-500 hover:text-slate-950 dark:text-slate-400 dark:hover:text-white"
            }`}
          >
            {getTimeRangeLabel(range)}
          </button>
        );
      })}
    </div>
  );
}
