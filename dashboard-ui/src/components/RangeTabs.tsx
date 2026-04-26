import type { TimeRange } from "../types";
import { getTimeRangeLabel } from "../lib/uiText";

interface RangeTabsProps {
  value: TimeRange;
  onChange: (value: TimeRange) => void;
}

const ranges: TimeRange[] = ["1D", "7D", "30D", "90D", "All"];

export function RangeTabs({ value, onChange }: RangeTabsProps) {
  return (
    <div className="inline-flex rounded-full border border-white/10 bg-[#09111e] p-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
      {ranges.map((range) => {
        const active = range === value;
        return (
          <button
            key={range}
            type="button"
            onClick={() => onChange(range)}
            className={`rounded-full px-3.5 py-1.5 text-xs font-semibold tracking-[0.18em] transition ${
              active
                ? "bg-[#2f7cff] text-white shadow-[0_10px_24px_rgba(47,124,255,0.32)]"
                : "text-slate-500 hover:text-slate-100"
            }`}
          >
            {getTimeRangeLabel(range)}
          </button>
        );
      })}
    </div>
  );
}
