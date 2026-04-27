import type { TimeRange } from "../types";
import { getTimeRangeLabel } from "../lib/uiText";
import { SegmentedControl } from "./SegmentedControl";

interface RangeTabsProps {
  value: TimeRange;
  onChange: (value: TimeRange) => void;
}

const ranges: TimeRange[] = ["1D", "7D", "30D", "90D", "All"];

export function RangeTabs({ value, onChange }: RangeTabsProps) {
  return (
    <SegmentedControl
      value={value}
      options={ranges.map((range) => ({ value: range, label: getTimeRangeLabel(range) }))}
      onChange={onChange}
      ariaLabel="时间范围"
    />
  );
}
