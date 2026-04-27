import { formatNumber } from "../lib/format";
import type { SystemPulse } from "../types";

interface SystemPulsePanelProps {
  systemPulse: SystemPulse;
}

const pulseItems = [
  { key: "cpu", label: "CPU 占用", tone: "is-blue" },
  { key: "memory", label: "内存占用", tone: "is-violet" },
  { key: "disk", label: "敞口压力", tone: "is-amber" },
  { key: "latency", label: "网络延迟", tone: "is-emerald", suffix: "ms" },
] as const;

export function SystemPulsePanel({ systemPulse }: SystemPulsePanelProps) {
  return (
    <section className="cockpit-sidebar-panel">
      <div className="flex items-center justify-between">
        <h2 className="panel-title">运行态脉冲</h2>
        <span className="panel-chip">Live</span>
      </div>

      <div className="mt-5 space-y-4">
        {pulseItems.map((item) => {
          const value = systemPulse[item.key];
          const width = item.key === "latency" ? Math.min(100, (value / 120) * 100) : value;
          const suffix = "suffix" in item ? item.suffix : "%";
          return (
            <div key={item.key} className="sidebar-meter">
              <div className="sidebar-meter-head">
                <span>{item.label}</span>
                <strong>{formatNumber(value, 0)}{suffix}</strong>
              </div>
              <div className="sidebar-meter-track">
                <div className={`sidebar-meter-fill ${item.tone}`} style={{ width: `${width}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
