import type { LogEntry } from "../types";
import { formatDateTime } from "../lib/format";
import { getLogLevelLabel } from "../lib/uiText";

interface LogPanelProps {
  logs: LogEntry[];
}

const LEVEL_TONE: Record<LogEntry["level"], string> = {
  INFO: "text-sky-300",
  SUCCESS: "text-up",
  WARN: "text-amber-300",
  ERROR: "text-down",
};

export function LogPanel({ logs }: LogPanelProps) {
  return (
    <section className="terminal-panel">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="panel-title">实时系统日志</h2>
        <span className="panel-chip">{logs.length} 条</span>
      </div>

      {logs.length === 0 ? (
        <div className="empty-state-panel">暂无最近运行日志。</div>
      ) : (
        <ul className="divide-y divide-white/5 font-mono text-[13px] leading-6">
          {logs.map((log) => (
            <li key={log.id} className="flex items-start gap-3 py-2">
              <span className="shrink-0 text-xs text-slate-400">{formatDateTime(log.time)}</span>
              <span className={`shrink-0 w-12 text-xs font-semibold uppercase tracking-wider ${LEVEL_TONE[log.level] ?? "text-slate-300"}`}>
                {getLogLevelLabel(log.level)}
              </span>
              <span className="min-w-0 flex-1 break-words text-slate-200">{log.message}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
