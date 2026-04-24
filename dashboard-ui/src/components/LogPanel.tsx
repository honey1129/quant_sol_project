import type { LogEntry } from "../types";
import { formatDateTime } from "../lib/format";
import { getLogLevelLabel } from "../lib/uiText";
import { StatusBadge } from "./StatusBadge";

interface LogPanelProps {
  logs: LogEntry[];
}

function toneFromLevel(level: LogEntry["level"]) {
  if (level === "SUCCESS") {
    return "emerald";
  }
  if (level === "WARN") {
    return "amber";
  }
  if (level === "ERROR") {
    return "rose";
  }
  return "sky";
}

export function LogPanel({ logs }: LogPanelProps) {
  return (
    <section className="terminal-panel">
      <div className="mb-5">
        <p className="terminal-kicker">运行日志</p>
        <h2 className="terminal-title">实时系统日志</h2>
      </div>

      {logs.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-white/10 bg-slate-950/[0.03] px-4 py-10 text-center text-sm text-slate-500 dark:bg-white/[0.03] dark:text-slate-400">
          暂无最近运行日志。
        </div>
      ) : (
      <div className="max-h-[420px] space-y-3 overflow-y-auto pr-2">
        {logs.map((log) => (
          <article
            key={log.id}
            className="rounded-2xl border border-white/10 bg-slate-950/[0.03] p-4 dark:bg-slate-950/70"
          >
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <StatusBadge label={getLogLevelLabel(log.level)} tone={toneFromLevel(log.level)} />
                <span className="font-mono text-xs text-slate-500">{formatDateTime(log.time)}</span>
              </div>
              <p className="text-sm text-slate-200">{log.message}</p>
            </div>
          </article>
        ))}
      </div>
      )}
    </section>
  );
}
