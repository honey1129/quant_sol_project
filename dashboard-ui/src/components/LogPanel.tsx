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
      <div className="mb-5 flex items-center justify-between">
        <div>
          <p className="panel-kicker">运行日志</p>
          <h2 className="panel-title">实时系统日志</h2>
        </div>
        <span className="panel-chip">{logs.length} 条</span>
      </div>

      {logs.length === 0 ? (
        <div className="empty-state-panel">
          暂无最近运行日志。
        </div>
      ) : (
        <div className="grid gap-3 xl:grid-cols-2">
          {logs.map((log) => (
            <article
              key={log.id}
              className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 transition hover:border-white/15 hover:bg-white/[0.05]"
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-3">
                    <StatusBadge label={getLogLevelLabel(log.level)} tone={toneFromLevel(log.level)} />
                    <span className="font-mono text-xs text-slate-500">{formatDateTime(log.time)}</span>
                  </div>
                  <p className="mt-3 text-sm leading-6 text-slate-200">{log.message}</p>
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
