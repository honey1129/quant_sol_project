import { formatDateTime } from "../lib/format";
import { getDataSourceLabel, getStrategyStatusLabel, getThemeLabel } from "../lib/uiText";
import type { DataSource, ExchangeName, StrategyStatus, ThemeMode } from "../types";

interface TopNavProps {
  productName: string;
  strategyName: string;
  exchange: ExchangeName;
  status: StrategyStatus;
  updatedAt: string;
  dataSource: DataSource;
  now: Date;
  theme: ThemeMode;
  onThemeToggle: () => void;
}

export function TopNav({
  productName,
  strategyName,
  exchange,
  status,
  updatedAt,
  dataSource,
  now,
  theme,
  onThemeToggle,
}: TopNavProps) {
  const nowLabel = new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone: "Asia/Shanghai",
  }).format(now);

  return (
    <header className="top-nav-shell">
      <div className="flex flex-1 items-center gap-4">
        <div className="top-search">
          <span className="top-search-icon">⌕</span>
          <input
            aria-label="搜索交易对"
            className="top-search-input"
            placeholder="搜索交易对 / 策略 / 功能"
            type="text"
          />
        </div>

        <div className="top-header-meta hidden 2xl:flex">
          <span>{nowLabel} (UTC+8)</span>
          <span className="top-header-separator" />
          <span>{productName}</span>
          <span className="top-header-separator" />
          <span>{strategyName}</span>
        </div>
      </div>

      <div className="top-actions">
        <div className="top-chip">
          <span className="text-slate-400">交易所</span>
          <strong>{exchange}</strong>
        </div>
        <div className="top-chip">
          <span className="text-slate-400">状态</span>
          <strong>{getStrategyStatusLabel(status)}</strong>
        </div>
        <div className="top-chip">
          <span className="text-slate-400">数据</span>
          <strong>{getDataSourceLabel(dataSource)}</strong>
        </div>
        <div className="top-chip hidden xl:flex">
          <span className="text-slate-400">更新</span>
          <strong>{formatDateTime(updatedAt)}</strong>
        </div>
        <button type="button" onClick={onThemeToggle} className="top-theme-button">
          {getThemeLabel(theme)}
        </button>
      </div>
    </header>
  );
}
