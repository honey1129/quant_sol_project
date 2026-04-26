import { startTransition, useEffect, useRef, useState } from "react";
import { dashboardSnapshot as mockSnapshot } from "./data/mockData";
import { DrawdownChart } from "./components/DrawdownChart";
import { EquityChart } from "./components/EquityChart";
import { LogPanel } from "./components/LogPanel";
import { MarketChartPanel } from "./components/MarketChartPanel";
import { MetricCard } from "./components/MetricCard";
import { OrderBookPanel } from "./components/OrderBookPanel";
import { ParamsPanel } from "./components/ParamsPanel";
import { PositionsTable } from "./components/PositionsTable";
import { RiskPanel } from "./components/RiskPanel";
import { SignalPanel } from "./components/SignalPanel";
import { SystemPulsePanel } from "./components/SystemPulsePanel";
import { TopNav } from "./components/TopNav";
import { TradesTable } from "./components/TradesTable";
import { buildDashboardSnapshotFromApi } from "./lib/dashboardAdapter";
import { formatClock, formatCurrency, formatDateTime, formatNumber, formatPercent } from "./lib/format";
import {
  getConnectionStatusLabel,
  getRiskLevelLabel,
  getSignalDirectionLabel,
} from "./lib/uiText";
import type {
  ApiDashboardBundle,
  ApiStrategyParamsSaveResponse,
  ApiStrategyRestartResponse,
  LogEntry,
  StrategyParams,
  ThemeMode,
  TimeRange,
} from "./types";

const POLL_MS = 10_000;

function getInitialTheme(): ThemeMode {
  if (typeof window === "undefined") {
    return "dark";
  }
  const savedTheme = window.localStorage.getItem("quant-alpha-theme");
  return savedTheme === "light" ? "light" : "dark";
}

function filterSeriesByRange(range: TimeRange, series = mockSnapshot.equityCurve) {
  if (range === "All") {
    return series;
  }

  const hoursMap: Record<Exclude<TimeRange, "All">, number> = {
    "1D": 24,
    "7D": 7 * 24,
    "30D": 30 * 24,
    "90D": 90 * 24,
  };

  const threshold = Date.now() - hoursMap[range] * 60 * 60 * 1000;
  return series.filter((point) => new Date(point.timestamp).getTime() >= threshold);
}

function prependLogEntry(current: LogEntry[], entry: LogEntry): LogEntry[] {
  return [entry, ...current].slice(0, 12);
}

function describePositionState(
  positionMode: string | null | undefined,
  netPositionQty: number | null | undefined,
  positionNotional: number | null | undefined,
) {
  const mode = String(positionMode || "").toLowerCase();
  const qty = netPositionQty ?? null;
  const hasQty = qty !== null && Number.isFinite(qty);
  const qtyLabel = hasQty ? `${qty > 0 ? "+" : ""}${formatNumber(qty, 4)}` : "--";
  const notionalLabel = positionNotional !== null && positionNotional !== undefined
    ? formatCurrency(positionNotional)
    : "--";

  if (mode === "mixed") {
    return {
      change: `双向持仓中 | 净仓 ${qtyLabel}`,
      helper: `总名义敞口 ${notionalLabel}`,
    };
  }

  if (mode === "long") {
    return {
      change: `净多仓 ${qtyLabel}`,
      helper: `当前名义敞口 ${notionalLabel}`,
    };
  }

  if (mode === "short") {
    return {
      change: `净空仓 ${qtyLabel}`,
      helper: `当前名义敞口 ${notionalLabel}`,
    };
  }

  return {
    change: "当前无活跃仓位",
    helper: `当前名义敞口 ${notionalLabel}`,
  };
}

const sidebarItems = [
  { label: "总览", icon: "⌂", active: true },
  { label: "策略中心", icon: "◫" },
  { label: "行情监控", icon: "⌁" },
  { label: "回测分析", icon: "◌" },
  { label: "实盘交易", icon: "◎" },
  { label: "风险控制", icon: "◍" },
  { label: "账户管理", icon: "◪" },
  { label: "系统设置", icon: "⚙" },
];

export default function App() {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);
  const [now, setNow] = useState<Date>(new Date());
  const [range, setRange] = useState<TimeRange>("30D");
  const [snapshot, setSnapshot] = useState(mockSnapshot);
  const [params, setParams] = useState<StrategyParams>(mockSnapshot.params);
  const [paramsDirty, setParamsDirty] = useState(false);
  const paramsDirtyRef = useRef(false);
  const [savedAt, setSavedAt] = useState<Date | null>(null);
  const [savingParams, setSavingParams] = useState(false);
  const [restartingStrategy, setRestartingStrategy] = useState(false);
  const [localLogs, setLocalLogs] = useState<LogEntry[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    window.localStorage.setItem("quant-alpha-theme", theme);
  }, [theme]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setNow(new Date());
    }, 1000);

    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    paramsDirtyRef.current = paramsDirty;
  }, [paramsDirty]);

  useEffect(() => {
    let active = true;
    let timer = 0;
    let controller: AbortController | null = null;

    async function loadDashboard() {
      controller?.abort();
      controller = new AbortController();

      try {
        const response = await fetch("/api/dashboard", {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Dashboard 接口返回异常状态码：${response.status}`);
        }

        const payload = (await response.json()) as ApiDashboardBundle;
        if (!active) {
          return;
        }

        const nextSnapshot = buildDashboardSnapshotFromApi(payload);
        startTransition(() => {
          setSnapshot(nextSnapshot);
          if (!paramsDirtyRef.current) {
            setParams(nextSnapshot.params);
          }
          setError("");
        });
      } catch (fetchError) {
        if (!active || (fetchError instanceof DOMException && fetchError.name === "AbortError")) {
          return;
        }
        const message = fetchError instanceof Error ? fetchError.message : "拉取 /api/dashboard 失败";
        startTransition(() => {
          setError(message);
          setSnapshot((current) => current);
        });
      }
    }

    void loadDashboard();
    timer = window.setInterval(() => {
      void loadDashboard();
    }, POLL_MS);

    return () => {
      active = false;
      controller?.abort();
      window.clearInterval(timer);
    };
  }, []);

  const filteredCurve = filterSeriesByRange(range, snapshot.equityCurve);
  const visibleLogs = [...localLogs, ...snapshot.logs].slice(0, 12);
  const positionMetricText = describePositionState(
    snapshot.metrics.positionMode,
    snapshot.metrics.netPositionQty,
    snapshot.metrics.positionNotional,
  );

  const metricCards: Array<{
    label: string;
    value: string;
    change: string;
    helper: string;
    tone: "neutral" | "positive" | "negative" | "highlight";
  }> = [
    {
      label: "总资产",
      value: formatCurrency(snapshot.metrics.equity).replace("$", ""),
      change: "USDT",
      helper: `≈ ${formatCurrency(snapshot.metrics.equity)} USD`,
      tone: "neutral" as const,
    },
    {
      label: "当日收益",
      value: formatCurrency(snapshot.metrics.dailyPnl),
      change: formatPercent(snapshot.metrics.dailyPnl / Math.max(snapshot.metrics.equity - snapshot.metrics.dailyPnl, 1) * 100, 2),
      helper: "近 24 小时权益变化",
      tone: snapshot.metrics.dailyPnl >= 0 ? "positive" : "negative",
    },
    {
      label: "累计收益率",
      value: formatPercent(snapshot.metrics.totalReturnPct),
      change: snapshot.dataSource === "live" ? "实时净值历史" : "回测/混合快照",
      helper: "账户累计表现",
      tone: "positive" as const,
    },
    {
      label: "夏普比率",
      value: formatNumber(snapshot.metrics.sharpeRatio, 2),
      change: snapshot.dataSource === "live" ? "近期研究快照" : "风险调整收益",
      helper: "策略稳定性",
      tone: "neutral" as const,
    },
    {
      label: "最大回撤",
      value: formatPercent(snapshot.metrics.maxDrawdownPct, 2, false),
      change: "历史极值",
      helper: "资金回撤压力",
      tone: "negative" as const,
    },
    {
      label: "胜率",
      value: formatPercent(snapshot.metrics.winRatePct, 2, false),
      change: `${snapshot.trades.length} 条最近交易映射`,
      helper: "执行质量",
      tone: "highlight" as const,
    },
  ];

  const systemStatusItems = [
    {
      label: "API 连接",
      value: getConnectionStatusLabel(snapshot.risk.apiStatus),
      tone: snapshot.risk.apiStatus === "Connected" ? "ok" : "warn",
    },
    {
      label: "行情连接",
      value: getConnectionStatusLabel(snapshot.risk.wsStatus),
      tone: snapshot.risk.wsStatus === "Connected" ? "ok" : "warn",
    },
    {
      label: "策略状态",
      value: getSignalDirectionLabel(snapshot.signal.direction),
      tone: snapshot.status === "Running" ? "ok" : "warn",
    },
    {
      label: "最近更新",
      value: formatClock(now),
      tone: "ok",
    },
  ];

  function handleParamChange<K extends keyof StrategyParams>(key: K, value: StrategyParams[K]) {
    setParamsDirty(true);
    setParams((current) => ({
      ...current,
      [key]: value,
    }));
  }

  async function handleSaveParams() {
    if (savingParams) {
      return;
    }

    setSavingParams(true);
    try {
      const response = await fetch("/api/strategy-params", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(params),
      });
      const result = (await response.json()) as ApiStrategyParamsSaveResponse;
      if (!response.ok || !result.ok) {
        throw new Error(result.error || `策略参数接口返回异常状态码：${response.status}`);
      }

      const nextSnapshot = result.bundle ? buildDashboardSnapshotFromApi(result.bundle) : snapshot;
      startTransition(() => {
        if (result.bundle) {
          setSnapshot(nextSnapshot);
          setParams(nextSnapshot.params);
        }
        setSavedAt(result.saved_at ? new Date(result.saved_at) : new Date());
        setParamsDirty(false);
      });

      const entry: LogEntry = {
        id: `save-${Date.now()}`,
        time: new Date().toISOString(),
        level: result.restart_required ? "WARN" : "SUCCESS",
        message: result.message || "策略参数已保存。",
      };
      setLocalLogs((current) => prependLogEntry(current, entry));
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "保存策略参数失败";
      const entry: LogEntry = {
        id: `save-error-${Date.now()}`,
        time: new Date().toISOString(),
        level: "ERROR",
        message,
      };
      setLocalLogs((current) => prependLogEntry(current, entry));
    } finally {
      setSavingParams(false);
    }
  }

  async function handleRestartStrategy() {
    if (restartingStrategy) {
      return;
    }

    setRestartingStrategy(true);
    try {
      const response = await fetch("/api/restart-strategy", {
        method: "POST",
      });
      const result = (await response.json()) as ApiStrategyRestartResponse;
      if (!response.ok || !result.ok) {
        throw new Error(result.error || `策略重启接口返回异常状态码：${response.status}`);
      }

      if (result.bundle) {
        const nextSnapshot = buildDashboardSnapshotFromApi(result.bundle);
        startTransition(() => {
          setSnapshot(nextSnapshot);
          if (!paramsDirtyRef.current) {
            setParams(nextSnapshot.params);
          }
        });
      }

      const entry: LogEntry = {
        id: `restart-${Date.now()}`,
        time: result.restarted_at || new Date().toISOString(),
        level: "WARN",
        message: result.output ? `${result.message} ${result.output}` : (result.message || "已发起策略重启。"),
      };
      setLocalLogs((current) => prependLogEntry(current, entry));
    } catch (restartError) {
      const message = restartError instanceof Error ? restartError.message : "重启策略失败";
      const entry: LogEntry = {
        id: `restart-error-${Date.now()}`,
        time: new Date().toISOString(),
        level: "ERROR",
        message,
      };
      setLocalLogs((current) => prependLogEntry(current, entry));
    } finally {
      setRestartingStrategy(false);
    }
  }

  function handleResetParams() {
    setParams(snapshot.params);
    setParamsDirty(false);
    const entry: LogEntry = {
      id: `reset-${Date.now()}`,
      time: new Date().toISOString(),
      level: "INFO",
      message: snapshot.dataSource === "live"
        ? "策略参数已重置为 /api/dashboard 当前返回的最新值。"
        : "策略参数已重置为模拟基线配置。",
    };
    setLocalLogs((current) => prependLogEntry(current, entry));
  }

  return (
    <div className="cockpit-shell">
      <aside className="cockpit-sidebar">
        <div className="cockpit-brand">
          <div className="cockpit-brand-mark">◫</div>
          <div>
            <p className="cockpit-brand-kicker">量化交易控制台</p>
            <h1 className="cockpit-brand-title">{snapshot.strategyName}</h1>
          </div>
        </div>

        <nav className="cockpit-nav">
          {sidebarItems.map((item) => (
            <button
              key={item.label}
              type="button"
              className={`cockpit-nav-item ${item.active ? "is-active" : ""}`}
            >
              <span className="cockpit-nav-icon">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <section className="cockpit-sidebar-panel">
          <div className="flex items-center justify-between">
            <p className="panel-kicker">系统状态</p>
            <span className="panel-chip">实时</span>
          </div>
          <div className="mt-5 space-y-3">
            {systemStatusItems.map((item) => (
              <div key={item.label} className="flex items-center justify-between text-sm">
                <span className="text-slate-400">{item.label}</span>
                <span className="flex items-center gap-2 font-medium text-slate-100">
                  <span className={`h-2.5 w-2.5 rounded-full ${item.tone === "ok" ? "bg-emerald-400" : "bg-amber-400"}`} />
                  {item.value}
                </span>
              </div>
            ))}
          </div>
          <div className="mt-6 rounded-2xl border border-white/8 bg-white/[0.03] p-4">
            <p className="text-xs uppercase tracking-[0.22em] text-slate-500">运行摘要</p>
            <p className="mt-3 text-sm text-slate-300">
              交易所 {snapshot.exchange} · 风险 {getRiskLevelLabel(snapshot.metrics.riskLevel)} · 持仓 {snapshot.metrics.openPositions}
            </p>
            <p className="mt-2 text-xs text-slate-500">
              最近更新 {formatDateTime(snapshot.updatedAt)} · 轮询 {POLL_MS / 1000}s
            </p>
          </div>
        </section>

        <SystemPulsePanel systemPulse={snapshot.systemPulse} />
      </aside>

      <main className="cockpit-main">
        <TopNav
          productName={snapshot.productName}
          strategyName={snapshot.strategyName}
          exchange={snapshot.exchange}
          status={snapshot.status}
          updatedAt={snapshot.updatedAt}
          dataSource={snapshot.dataSource}
          now={now}
          theme={theme}
          onThemeToggle={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
        />

        {error ? (
          <div className="mb-5 rounded-2xl border border-amber-400/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
            实时接口拉取告警：{error}。当前页面继续保留最近一次成功获取的状态。
          </div>
        ) : null}

        <section className="grid gap-4 md:grid-cols-2 2xl:grid-cols-6">
          {metricCards.map((metric, index) => (
            <MetricCard
              key={metric.label}
              label={metric.label}
              value={metric.value}
              change={metric.change}
              helper={metric.helper}
              tone={metric.tone}
              index={index}
            />
          ))}
        </section>

        <section className="mt-6 grid gap-6 2xl:grid-cols-[1.12fr_1.08fr_0.8fr]">
          <div className="space-y-6">
            <EquityChart data={filteredCurve} range={range} onRangeChange={setRange} />
            <TradesTable trades={snapshot.trades} />
            <DrawdownChart data={filteredCurve} range={range} />
          </div>

          <div className="space-y-6">
            <MarketChartPanel marketChart={snapshot.marketChart} />
            <PositionsTable positions={snapshot.positions} />
            <ParamsPanel
              params={params}
              savedAt={savedAt}
              saving={savingParams}
              restarting={restartingStrategy}
              onChange={handleParamChange}
              onSave={handleSaveParams}
              onRestart={handleRestartStrategy}
              onReset={handleResetParams}
            />
          </div>

          <div className="space-y-6">
            <SignalPanel signal={snapshot.signal} now={now} />
            <OrderBookPanel orderBook={snapshot.orderBook} symbol={snapshot.marketChart.symbol} />
            <RiskPanel risk={snapshot.risk} />
          </div>
        </section>

        <section className="mt-6">
          <LogPanel logs={visibleLogs} />
        </section>

        <footer className="mt-6 grid gap-4 xl:grid-cols-4">
          <div className="cockpit-footer-stat">
            <p>当前信号</p>
            <strong>{getSignalDirectionLabel(snapshot.signal.direction)}</strong>
          </div>
          <div className="cockpit-footer-stat">
            <p>风险等级</p>
            <strong>{getRiskLevelLabel(snapshot.metrics.riskLevel)}</strong>
          </div>
          <div className="cockpit-footer-stat">
            <p>名义敞口</p>
            <strong>{positionMetricText.helper.replace("当前名义敞口 ", "").replace("总名义敞口 ", "")}</strong>
          </div>
          <div className="cockpit-footer-stat">
            <p>账户模式</p>
            <strong>{snapshot.metrics.openPositions > 0 ? positionMetricText.change : "空仓待命"}</strong>
          </div>
        </footer>
      </main>
    </div>
  );
}
