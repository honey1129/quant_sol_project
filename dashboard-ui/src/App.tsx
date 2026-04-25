import { startTransition, useEffect, useRef, useState, type CSSProperties } from "react";
import { dashboardSnapshot as mockSnapshot } from "./data/mockData";
import { DrawdownChart } from "./components/DrawdownChart";
import { EquityChart } from "./components/EquityChart";
import { LogPanel } from "./components/LogPanel";
import { MetricCard } from "./components/MetricCard";
import { ParamsPanel } from "./components/ParamsPanel";
import { PositionsTable } from "./components/PositionsTable";
import { RiskPanel } from "./components/RiskPanel";
import { SignalPanel } from "./components/SignalPanel";
import { TopNav } from "./components/TopNav";
import { TradesTable } from "./components/TradesTable";
import { buildDashboardSnapshotFromApi } from "./lib/dashboardAdapter";
import { formatCurrency, formatNumber, formatPercent } from "./lib/format";
import { getRiskLevelLabel, getSignalDirectionLabel, getStrategyStatusLabel } from "./lib/uiText";
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
      value: formatCurrency(snapshot.metrics.equity),
      change: formatCurrency(snapshot.metrics.dailyPnl),
      helper: "当前账户权益",
      tone: "highlight" as const,
    },
    {
      label: "今日收益",
      value: formatCurrency(snapshot.metrics.dailyPnl),
      change: snapshot.metrics.dailyPnl >= 0 ? "当前交易时段盈利中" : "当前交易时段低于目标",
      helper: "近 24 小时盈亏",
      tone: snapshot.metrics.dailyPnl >= 0 ? "positive" : "negative",
    },
    {
      label: "累计收益",
      value: formatPercent(snapshot.metrics.totalReturnPct),
      change: snapshot.dataSource === "mock" ? "模拟基线数据" : "来源于实时净值历史",
      helper: "复合累计收益率",
      tone: "positive" as const,
    },
    {
      label: "最大回撤",
      value: formatPercent(snapshot.metrics.maxDrawdownPct, 2, false),
      change: "历史最大资金回撤",
      helper: "从峰值到谷值的跌幅",
      tone: "negative" as const,
    },
    {
      label: "夏普比率",
      value: formatNumber(snapshot.metrics.sharpeRatio, 2),
      change: snapshot.dataSource === "live"
        ? "来源于最新回测研究快照"
        : "风险调整后收益",
      helper: "年化估算值",
      tone: "neutral" as const,
    },
    {
      label: "胜率",
      value: formatPercent(snapshot.metrics.winRatePct, 1, false),
      change: snapshot.dataSource === "live"
        ? "来源于最新回测平仓交易"
        : "仅统计已成交交易",
      helper: "执行质量表现",
      tone: "positive" as const,
    },
    {
      label: "活跃仓位",
      value: String(snapshot.metrics.openPositions),
      change: positionMetricText.change,
      helper: positionMetricText.helper,
      tone: "neutral" as const,
    },
    {
      label: "风险等级",
      value: getRiskLevelLabel(snapshot.metrics.riskLevel),
      change: "基于杠杆与保证金占用",
      helper: "当前交易时段风险分级",
      tone: snapshot.metrics.riskLevel === "High" ? "negative" : "highlight",
    },
  ];

  const streamItems = [
    `状态 ${getStrategyStatusLabel(snapshot.status)}`,
    `信号 ${getSignalDirectionLabel(snapshot.signal.direction)}`,
    `风险 ${getRiskLevelLabel(snapshot.metrics.riskLevel)}`,
    `持仓 ${snapshot.metrics.openPositions}`,
    `净值 ${formatCurrency(snapshot.metrics.equity)}`,
    `今日 ${formatCurrency(snapshot.metrics.dailyPnl)}`,
    `轮询 ${POLL_MS / 1000}s`,
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
    <div className="dashboard-shell min-h-screen px-4 py-4 sm:px-6 xl:px-8">
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

      <section
        className="dashboard-hero panel-enter"
        style={{ "--panel-delay": "0.04s" } as CSSProperties}
      >
        <div className="dashboard-hero-copy">
          <p className="terminal-kicker">Runtime Pulse</p>
          <h2 className="terminal-title">让页面状态跟着策略节奏一起呼吸</h2>
          <p className="terminal-subtitle">
            保留量化监控的密度，同时把关键状态做成持续动态反馈，方便快速扫一眼就知道系统是否在正常工作。
          </p>
        </div>
        <div className="dashboard-stream" aria-hidden="true">
          <div className="dashboard-stream-track">
            {[...streamItems, ...streamItems].map((item, index) => (
              <span key={`${item}-${index}`} className="dashboard-stream-item">
                {item}
              </span>
            ))}
          </div>
        </div>
      </section>

      {error ? (
        <div className="mb-6 rounded-2xl border border-amber-400/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          实时接口拉取告警：{error}。当前页面继续保留最近一次成功获取的状态。
        </div>
      ) : null}

      <section
        className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-8"
        style={{ "--panel-delay": "0.08s" } as CSSProperties}
      >
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

      <section
        className="mt-6 grid gap-6 xl:grid-cols-[1.55fr_0.95fr]"
        style={{ "--panel-delay": "0.12s" } as CSSProperties}
      >
        <div className="space-y-6 panel-enter">
          <EquityChart data={filteredCurve} range={range} onRangeChange={setRange} />
          <DrawdownChart data={filteredCurve} range={range} />
        </div>

        <div className="space-y-6 panel-enter" style={{ "--panel-delay": "0.18s" } as CSSProperties}>
          <SignalPanel signal={snapshot.signal} now={now} />
          <RiskPanel risk={snapshot.risk} />
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
      </section>

      <section
        className="mt-6 grid gap-6 xl:grid-cols-[1.15fr_1fr]"
        style={{ "--panel-delay": "0.22s" } as CSSProperties}
      >
        <div className="panel-enter">
          <PositionsTable positions={snapshot.positions} />
        </div>
        <div className="panel-enter" style={{ "--panel-delay": "0.26s" } as CSSProperties}>
          <TradesTable trades={snapshot.trades} />
        </div>
      </section>

      <section className="mt-6 panel-enter" style={{ "--panel-delay": "0.3s" } as CSSProperties}>
        <LogPanel logs={visibleLogs} />
      </section>
    </div>
  );
}
