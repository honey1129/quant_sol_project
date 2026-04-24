import { startTransition, useEffect, useRef, useState } from "react";
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
          throw new Error(`Dashboard API returned ${response.status}`);
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
        const message = fetchError instanceof Error ? fetchError.message : "Failed to fetch /api/dashboard";
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

  const metricCards: Array<{
    label: string;
    value: string;
    change: string;
    helper: string;
    tone: "neutral" | "positive" | "negative" | "highlight";
  }> = [
    {
      label: "Equity",
      value: formatCurrency(snapshot.metrics.equity),
      change: formatCurrency(snapshot.metrics.dailyPnl),
      helper: "Current account equity",
      tone: "highlight" as const,
    },
    {
      label: "Daily PnL",
      value: formatCurrency(snapshot.metrics.dailyPnl),
      change: snapshot.metrics.dailyPnl >= 0 ? "Live session profitable" : "Session below target",
      helper: "PnL over the last 24h",
      tone: snapshot.metrics.dailyPnl >= 0 ? "positive" : "negative",
    },
    {
      label: "Total Return",
      value: formatPercent(snapshot.metrics.totalReturnPct),
      change: snapshot.dataSource === "mock" ? "Mock baseline" : "Derived from live equity history",
      helper: "Compounded total return",
      tone: "positive" as const,
    },
    {
      label: "Max Drawdown",
      value: formatPercent(snapshot.metrics.maxDrawdownPct, 2, false),
      change: "Worst observed capital dip",
      helper: "Historical trough from peak",
      tone: "negative" as const,
    },
    {
      label: "Sharpe Ratio",
      value: formatNumber(snapshot.metrics.sharpeRatio, 2),
      change: snapshot.dataSource === "live"
        ? "Derived from latest backtest research snapshot"
        : "Risk-adjusted performance",
      helper: "Annualized estimate",
      tone: "neutral" as const,
    },
    {
      label: "Win Rate",
      value: formatPercent(snapshot.metrics.winRatePct, 1, false),
      change: snapshot.dataSource === "live"
        ? "Derived from latest backtest closed trades"
        : "Filled trades only",
      helper: "Execution quality",
      tone: "positive" as const,
    },
    {
      label: "Open Positions",
      value: String(snapshot.metrics.openPositions),
      change: "Active derivatives exposure",
      helper: "Across strategy book",
      tone: "neutral" as const,
    },
    {
      label: "Risk Level",
      value: snapshot.metrics.riskLevel,
      change: "Based on leverage and margin usage",
      helper: "Current session classification",
      tone: snapshot.metrics.riskLevel === "High" ? "negative" : "highlight",
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
        throw new Error(result.error || `Strategy params API returned ${response.status}`);
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
        message: result.message || "Strategy parameters saved successfully.",
      };
      setLocalLogs((current) => prependLogEntry(current, entry));
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "Failed to save strategy params";
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
        throw new Error(result.error || `Restart API returned ${response.status}`);
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
        message: result.output ? `${result.message} ${result.output}` : (result.message || "Strategy restart requested."),
      };
      setLocalLogs((current) => prependLogEntry(current, entry));
    } catch (restartError) {
      const message = restartError instanceof Error ? restartError.message : "Failed to restart strategy";
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
          ? "Strategy parameters reset to the latest values exposed by /api/dashboard."
          : "Strategy parameters reset to baseline mock profile.",
      };
    setLocalLogs((current) => prependLogEntry(current, entry));
  }

  return (
    <div className="min-h-screen px-4 py-4 sm:px-6 xl:px-8">
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
        <div className="mb-6 rounded-2xl border border-amber-400/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          Live API fetch warning: {error}. The dashboard is keeping the last known state on screen.
        </div>
      ) : null}

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-8">
        {metricCards.map((metric) => (
          <MetricCard
            key={metric.label}
            label={metric.label}
            value={metric.value}
            change={metric.change}
            helper={metric.helper}
            tone={metric.tone}
          />
        ))}
      </section>

      <section className="mt-6 grid gap-6 xl:grid-cols-[1.55fr_0.95fr]">
        <div className="space-y-6">
          <EquityChart data={filteredCurve} range={range} onRangeChange={setRange} />
          <DrawdownChart data={filteredCurve} range={range} />
        </div>

        <div className="space-y-6">
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

      <section className="mt-6 grid gap-6 xl:grid-cols-[1.15fr_1fr]">
        <PositionsTable positions={snapshot.positions} />
        <TradesTable trades={snapshot.trades} />
      </section>

      <section className="mt-6">
        <LogPanel logs={visibleLogs} />
      </section>
    </div>
  );
}
