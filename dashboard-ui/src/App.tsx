import { startTransition, useEffect, useRef, useState, type MouseEvent } from "react";
import { dashboardSnapshot as mockSnapshot } from "./data/mockData";
import { DrawdownChart } from "./components/DrawdownChart";
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
  getDataSourceLabel,
  getRiskLevelLabel,
  getSignalDirectionLabel,
  getStrategyStatusLabel,
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
const MAX_LOG_ENTRIES = 12;

type NavPageId =
  | "overview"
  | "strategy"
  | "market"
  | "backtest"
  | "live"
  | "risk"
  | "account"
  | "settings";

type DashboardMetric = {
  label: string;
  value: string;
  change: string;
  helper: string;
  tone: "neutral" | "positive" | "negative" | "highlight";
};

type SummaryItem = {
  label: string;
  value: string;
  helper?: string;
  tone?: "positive" | "negative";
};

type SidebarItem = {
  id: NavPageId;
  label: string;
  icon: string;
  description: string;
};

const DEFAULT_PAGE: NavPageId = "overview";
const pageRouteMap: Record<NavPageId, string> = {
  overview: "/",
  strategy: "/strategy",
  market: "/market",
  backtest: "/backtest",
  live: "/live",
  risk: "/risk",
  account: "/account",
  settings: "/settings",
};

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
  return [entry, ...current].slice(0, MAX_LOG_ENTRIES);
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

function normalizePathname(pathname: string): string {
  const normalized = pathname.trim().replace(/\/+$/, "");
  if (!normalized) {
    return "/";
  }

  return normalized.startsWith("/") ? normalized : `/${normalized}`;
}

function parseHashRoute(hash: string): NavPageId | null {
  const route = hash.replace(/^#\/?/, "").trim().toLowerCase();
  switch (route) {
    case "":
    case "overview":
      return "overview";
    case "strategy":
    case "market":
    case "backtest":
    case "live":
    case "risk":
    case "account":
    case "settings":
      return route;
    default:
      return null;
  }
}

function parsePathRoute(pathname: string): NavPageId | null {
  switch (normalizePathname(pathname).toLowerCase()) {
    case "/":
    case "/overview":
      return "overview";
    case "/strategy":
      return "strategy";
    case "/market":
      return "market";
    case "/backtest":
      return "backtest";
    case "/live":
      return "live";
    case "/risk":
      return "risk";
    case "/account":
      return "account";
    case "/settings":
      return "settings";
    default:
      return null;
  }
}

function buildPathRoute(page: NavPageId): string {
  return pageRouteMap[page];
}

function getInitialPage(): NavPageId {
  if (typeof window === "undefined") {
    return DEFAULT_PAGE;
  }

  return parseHashRoute(window.location.hash) ?? parsePathRoute(window.location.pathname) ?? DEFAULT_PAGE;
}

const sidebarItems: SidebarItem[] = [
  { id: "overview", label: "总览", icon: "⌂", description: "主控视图" },
  { id: "strategy", label: "策略中心", icon: "◫", description: "信号与研究" },
  { id: "market", label: "行情监控", icon: "⌁", description: "K 线与盘口" },
  { id: "backtest", label: "回测分析", icon: "◌", description: "收益与回撤" },
  { id: "live", label: "实盘交易", icon: "◎", description: "持仓与成交" },
  { id: "risk", label: "风险控制", icon: "◍", description: "风险阈值" },
  { id: "account", label: "账户管理", icon: "◪", description: "资产与敞口" },
  { id: "settings", label: "系统设置", icon: "⚙", description: "参数与运行态" },
];

const pageMeta: Record<NavPageId, { kicker: string; title: string; description: string }> = {
  overview: {
    kicker: "Overview",
    title: "交易总览",
    description: "把最常看的状态集中到一个入口：市场主图、当前信号、核心仓位和最近执行。",
  },
  strategy: {
    kicker: "Strategy Center",
    title: "策略中心",
    description: "聚焦信号输出、策略健康度和研究侧的稳定性，而不是把执行和监控都挤在同一页。",
  },
  market: {
    kicker: "Market Monitor",
    title: "行情监控",
    description: "单独观察市场主图、盘口深度和短时结构，避免行情判断被其它业务面板打断。",
  },
  backtest: {
    kicker: "Backtest Review",
    title: "回测分析",
    description: "这里专门看回撤压力、研究收益质量和成交样本，保留复盘视角，不再塞进首页。",
  },
  live: {
    kicker: "Live Trading",
    title: "实盘交易",
    description: "把持仓、成交和运行日志汇总到执行页，方便盯盘时连续观察状态演进。",
  },
  risk: {
    kicker: "Risk Control",
    title: "风险控制",
    description: "集中展示杠杆、敞口、日内亏损使用率和回撤轨迹，减少风控信息分散。",
  },
  account: {
    kicker: "Account",
    title: "账户管理",
    description: "把权益、收益、名义敞口和账户状态收束成账户页，便于看资金层面的真实状态。",
  },
  settings: {
    kicker: "System Settings",
    title: "系统设置",
    description: "参数修改、运行态脉冲和运行环境摘要统一归档到设置页，操作边界更清晰。",
  },
};

function SummaryPanel({
  kicker,
  title,
  description,
  items,
  badge = "摘要",
}: {
  kicker: string;
  title: string;
  description: string;
  items: SummaryItem[];
  badge?: string;
}) {
  return (
    <section className="terminal-panel">
      <div className="flex items-start justify-between gap-4">
        <h2 className="panel-title">{title}</h2>
        <span className="panel-chip">{badge}</span>
      </div>

      <div className="page-summary-grid mt-6">
        {items.map((item) => (
          <article key={item.label} className="page-summary-card">
            <p>{item.label}</p>
            <strong className={item.tone ? `is-${item.tone}` : ""}>{item.value}</strong>
            {item.helper ? <span>{item.helper}</span> : null}
          </article>
        ))}
      </div>
    </section>
  );
}

export default function App() {
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);
  const [now, setNow] = useState<Date>(new Date());
  const [currentPage, setCurrentPage] = useState<NavPageId>(getInitialPage);
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
    if (typeof window === "undefined") {
      return undefined;
    }

    const syncRoute = () => {
      const legacyHashPage = parseHashRoute(window.location.hash);
      if (legacyHashPage) {
        window.history.replaceState(null, "", `${buildPathRoute(legacyHashPage)}${window.location.search}`);
        setCurrentPage(legacyHashPage);
        return;
      }

      const nextPage = parsePathRoute(window.location.pathname);
      if (nextPage) {
        setCurrentPage(nextPage);
        return;
      }

      window.history.replaceState(null, "", `${buildPathRoute(DEFAULT_PAGE)}${window.location.search}`);
      setCurrentPage(DEFAULT_PAGE);
    };

    syncRoute();
    window.addEventListener("popstate", syncRoute);
    window.addEventListener("hashchange", syncRoute);
    return () => {
      window.removeEventListener("popstate", syncRoute);
      window.removeEventListener("hashchange", syncRoute);
    };
  }, []);

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

  const drawdownRange: TimeRange = "30D";
  const drawdownCurve = filterSeriesByRange(drawdownRange, snapshot.equityCurve);
  const visibleLogs = [...localLogs, ...snapshot.logs].slice(0, 12);
  const positionMetricText = describePositionState(
    snapshot.metrics.positionMode,
    snapshot.metrics.netPositionQty,
    snapshot.metrics.positionNotional,
  );

  const metricCards: DashboardMetric[] = [
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

  const activePage = sidebarItems.find((item) => item.id === currentPage) ?? sidebarItems[0];
  const activePageInfo = pageMeta[currentPage];
  const latestCandle = snapshot.marketChart.candles[snapshot.marketChart.candles.length - 1] ?? null;
  const previousCandle = snapshot.marketChart.candles[snapshot.marketChart.candles.length - 2] ?? latestCandle;
  const marketDelta = latestCandle && previousCandle ? latestCandle.close - previousCandle.close : 0;
  const marketDeltaPct = latestCandle && previousCandle && previousCandle.close
    ? (marketDelta / previousCandle.close) * 100
    : 0;
  const bidDepth = snapshot.orderBook.bids.reduce((sum, level) => sum + level.total, 0);
  const askDepth = snapshot.orderBook.asks.reduce((sum, level) => sum + level.total, 0);
  const depthRatio = askDepth > 0 ? bidDepth / askDepth : 0;
  const dailyReturnPct = snapshot.metrics.dailyPnl / Math.max(snapshot.metrics.equity - snapshot.metrics.dailyPnl, 1) * 100;
  const recentLogs = visibleLogs.slice(0, 6);
  const overviewLogs = visibleLogs.slice(0, 4);

  const strategyMetricCards: DashboardMetric[] = [
    {
      label: "当前信号",
      value: getSignalDirectionLabel(snapshot.signal.direction),
      change: `评分 ${snapshot.signal.score}/100`,
      helper: "主模型与确认因子综合输出",
      tone: snapshot.signal.direction === "Long"
        ? "positive"
        : snapshot.signal.direction === "Short"
          ? "negative"
          : "highlight",
    },
    {
      label: "运行周期",
      value: snapshot.params.timeframe,
      change: `MA ${snapshot.params.maPeriod} / RSI ${snapshot.params.rsiPeriod}`,
      helper: "当前核心指标窗口",
      tone: "neutral",
    },
    {
      label: "止盈 / 止损",
      value: `${formatNumber(snapshot.params.takeProfitPct, 1)}% / ${formatNumber(snapshot.params.stopLossPct, 1)}%`,
      change: `ATR ${formatNumber(snapshot.params.atrMultiplier, 1)}x`,
      helper: "收益风险配比",
      tone: "highlight",
    },
    {
      label: "仓位与杠杆",
      value: `${formatNumber(snapshot.params.positionSizePct, 1)}%`,
      change: `最大 ${formatNumber(snapshot.params.maxLeverage, 1)}x`,
      helper: "执行层边界",
      tone: "neutral",
    },
  ];

  const marketMetricCards: DashboardMetric[] = [
    {
      label: "最新价格",
      value: latestCandle ? formatNumber(latestCandle.close, 2) : "--",
      change: `${marketDelta >= 0 ? "+" : ""}${formatNumber(marketDelta, 2)} / ${marketDeltaPct >= 0 ? "+" : ""}${formatNumber(marketDeltaPct, 2)}%`,
      helper: `${snapshot.marketChart.symbol} · ${snapshot.marketChart.timeframe}`,
      tone: marketDelta >= 0 ? "positive" : "negative",
    },
    {
      label: "盘口价差",
      value: formatNumber(snapshot.orderBook.spread, 2),
      change: formatPercent(snapshot.orderBook.spreadPct, 4, false),
      helper: "L2 深度即时快照",
      tone: "neutral",
    },
    {
      label: "买卖深度比",
      value: `${formatNumber(depthRatio, 2)}x`,
      change: `Bid ${formatNumber(bidDepth, 1)} / Ask ${formatNumber(askDepth, 1)}`,
      helper: "短时订单流倾向",
      tone: depthRatio >= 1 ? "positive" : "negative",
    },
    {
      label: "交易场所",
      value: snapshot.marketChart.venue,
      change: snapshot.marketChart.timeframe,
      helper: "当前监控图表来源",
      tone: "highlight",
    },
  ];

  const liveMetricCards: DashboardMetric[] = [
    {
      label: "活跃持仓",
      value: String(snapshot.metrics.openPositions),
      change: snapshot.metrics.openPositions > 0 ? "执行中" : "空仓待命",
      helper: positionMetricText.change,
      tone: snapshot.metrics.openPositions > 0 ? "positive" : "neutral",
    },
    {
      label: "名义敞口",
      value: positionMetricText.helper.replace("当前名义敞口 ", "").replace("总名义敞口 ", ""),
      change: snapshot.metrics.positionMode ? String(snapshot.metrics.positionMode).toUpperCase() : "FLAT",
      helper: "实时仓位暴露",
      tone: "highlight",
    },
    {
      label: "API / WS",
      value: `${getConnectionStatusLabel(snapshot.risk.apiStatus)} / ${getConnectionStatusLabel(snapshot.risk.wsStatus)}`,
      change: snapshot.risk.riskTriggered ? "风控已触发" : "执行链路正常",
      helper: "实盘连接健康度",
      tone: snapshot.risk.riskTriggered ? "negative" : "positive",
    },
    {
      label: "轮询节奏",
      value: `${POLL_MS / 1000}s`,
      change: getStrategyStatusLabel(snapshot.status),
      helper: "策略执行频率",
      tone: "neutral",
    },
  ];

  const riskMetricCards: DashboardMetric[] = [
    {
      label: "风险等级",
      value: getRiskLevelLabel(snapshot.metrics.riskLevel),
      change: snapshot.risk.riskTriggered ? "已进入限制态" : "监控正常",
      helper: "综合风控判定",
      tone: snapshot.risk.riskTriggered ? "negative" : "highlight",
    },
    {
      label: "风险暴露",
      value: formatPercent(snapshot.risk.marginUsagePct, 1, false),
      change: `当前杠杆 ${formatNumber(snapshot.risk.currentLeverage, 1)}x`,
      helper: "保证金占用率",
      tone: snapshot.risk.marginUsagePct > 60 ? "negative" : "neutral",
    },
    {
      label: "日内亏损使用",
      value: formatPercent(snapshot.risk.dailyLossUsedPct, 1, false),
      change: `上限 ${formatPercent(snapshot.risk.dailyLossLimitPct, 1, false)}`,
      helper: "日损阈值消耗",
      tone: snapshot.risk.dailyLossUsedPct > snapshot.risk.dailyLossLimitPct * 0.7 ? "negative" : "highlight",
    },
    {
      label: "单笔风险",
      value: formatPercent(snapshot.risk.maxLossPerTradePct, 1, false),
      change: "每笔交易预算",
      helper: "避免单次失控",
      tone: "neutral",
    },
  ];

  const accountMetricCards: DashboardMetric[] = [
    metricCards[0],
    metricCards[1],
    {
      label: "账户模式",
      value: snapshot.metrics.openPositions > 0 ? positionMetricText.change : "空仓待命",
      change: getStrategyStatusLabel(snapshot.status),
      helper: "当前仓位状态",
      tone: snapshot.metrics.openPositions > 0 ? "highlight" : "neutral",
    },
    {
      label: "风控等级",
      value: getRiskLevelLabel(snapshot.metrics.riskLevel),
      change: `最大回撤 ${formatPercent(snapshot.metrics.maxDrawdownPct, 2, false)}`,
      helper: "资金压力快照",
      tone: snapshot.metrics.riskLevel === "High" ? "negative" : "neutral",
    },
  ];

  const settingsMetricCards: DashboardMetric[] = [
    {
      label: "交易所",
      value: snapshot.exchange,
      change: snapshot.marketChart.symbol,
      helper: "当前连接市场",
      tone: "neutral",
    },
    {
      label: "系统状态",
      value: getStrategyStatusLabel(snapshot.status),
      change: getDataSourceLabel(snapshot.dataSource),
      helper: "运行模式与数据来源",
      tone: snapshot.status === "Error" ? "negative" : "highlight",
    },
    {
      label: "最近更新",
      value: formatDateTime(snapshot.updatedAt),
      change: `轮询 ${POLL_MS / 1000}s`,
      helper: "页面最近同步时间",
      tone: "neutral",
    },
    {
      label: "连接健康",
      value: `${getConnectionStatusLabel(snapshot.risk.apiStatus)} / ${getConnectionStatusLabel(snapshot.risk.wsStatus)}`,
      change: snapshot.risk.riskTriggered ? "风控保护中" : "运行稳定",
      helper: "网络与执行链路",
      tone: snapshot.risk.riskTriggered ? "negative" : "positive",
    },
  ];

  const strategySummaryItems: SummaryItem[] = [
    {
      label: "策略状态",
      value: getStrategyStatusLabel(snapshot.status),
      helper: `最近信号 ${formatDateTime(snapshot.signal.lastTriggeredAt)}`,
    },
    {
      label: "信号来源",
      value: snapshot.signal.sources.join(" · ") || "暂无",
      helper: "当前用于触发的确认因子",
    },
    {
      label: "累计收益率",
      value: formatPercent(snapshot.metrics.totalReturnPct),
      helper: `夏普 ${formatNumber(snapshot.metrics.sharpeRatio, 2)} / 胜率 ${formatPercent(snapshot.metrics.winRatePct, 2, false)}`,
      tone: snapshot.metrics.totalReturnPct >= 0 ? "positive" : "negative",
    },
    {
      label: "策略节奏",
      value: `${POLL_MS / 1000}s`,
      helper: `下次运行前 ${Math.max(0, Math.floor((new Date(snapshot.signal.nextRunAt).getTime() - now.getTime()) / 1000))} 秒`,
    },
  ];

  const marketSummaryItems: SummaryItem[] = [
    {
      label: "市场结构",
      value: `${snapshot.marketChart.symbol} · ${snapshot.marketChart.timeframe}`,
      helper: snapshot.marketChart.venue,
    },
    {
      label: "成交方向",
      value: `${marketDelta >= 0 ? "+" : ""}${formatNumber(marketDeltaPct, 2)}%`,
      helper: "相对上一根已处理 K 线变化",
      tone: marketDelta >= 0 ? "positive" : "negative",
    },
    {
      label: "盘口厚度",
      value: `${formatNumber(bidDepth, 1)} / ${formatNumber(askDepth, 1)}`,
      helper: "Bid / Ask 累计量",
    },
    {
      label: "订单流倾向",
      value: `${formatNumber(depthRatio, 2)}x`,
      helper: "大于 1 更偏买盘",
      tone: depthRatio >= 1 ? "positive" : "negative",
    },
  ];

  const backtestSummaryItems: SummaryItem[] = [
    {
      label: "累计收益",
      value: formatPercent(snapshot.metrics.totalReturnPct),
      helper: "研究期收益快照",
      tone: snapshot.metrics.totalReturnPct >= 0 ? "positive" : "negative",
    },
    {
      label: "最大回撤",
      value: formatPercent(snapshot.metrics.maxDrawdownPct, 2, false),
      helper: "资金谷底深度",
      tone: "negative",
    },
    {
      label: "收益质量",
      value: `Sharpe ${formatNumber(snapshot.metrics.sharpeRatio, 2)}`,
      helper: `胜率 ${formatPercent(snapshot.metrics.winRatePct, 2, false)}`,
    },
    {
      label: "交易样本",
      value: `${snapshot.trades.length} 条`,
      helper: "最近映射到研究面板的成交记录",
    },
  ];

  const liveSummaryItems: SummaryItem[] = [
    {
      label: "执行状态",
      value: getStrategyStatusLabel(snapshot.status),
      helper: `数据源 ${getDataSourceLabel(snapshot.dataSource)}`,
    },
    {
      label: "当前持仓",
      value: snapshot.metrics.openPositions > 0 ? `${snapshot.metrics.openPositions} 笔` : "空仓",
      helper: positionMetricText.change,
      tone: snapshot.metrics.openPositions > 0 ? "positive" : undefined,
    },
    {
      label: "名义敞口",
      value: positionMetricText.helper.replace("当前名义敞口 ", "").replace("总名义敞口 ", ""),
      helper: "当前执行仓位暴露",
    },
    {
      label: "链路健康",
      value: `API ${getConnectionStatusLabel(snapshot.risk.apiStatus)} / WS ${getConnectionStatusLabel(snapshot.risk.wsStatus)}`,
      helper: "交易与行情连接状态",
      tone: snapshot.risk.riskTriggered ? "negative" : "positive",
    },
  ];

  const riskSummaryItems: SummaryItem[] = [
    {
      label: "风险暴露",
      value: formatPercent(snapshot.risk.marginUsagePct, 1, false),
      helper: `当前杠杆 ${formatNumber(snapshot.risk.currentLeverage, 1)}x`,
      tone: snapshot.risk.marginUsagePct > 60 ? "negative" : undefined,
    },
    {
      label: "日内亏损使用",
      value: formatPercent(snapshot.risk.dailyLossUsedPct, 1, false),
      helper: `上限 ${formatPercent(snapshot.risk.dailyLossLimitPct, 1, false)}`,
      tone: snapshot.risk.dailyLossUsedPct > snapshot.risk.dailyLossLimitPct * 0.7 ? "negative" : undefined,
    },
    {
      label: "单笔风险预算",
      value: formatPercent(snapshot.risk.maxLossPerTradePct, 1, false),
      helper: "每笔交易最大允许亏损",
    },
    {
      label: "风控状态",
      value: snapshot.risk.riskTriggered ? "已触发" : "正常",
      helper: "是否进入限制态",
      tone: snapshot.risk.riskTriggered ? "negative" : "positive",
    },
  ];

  const accountSummaryItems: SummaryItem[] = [
    {
      label: "账户净值",
      value: formatCurrency(snapshot.metrics.equity).replace("$", ""),
      helper: "USDT 计价总资产",
    },
    {
      label: "当日收益",
      value: formatCurrency(snapshot.metrics.dailyPnl),
      helper: formatPercent(dailyReturnPct, 2),
      tone: snapshot.metrics.dailyPnl >= 0 ? "positive" : "negative",
    },
    {
      label: "账户模式",
      value: snapshot.metrics.openPositions > 0 ? positionMetricText.change : "空仓待命",
      helper: positionMetricText.helper,
    },
    {
      label: "风险概览",
      value: getRiskLevelLabel(snapshot.metrics.riskLevel),
      helper: `最大回撤 ${formatPercent(snapshot.metrics.maxDrawdownPct, 2, false)}`,
    },
  ];

  const settingsSummaryItems: SummaryItem[] = [
    {
      label: "运行环境",
      value: snapshot.exchange,
      helper: `${snapshot.productName} / ${snapshot.strategyName}`,
    },
    {
      label: "策略状态",
      value: getStrategyStatusLabel(snapshot.status),
      helper: `最近更新 ${formatDateTime(snapshot.updatedAt)}`,
    },
    {
      label: "数据来源",
      value: getDataSourceLabel(snapshot.dataSource),
      helper: "实时接口与本地回退自动切换",
    },
    {
      label: "网络健康",
      value: `API ${getConnectionStatusLabel(snapshot.risk.apiStatus)}`,
      helper: `WS ${getConnectionStatusLabel(snapshot.risk.wsStatus)}`,
      tone: snapshot.risk.riskTriggered ? "negative" : "positive",
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

  function navigateToPage(page: NavPageId) {
    if (typeof window === "undefined") {
      setCurrentPage(page);
      return;
    }

    const nextPath = buildPathRoute(page);
    const currentPath = normalizePathname(window.location.pathname);
    const targetPath = normalizePathname(nextPath);

    if (currentPath !== targetPath || window.location.hash) {
      window.history.pushState(null, "", `${nextPath}${window.location.search}`);
    }

    setCurrentPage(page);
  }

  function handleSidebarItemClick(event: MouseEvent<HTMLAnchorElement>, page: NavPageId) {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
      return;
    }

    event.preventDefault();
    navigateToPage(page);
  }

  function renderMetricGrid(cards: DashboardMetric[], columnsClassName: string) {
    return (
      <section className={`grid gap-4 ${columnsClassName}`}>
        {cards.map((metric, index) => (
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
    );
  }

  function renderPageContent() {
    switch (currentPage) {
      case "strategy":
        return (
          <>
            {renderMetricGrid(strategyMetricCards, "md:grid-cols-2 2xl:grid-cols-4")}
            <section className="dashboard-hero-grid mt-6">
              <SignalPanel signal={snapshot.signal} now={now} />
              <SummaryPanel
                kicker="策略摘要"
                title="策略运行轮廓"
                description="把信号、来源、收益质量和调度节奏单独聚合，避免策略页被执行细节挤占。"
                items={strategySummaryItems}
                badge="Core"
              />
            </section>
            <section className="dashboard-analysis-grid mt-6">
              <DrawdownChart data={drawdownCurve} range={drawdownRange} />
              <RiskPanel risk={snapshot.risk} />
            </section>
          </>
        );
      case "market":
        return (
          <>
            {renderMetricGrid(marketMetricCards, "md:grid-cols-2 2xl:grid-cols-4")}
            <section className="dashboard-hero-grid mt-6">
              <MarketChartPanel marketChart={snapshot.marketChart} />
              <OrderBookPanel orderBook={snapshot.orderBook} symbol={snapshot.marketChart.symbol} />
            </section>
            <section className="dashboard-dual-grid mt-6">
              <SummaryPanel
                kicker="盘口结构"
                title="市场微观状态"
                description="用价格、深度和订单流倾向来辅助判断当前信号是否有足够的市场支持。"
                items={marketSummaryItems}
                badge="Depth"
              />
              <SignalPanel signal={snapshot.signal} now={now} />
            </section>
          </>
        );
      case "backtest":
        return (
          <>
            {renderMetricGrid([metricCards[1], metricCards[2], metricCards[3], metricCards[4]], "md:grid-cols-2 2xl:grid-cols-4")}
            <section className="dashboard-analysis-grid mt-6">
              <DrawdownChart data={drawdownCurve} range={drawdownRange} />
              <SummaryPanel
                kicker="研究结论"
                title="回测表现摘要"
                description="这里只保留复盘最常用的维度：收益、回撤、收益质量和最近成交样本。"
                items={backtestSummaryItems}
                badge="Review"
              />
            </section>
            <section className="mt-6">
              <TradesTable trades={snapshot.trades} />
            </section>
          </>
        );
      case "live":
        return (
          <>
            {renderMetricGrid(liveMetricCards, "md:grid-cols-2 2xl:grid-cols-4")}
            <section className="dashboard-hero-grid mt-6">
              <PositionsTable positions={snapshot.positions} />
              <div className="dashboard-right-rail">
                <SignalPanel signal={snapshot.signal} now={now} />
                <SummaryPanel
                  kicker="执行摘要"
                  title="实盘执行面"
                  description="持仓、链路健康和名义敞口放在同一页，方便盯盘时直接判断是否存在执行异常。"
                  items={liveSummaryItems}
                  badge="Execution"
                />
              </div>
            </section>
            <section className="dashboard-dual-grid mt-6">
              <TradesTable trades={snapshot.trades} />
              <LogPanel logs={recentLogs} />
            </section>
          </>
        );
      case "risk":
        return (
          <>
            {renderMetricGrid(riskMetricCards, "md:grid-cols-2 2xl:grid-cols-4")}
            <section className="dashboard-analysis-grid mt-6">
              <DrawdownChart data={drawdownCurve} range={drawdownRange} />
              <RiskPanel risk={snapshot.risk} />
            </section>
            <section className="dashboard-dual-grid mt-6">
              <SummaryPanel
                kicker="风控边界"
                title="风险阈值摘要"
                description="聚合杠杆、日损、单笔风险和保护状态，便于确认策略是否接近失控边缘。"
                items={riskSummaryItems}
                badge="Guard"
              />
              <LogPanel logs={overviewLogs} />
            </section>
          </>
        );
      case "account":
        return (
          <>
            {renderMetricGrid(accountMetricCards, "md:grid-cols-2 2xl:grid-cols-4")}
            <section className="dashboard-dual-grid mt-6">
              <SummaryPanel
                kicker="账户总览"
                title="资产与敞口"
                description="把账户净值、收益、账户模式与风险概览集中到一起，专门看资金层面的真实状态。"
                items={accountSummaryItems}
                badge="Account"
              />
              <PositionsTable positions={snapshot.positions} />
            </section>
            <section className="mt-6">
              <TradesTable trades={snapshot.trades} />
            </section>
          </>
        );
      case "settings":
        return (
          <>
            {renderMetricGrid(settingsMetricCards, "md:grid-cols-2 2xl:grid-cols-4")}
            <section className="dashboard-hero-grid mt-6">
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
              <div className="dashboard-right-rail">
                <SystemPulsePanel systemPulse={snapshot.systemPulse} />
                <SummaryPanel
                  kicker="运行环境"
                  title="系统设置摘要"
                  description="参数修改、连接状态和运行环境集中归档，避免和交易执行页混在一起。"
                  items={settingsSummaryItems}
                  badge="Runtime"
                />
              </div>
            </section>
            <section className="mt-6">
              <LogPanel logs={recentLogs} />
            </section>
          </>
        );
      case "overview":
      default:
        return (
          <>
            {renderMetricGrid(metricCards, "md:grid-cols-2 2xl:grid-cols-6")}
            <section className="dashboard-hero-grid mt-6">
              <div className="min-w-0 space-y-6">
                <MarketChartPanel marketChart={snapshot.marketChart} />
                <PositionsTable positions={snapshot.positions} />
              </div>
              <div className="dashboard-right-rail">
                <SignalPanel signal={snapshot.signal} now={now} />
                <RiskPanel risk={snapshot.risk} />
              </div>
            </section>
            <section className="dashboard-dual-grid mt-6">
              <TradesTable trades={snapshot.trades} />
              <LogPanel logs={overviewLogs} />
            </section>
          </>
        );
    }
  }

  return (
    <div className="cockpit-shell">
      <aside className="cockpit-sidebar">
        <div className="cockpit-sidebar-stack">
          <div className="cockpit-sidebar-section cockpit-brand">
            <div className="cockpit-brand-mark">◫</div>
            <div>
              <p className="cockpit-brand-kicker">量化交易控制台</p>
              <h1 className="cockpit-brand-title">{snapshot.strategyName}</h1>
            </div>
          </div>

          <nav className="cockpit-sidebar-section cockpit-nav">
            {sidebarItems.map((item) => (
              <a
                key={item.id}
                href={buildPathRoute(item.id)}
                className={`cockpit-nav-item ${item.id === currentPage ? "is-active" : ""}`}
                onClick={(event) => handleSidebarItemClick(event, item.id)}
                aria-current={item.id === currentPage ? "page" : undefined}
              >
                <span className="cockpit-nav-icon">{item.icon}</span>
                <span className="cockpit-nav-copy">
                  <span>{item.label}</span>
                  <span className="cockpit-nav-meta">{item.description}</span>
                </span>
              </a>
            ))}
          </nav>

          <section className="cockpit-sidebar-section cockpit-sidebar-panel">
            <div className="flex items-center justify-between">
              <p className="panel-kicker">系统状态</p>
              <span className="panel-chip">实时</span>
            </div>
            <div className="mt-4 space-y-3">
              {systemStatusItems.map((item) => (
                <div key={item.label} className="flex items-center justify-between text-sm">
                  <span className="text-slate-400">{item.label}</span>
                  <span className="flex items-center gap-2 font-medium text-slate-100">
                    <span className={`h-2.5 w-2.5 rounded-full ${item.tone === "ok" ? "bg-up" : "bg-amber-400"}`} />
                    {item.value}
                  </span>
                </div>
              ))}
            </div>
            <div className="mt-5 surface-2 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-500">运行摘要</p>
              <p className="mt-2 text-sm text-slate-300">
                当前页 {activePage.label} · 风险 {getRiskLevelLabel(snapshot.metrics.riskLevel)} · 持仓 {snapshot.metrics.openPositions}
              </p>
              <p className="mt-2 text-xs text-slate-500">
                最近更新 {formatDateTime(snapshot.updatedAt)} · 轮询 {POLL_MS / 1000}s
              </p>
            </div>
          </section>
        </div>
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

        <section className="page-banner">
          <div className="page-banner-grid">
            <div className="page-banner-copy">
              <h2 className="page-banner-title">{activePageInfo.title}</h2>
              <div className="page-banner-meta">
                <span className="panel-pill">{snapshot.marketChart.symbol}</span>
                <span className="panel-pill">{getDataSourceLabel(snapshot.dataSource)}</span>
              </div>
            </div>
          </div>
        </section>

        <div className="mt-6">
          {renderPageContent()}
        </div>
      </main>
    </div>
  );
}
