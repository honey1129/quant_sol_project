import { dashboardSnapshot as mockSnapshot } from "../data/mockData";
import type {
  ApiDashboardBundle,
  ApiDashboardHistoryPoint,
  DashboardSnapshot,
  DataSource,
  EquityPoint,
  LogEntry,
  LogLevel,
  PositionRow,
  RiskLevel,
  RiskSnapshot,
  SignalDirection,
  StrategyParams,
  StrategyStatus,
  TradeRow,
} from "../types";

const ONE_DAY_MS = 24 * 60 * 60 * 1000;

function toNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function toIsoString(value: unknown, fallback: string): string {
  if (typeof value !== "string" || !value) {
    return fallback;
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? fallback : date.toISOString();
}

function parseLogTimestamp(value: string, fallback: string): string {
  const match = value.match(/^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})/);
  if (!match) {
    return fallback;
  }
  const iso = match[1].replace(" ", "T").replace(",", ".");
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? fallback : date.toISOString();
}

function stripLogPrefix(value: string): string {
  const match = value.match(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - [A-Z]+ - (.*)$/);
  return match ? match[1] : value;
}

function normalizeStatus(value?: string): StrategyStatus {
  if (value === "error") {
    return "Error";
  }
  if (value === "paused") {
    return "Paused";
  }
  return "Running";
}

function normalizeSignalDirection(value: string | null | undefined): SignalDirection {
  if (value === "Long" || value === "Short" || value === "Neutral") {
    return value;
  }
  return "Neutral";
}

function deriveSignalDirection(longProb: number | null, shortProb: number | null): SignalDirection {
  if (longProb === null || shortProb === null) {
    return "Neutral";
  }
  if (longProb - shortProb > 0.06) {
    return "Long";
  }
  if (shortProb - longProb > 0.06) {
    return "Short";
  }
  return "Neutral";
}

function detectLogLevel(message: string): LogLevel {
  if (/异常|ERROR|失败|未成交|disconnect|lag/i.test(message)) {
    return "ERROR";
  }
  if (/心跳|等待|跳过|warn/i.test(message)) {
    return "WARN";
  }
  if (/执行|started|ready|成功|filled|running/i.test(message)) {
    return "SUCCESS";
  }
  return "INFO";
}

function extractBaseAsset(symbol: string): string {
  if (!symbol) {
    return "资产";
  }
  return symbol.split("-")[0]?.split("USDT")[0] || symbol;
}

function timeframeToMinutes(timeframe: string): number {
  const match = timeframe.match(/^(\d+)(m|h|d)$/i);
  if (!match) {
    return 5;
  }
  const count = Number(match[1]);
  const unit = match[2].toLowerCase();
  if (unit === "m") {
    return count;
  }
  if (unit === "h") {
    return count * 60;
  }
  return count * 24 * 60;
}

function humanizeMinutes(totalMinutes: number): string {
  if (!Number.isFinite(totalMinutes) || totalMinutes <= 0) {
    return "0分钟";
  }
  const days = Math.floor(totalMinutes / (24 * 60));
  const hours = Math.floor((totalMinutes % (24 * 60)) / 60);
  const minutes = Math.floor(totalMinutes % 60);
  if (days > 0) {
    return `${days}天 ${hours}小时`;
  }
  if (hours > 0) {
    return `${hours}小时 ${minutes}分钟`;
  }
  return `${minutes}分钟`;
}

function computeUnrealizedPnl(direction: string, qty: number | null, entryPrice: number | null, currentPrice: number | null): number | null {
  if (qty === null || entryPrice === null || currentPrice === null) {
    return null;
  }
  if (direction === "short") {
    return (entryPrice - currentPrice) * Math.abs(qty);
  }
  return (currentPrice - entryPrice) * Math.abs(qty);
}

function computeApproxRiskBands(
  direction: string,
  entryPrice: number | null,
  params: StrategyParams,
): { stopLoss: number | null; takeProfit: number | null } {
  if (entryPrice === null) {
    return { stopLoss: null, takeProfit: null };
  }
  const stopLossFactor = params.stopLossPct / 100;
  const takeProfitFactor = params.takeProfitPct / 100;
  if (direction === "short") {
    return {
      stopLoss: entryPrice * (1 + stopLossFactor),
      takeProfit: entryPrice * (1 - takeProfitFactor),
    };
  }
  return {
    stopLoss: entryPrice * (1 - stopLossFactor),
    takeProfit: entryPrice * (1 + takeProfitFactor),
  };
}

function buildEquityCurve(
  history: ApiDashboardHistoryPoint[],
  fallbackCurve: EquityPoint[],
  currentEquity: number | null,
  currentPrice: number | null,
  updatedAt: string,
): EquityPoint[] {
  const normalizedHistory = history
    .map((point) => ({
      timestamp: toIsoString(point.bar_ts || point.timestamp, ""),
      totalEq: toNumber(point.total_eq),
      price: toNumber(point.price),
    }))
    .filter((point) => point.timestamp && point.totalEq !== null);

  if (normalizedHistory.length === 0) {
    if (currentEquity !== null) {
      return [
        {
          timestamp: updatedAt,
          equity: currentEquity,
          benchmark: currentPrice ?? currentEquity,
          drawdown: 0,
        },
      ];
    }
    return fallbackCurve;
  }

  const baseEquity = normalizedHistory[0].totalEq || 1;
  const basePrice = normalizedHistory.find((point) => point.price !== null)?.price;
  let rollingPeak = Number.NEGATIVE_INFINITY;

  return normalizedHistory.map((point) => {
    const equity = point.totalEq || 0;
    rollingPeak = Math.max(rollingPeak, equity);
    const benchmark = basePrice && point.price
      ? baseEquity * (point.price / basePrice)
      : equity;
    return {
      timestamp: point.timestamp,
      equity,
      benchmark,
      drawdown: rollingPeak > 0 ? ((equity - rollingPeak) / rollingPeak) * 100 : 0,
    };
  });
}

function deriveDailyPnlFromCurve(curve: EquityPoint[]): number {
  if (curve.length === 0) {
    return mockSnapshot.metrics.dailyPnl;
  }
  const latest = curve[curve.length - 1];
  const latestTs = new Date(latest.timestamp).getTime();
  const dailyAnchor = curve.find((point) => latestTs - new Date(point.timestamp).getTime() <= ONE_DAY_MS) || curve[0];
  return latest.equity - dailyAnchor.equity;
}

function deriveRiskLevel(exposurePct: number, leverage: number, status: StrategyStatus): RiskLevel {
  if (status === "Error" || exposurePct >= 70 || leverage >= 5) {
    return "High";
  }
  if (exposurePct >= 35 || leverage >= 3) {
    return "Medium";
  }
  return "Low";
}

function normalizeRiskLevel(value: unknown, fallback: RiskLevel): RiskLevel {
  if (value === "Low" || value === "Medium" || value === "High") {
    return value;
  }
  return fallback;
}

function normalizeApiStatus(value: unknown, fallback: RiskSnapshot["apiStatus"]): RiskSnapshot["apiStatus"] {
  if (value === "Connected" || value === "Degraded" || value === "Disconnected") {
    return value;
  }
  return fallback;
}

function normalizeWsStatus(value: unknown, fallback: RiskSnapshot["wsStatus"]): RiskSnapshot["wsStatus"] {
  if (value === "Connected" || value === "Lagging" || value === "Disconnected") {
    return value;
  }
  return fallback;
}

function normalizeTradeSide(value: unknown, fallback: TradeRow["side"]): TradeRow["side"] {
  if (value === "Long" || value === "Short") {
    return value;
  }
  return fallback;
}

function buildSignalSources(bundle: ApiDashboardBundle): string[] {
  if (bundle.signal_summary?.sources?.length) {
    return bundle.signal_summary.sources;
  }

  const sources: string[] = ["ML Model"];
  const signal = bundle.status?.signal;
  if (toNumber(signal?.money_flow_ratio) !== null) {
    sources.push("RSI");
  }
  if (toNumber(signal?.volatility) !== null) {
    sources.push("ATR");
  }
  if (toNumber(signal?.atr_ratio) !== null) {
    sources.push("MACD");
  }
  return Array.from(new Set(sources));
}

function buildStrategyParams(bundle: ApiDashboardBundle): StrategyParams {
  const liveParams = bundle.strategy_params;
  return {
    timeframe: liveParams?.timeframe || mockSnapshot.params.timeframe,
    maPeriod: toNumber(liveParams?.ma_period) ?? mockSnapshot.params.maPeriod,
    rsiPeriod: toNumber(liveParams?.rsi_period) ?? mockSnapshot.params.rsiPeriod,
    atrMultiplier: toNumber(liveParams?.atr_multiplier) ?? mockSnapshot.params.atrMultiplier,
    stopLossPct: toNumber(liveParams?.stop_loss_pct) ?? mockSnapshot.params.stopLossPct,
    takeProfitPct: toNumber(liveParams?.take_profit_pct) ?? mockSnapshot.params.takeProfitPct,
    positionSizePct: toNumber(liveParams?.position_size_pct) ?? mockSnapshot.params.positionSizePct,
    maxLeverage: toNumber(liveParams?.max_leverage) ?? mockSnapshot.params.maxLeverage,
  };
}

function buildPositionRows(bundle: ApiDashboardBundle, params: StrategyParams, fallbackRows: PositionRow[]): PositionRow[] {
  const market = bundle.status?.market;
  const position = bundle.status?.position;
  const direction = String(position?.direction || "flat").toLowerCase();
  const qty = toNumber(position?.net_qty);

  if (!bundle.status || !market?.symbol) {
    return fallbackRows;
  }
  if (direction === "flat" || qty === null || qty === 0) {
    return [];
  }

  const entryPrice = toNumber(position?.entry_price);
  const currentPrice = toNumber(market?.last_price);
  const leverage = toNumber(market?.leverage) ?? params.maxLeverage;
  const unrealizedPnl = computeUnrealizedPnl(direction, qty, entryPrice, currentPrice);
  const riskBands = computeApproxRiskBands(direction, entryPrice, params);
  const holdBars = toNumber(position?.hold_bars) ?? 0;
  const holdingMinutes = holdBars * timeframeToMinutes(params.timeframe);

  return [
    {
      symbol: market.symbol,
      direction: direction === "short" ? "Short" : "Long",
      entryPrice,
      currentPrice,
      positionSize: `${Math.abs(qty).toFixed(4)} ${extractBaseAsset(market.symbol)}`,
      leverage: `${leverage.toFixed(0)}x`,
      unrealizedPnl,
      stopLoss: riskBands.stopLoss,
      takeProfit: riskBands.takeProfit,
      holdingTime: humanizeMinutes(holdingMinutes),
    },
  ];
}

function buildTradeRows(bundle: ApiDashboardBundle): TradeRow[] {
  if (bundle.recent_trades?.length) {
    return bundle.recent_trades.map((trade) => ({
      time: toIsoString(trade.time, new Date().toISOString()),
      symbol: trade.symbol || bundle.status?.market?.symbol || "N/A",
      side: normalizeTradeSide(
        trade.side,
        deriveSignalDirection(
          toNumber(bundle.status?.signal?.long_prob),
          toNumber(bundle.status?.signal?.short_prob),
        ) === "Short" ? "Short" : "Long",
      ),
      entry: toNumber(trade.entry),
      exit: toNumber(trade.exit),
      pnl: toNumber(trade.pnl),
      fee: toNumber(trade.fee),
      slippage: toNumber(trade.slippage),
      reason: trade.reason || "最近一次执行",
      status: trade.status || "Filled",
    }));
  }

  const events = bundle.recent_events || [];
  const marketSymbol = bundle.status?.market?.symbol || "N/A";
  const signalDirection = deriveSignalDirection(
    toNumber(bundle.status?.signal?.long_prob),
    toNumber(bundle.status?.signal?.short_prob),
  );

  const rows = events
    .filter((line) => /执行开仓|执行平仓|执行调仓|未成交/.test(line))
    .map((line) => {
      const message = stripLogPrefix(line);
      const time = parseLogTimestamp(line, new Date().toISOString());
      return {
        time,
        symbol: marketSymbol,
        side: signalDirection === "Short" ? "Short" : "Long",
        entry: toNumber(bundle.status?.position?.entry_price),
        exit: /执行平仓/.test(message) ? toNumber(bundle.status?.market?.last_price) : null,
        pnl: null,
        fee: null,
        slippage: null,
        reason: message,
        status: /未成交/.test(message) ? "Canceled" : "Filled",
      } as TradeRow;
    });

  if (rows.length > 0) {
    return rows;
  }

  const lastExecution = bundle.status?.last_execution;
  if (lastExecution?.timestamp) {
    return [
      {
        time: toIsoString(lastExecution.timestamp, new Date().toISOString()),
        symbol: marketSymbol,
        side: signalDirection === "Short" ? "Short" : "Long",
        entry: toNumber(bundle.status?.position?.entry_price),
        exit: lastExecution.action === "CLOSE" ? toNumber(bundle.status?.market?.last_price) : null,
        pnl: null,
        fee: null,
        slippage: null,
        reason: lastExecution.reason || bundle.status?.decision?.reason || "最后一次执行",
        status: lastExecution.success === false ? "Canceled" : "Filled",
      },
    ];
  }

  return [];
}

function buildLogs(bundle: ApiDashboardBundle, fallbackLogs: LogEntry[]): LogEntry[] {
  const events = bundle.recent_events || [];
  if (events.length === 0) {
    if (bundle.status && Object.keys(bundle.status).length > 0) {
      return [];
    }
    return fallbackLogs;
  }

  return events
    .slice()
    .reverse()
    .map((event, index) => ({
      id: `api-log-${index}-${event.slice(0, 16)}`,
      time: parseLogTimestamp(event, bundle.generated_at || new Date().toISOString()),
      level: detectLogLevel(event),
      message: stripLogPrefix(event),
    }));
}

function buildDataSource(bundle: ApiDashboardBundle): DataSource {
  const hasStructuredLiveFields = Boolean(
    bundle.strategy_meta &&
    bundle.strategy_params &&
    bundle.metrics &&
    bundle.signal_summary &&
    bundle.risk_snapshot,
  );

  if (hasStructuredLiveFields) {
    return "live";
  }
  if (bundle.status && Object.keys(bundle.status).length > 0) {
    return "hybrid";
  }
  return "mock";
}

export function buildDashboardSnapshotFromApi(bundle: ApiDashboardBundle | null | undefined): DashboardSnapshot {
  if (!bundle || (!bundle.status && !bundle.strategy_meta && !bundle.metrics)) {
    return {
      ...mockSnapshot,
      updatedAt: new Date().toISOString(),
      dataSource: "mock",
    };
  }

  const dataSource = buildDataSource(bundle);
  const market = bundle.status?.market || {};
  const account = bundle.status?.account || {};
  const performance = bundle.status?.performance || {};
  const signal = bundle.status?.signal || {};
  const runtime = bundle.status?.runtime || {};
  const position = bundle.status?.position || {};
  const strategyMeta = bundle.strategy_meta;
  const liveMetrics = bundle.metrics;
  const liveRisk = bundle.risk_snapshot;
  const liveSignal = bundle.signal_summary;

  const updatedAt = toIsoString(
    bundle.generated_at || bundle.status?.updated_at,
    new Date().toISOString(),
  );

  const currentEquity = toNumber(liveMetrics?.equity)
    ?? toNumber(account.total_eq)
    ?? toNumber(performance.current_total_eq)
    ?? mockSnapshot.metrics.equity;
  const currentPrice = toNumber(market.last_price);
  const params = buildStrategyParams(bundle);
  const equityCurve = buildEquityCurve(bundle.history || [], mockSnapshot.equityCurve, currentEquity, currentPrice, updatedAt);
  const positions = buildPositionRows(bundle, params, []);
  const positionNotional = toNumber(position.notional);
  const exposurePct = currentEquity > 0 && positionNotional !== null ? (positionNotional / currentEquity) * 100 : 0;
  const leverage = toNumber(liveRisk?.current_leverage) ?? toNumber(market.leverage) ?? params.maxLeverage;
  const runtimeStatus = normalizeStatus(runtime.last_status);
  const derivedRiskLevel = deriveRiskLevel(exposurePct, leverage, runtimeStatus);
  const signalDirection = liveSignal?.direction
    ? normalizeSignalDirection(liveSignal.direction)
    : deriveSignalDirection(toNumber(signal.long_prob), toNumber(signal.short_prob));
  const dailyPnl = toNumber(liveMetrics?.daily_pnl) ?? deriveDailyPnlFromCurve(equityCurve);

  return {
    productName: strategyMeta?.product_name || "Quant Alpha 控制台",
    strategyName: strategyMeta?.strategy_name || (market.symbol ? `${market.symbol} 实盘策略控制台` : mockSnapshot.strategyName),
    exchange: strategyMeta?.exchange || market.exchange || mockSnapshot.exchange,
    status: runtimeStatus,
    updatedAt,
    dataSource,
    equityCurve,
    metrics: {
      equity: currentEquity,
      dailyPnl,
      totalReturnPct: toNumber(liveMetrics?.total_return_pct) ?? mockSnapshot.metrics.totalReturnPct,
      maxDrawdownPct: toNumber(liveMetrics?.max_drawdown_pct) ?? Math.min(...equityCurve.map((point) => point.drawdown)),
      sharpeRatio: toNumber(liveMetrics?.sharpe_ratio) ?? mockSnapshot.metrics.sharpeRatio,
      winRatePct: toNumber(liveMetrics?.win_rate_pct) ?? mockSnapshot.metrics.winRatePct,
      openPositions: toNumber(liveMetrics?.open_positions) ?? positions.length,
      riskLevel: normalizeRiskLevel(liveMetrics?.risk_level, derivedRiskLevel),
    },
    signal: {
      direction: signalDirection,
      sources: buildSignalSources(bundle),
      score: toNumber(liveSignal?.score) ?? (
        toNumber(signal.long_prob) !== null && toNumber(signal.short_prob) !== null
          ? Math.round(Math.max(toNumber(signal.long_prob) || 0, toNumber(signal.short_prob) || 0) * 100)
          : mockSnapshot.signal.score
      ),
      lastTriggeredAt: toIsoString(
        liveSignal?.last_triggered_at
          || bundle.status?.last_execution?.timestamp
          || bundle.status?.bar?.latest_closed_bar_ts,
        updatedAt,
      ),
      nextRunAt: toIsoString(
        liveSignal?.next_run_at,
        new Date(Date.now() + ((toNumber(runtime.poll_sec) ?? 10) * 1000)).toISOString(),
      ),
    },
    positions,
    trades: buildTradeRows(bundle),
    risk: {
      currentLeverage: leverage,
      maxLossPerTradePct: toNumber(liveRisk?.max_loss_per_trade_pct) ?? mockSnapshot.risk.maxLossPerTradePct,
      marginUsagePct: toNumber(liveRisk?.margin_usage_pct) ?? exposurePct,
      dailyLossLimitPct: toNumber(liveRisk?.daily_loss_limit_pct) ?? mockSnapshot.risk.dailyLossLimitPct,
      dailyLossUsedPct: toNumber(liveRisk?.daily_loss_used_pct) ?? (
        dailyPnl < 0 && currentEquity > 0 ? (Math.abs(dailyPnl) / currentEquity) * 100 : 0
      ),
      riskTriggered: Boolean(liveRisk?.risk_triggered) || runtimeStatus === "Error" || Boolean(runtime.last_error),
      apiStatus: normalizeApiStatus(liveRisk?.api_status, runtimeStatus === "Error" ? "Degraded" : "Connected"),
      wsStatus: normalizeWsStatus(liveRisk?.ws_status, runtimeStatus === "Error" ? "Lagging" : "Connected"),
    },
    params,
    logs: buildLogs(bundle, mockSnapshot.logs),
  };
}
