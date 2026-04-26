import { dashboardSnapshot as mockSnapshot } from "../data/mockData";
import type {
  ApiDashboardBundle,
  ApiDashboardHistoryPoint,
  DashboardSnapshot,
  DataSource,
  EquityPoint,
  LogEntry,
  LogLevel,
  MarketCandle,
  PositionRow,
  RiskLevel,
  RiskSnapshot,
  SignalDirection,
  StrategyParams,
  StrategyStatus,
  SystemPulse,
  TradeRow,
} from "../types";

const ONE_DAY_MS = 24 * 60 * 60 * 1000;
const DISPLAY_TIME_OFFSET = "+08:00";
const REASON_CODE_PATTERN = /reason=([A-Za-z0-9_().,-]+)/;
const TARGET_RATIO_PATTERN = /target_ratio=(-?\d+(?:\.\d+)?)/;
const DELTA_QTY_PATTERN = /delta_qty=(-?\d+(?:\.\d+)?)/;
const QTY_PATTERN = /qty=(-?\d+(?:\.\d+)?)/;
const MIN_HOLD_REASON_PATTERN = /MinHold\((\d+)\/(\d+)\)/;
const WEAK_REVERSE_REASON_PATTERN = /WeakReverseSignal\(gap=([0-9.]+),dominant=([0-9.]+),ratio=([0-9.]+)\)/;

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
  // Runtime log timestamps are rendered directly in Asia/Shanghai on the server.
  const iso = `${match[1].replace(" ", "T").replace(",", ".")}${DISPLAY_TIME_OFFSET}`;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? fallback : date.toISOString();
}

function stripLogPrefix(value: string): string {
  const match = value.match(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - [A-Z]+ - (.*)$/);
  return match ? match[1] : value;
}

function humanizeReasonCode(reason: string): string {
  const trimmed = String(reason || "").trim();
  if (!trimmed) {
    return "本次执行由策略最新状态驱动，未返回更具体的决策原因。";
  }

  const minHoldMatch = trimmed.match(MIN_HOLD_REASON_PATTERN);
  if (minHoldMatch) {
    return `当前仓位仍处于最小持有期内，为避免过早反复交易，系统选择继续持有（已持有 ${minHoldMatch[1]}/${minHoldMatch[2]} 根 K 线）。`;
  }

  const weakReverseMatch = trimmed.match(WEAK_REVERSE_REASON_PATTERN);
  if (weakReverseMatch) {
    return `检测到反向信号，但强度仍不足以直接反手，系统继续持仓等待更明确确认（概率差 ${weakReverseMatch[1]}，主导概率 ${weakReverseMatch[2]}，目标仓位比例 ${weakReverseMatch[3]}）。`;
  }

  const reasonMap: Record<string, string> = {
    OpenFromFlat: "空仓状态下出现满足阈值的有效信号，系统首次建立新仓位。",
    SameDirRebalance: "当前信号方向未变，但目标仓位与现有仓位存在明显偏差，系统按策略计划执行同方向调仓。",
    "TP/SL": "当前持仓已触发止盈或止损阈值，系统执行风险退出以锁定结果或控制回撤。",
    ReverseClose: "出现了足够强的反向信号，系统先平掉当前仓位，等待后续确认后再决定是否反手。",
    FlatNoSignal: "当前为空仓，且信号强度不足以支持新开仓，系统继续等待更清晰机会。",
    SameDirNoRebalance: "当前方向判断未变，但仓位偏差不足以触发调仓，系统保持现有持仓不动。",
    NoSignalKeep: "当前没有足够优势的开平仓信号，系统维持现有仓位并继续观察。",
    SameClosedBarSkip: "这根已收盘 K 线已经处理过，本轮仅更新状态并等待下一根新的已确认 K 线。",
    DualSidePosition: "检测到账户同时存在多仓和空仓，系统进入保护性对账流程，优先处理异常双向持仓。",
    LoopException: "主循环中出现异常，系统已记录错误并等待下一轮恢复执行。",
    MonitorBoot: "监控进程刚完成启动，当前展示的是启动后的首次状态快照。",
  };

  return reasonMap[trimmed] || trimmed;
}

function humanizeTradeReason(reason: string, fallbackMessage?: string): string {
  const raw = String(reason || fallbackMessage || "").trim();
  const baseMessage = fallbackMessage ? stripLogPrefix(fallbackMessage) : raw;
  const targetRatio = baseMessage.match(TARGET_RATIO_PATTERN)?.[1];
  const deltaQty = baseMessage.match(DELTA_QTY_PATTERN)?.[1];
  const qty = baseMessage.match(QTY_PATTERN)?.[1];
  const reasonCode = (baseMessage.match(REASON_CODE_PATTERN)?.[1] || raw).trim();
  const reasonText = humanizeReasonCode(reasonCode);

  if (/执行开仓/.test(baseMessage) || /开仓未成交/.test(baseMessage)) {
    const ratioText = targetRatio ? `目标仓位约 ${(Number(targetRatio) * 100).toFixed(1)}%` : "目标仓位按策略实时计算";
    const qtyText = qty ? `，计划下单数量 ${Number(qty).toFixed(4)}` : "";
    const statusText = /未成交/.test(baseMessage) ? "本次下单未成交，系统已回到仓位同步状态。" : "系统已据此发起开仓执行。";
    return `${reasonText}${ratioText}${qtyText}。${statusText}`;
  }

  if (/执行调仓/.test(baseMessage) || /调仓未成交/.test(baseMessage)) {
    const deltaText = deltaQty
      ? `本次${Number(deltaQty) > 0 ? "增加" : "减少"}仓位 ${Math.abs(Number(deltaQty)).toFixed(4)}`
      : "本次调仓数量按目标仓位偏差动态计算";
    const statusText = /未成交/.test(baseMessage) ? "调仓委托未成交，系统已重新同步当前仓位。" : "系统已执行本次仓位调整。";
    return `${reasonText}${deltaText}。${statusText}`;
  }

  if (/执行平仓/.test(baseMessage) || /平仓未成交/.test(baseMessage)) {
    const statusText = /未成交/.test(baseMessage) ? "平仓委托未成交，系统已重新同步当前持仓。" : "系统已按规则执行平仓退出。";
    return `${reasonText}${statusText}`;
  }

  return reasonText;
}

function humanizeRuntimeLogMessage(message: string): string {
  const text = stripLogPrefix(String(message || "").trim());

  if (/^已恢复最近处理 bar:/.test(text)) {
    return text.replace(/^已恢复最近处理 bar:\s*/, "系统已恢复上次处理到的 K 线位置：");
  }

  if (/^reward_risk=/.test(text)) {
    const value = text.split("=")[1];
    return `仓位管理模块已刷新 reward/risk 参数，当前使用值为 ${value}。`;
  }

  if (/^新bar=/.test(text)) {
    const bar = text.match(/新bar=([^\s]+)/)?.[1] || "--";
    const price = text.match(/price=(-?\d+(?:\.\d+)?)/)?.[1];
    const longProb = text.match(/long=(-?\d+(?:\.\d+)?)/)?.[1];
    const shortProb = text.match(/short=(-?\d+(?:\.\d+)?)/)?.[1];
    const mf = text.match(/mf=(-?\d+(?:\.\d+)?)/)?.[1];
    const vol = text.match(/vol=(-?\d+(?:\.\d+)?)/)?.[1];
    const atr = text.match(/atr_ratio=(-?\d+(?:\.\d+)?%)/)?.[1];
    return `新一根已确认收盘 K 线到达：bar=${bar}，价格 ${price || "--"}，多头概率 ${longProb || "--"}，空头概率 ${shortProb || "--"}，资金流比 ${mf || "--"}，波动率 ${vol || "--"}，ATR 比率 ${atr || "--"}。`;
  }

  if (/^心跳:/.test(text)) {
    return text
      .replace(/^心跳:\s*运行中，/, "系统心跳正常，")
      .replace("最近已处理bar=", "最近已处理 K 线：")
      .replace("当前最新已收盘bar=", "当前最新已收盘 K 线：")
      .replace("连续跳过同bar次数=", "连续跳过同一根 K 线次数：");
  }

  if (/执行开仓|开仓未成交|执行调仓|调仓未成交|执行平仓|平仓未成交/.test(text)) {
    return humanizeTradeReason(text, text);
  }

  if (text === "无明显信号或目标为0：保持仓位不变") {
    return "当前没有足够强的有效信号，或目标仓位接近 0，系统继续保持现有仓位不变。";
  }

  if (text === "检测到同时多空持仓，进入恢复流程，只做清仓对账，不执行新信号。") {
    return "检测到账户同时存在多仓和空仓，系统进入异常恢复流程，只执行对账与清仓保护，不再发出新信号。";
  }

  if (text === "双边持仓仍未清理完成，本轮跳过信号执行。") {
    return "双向持仓异常尚未完全清理，本轮为保护账户安全，系统跳过新的交易信号执行。";
  }

  if (text.startsWith("启动快照初始化失败")) {
    return text.replace("启动快照初始化失败，将继续进入主循环:", "启动阶段快照初始化失败，但主循环仍继续运行，错误详情：");
  }

  if (text.startsWith("实盘循环异常:")) {
    return text.replace("实盘循环异常:", "实盘主循环发生异常，已进入错误记录与下一轮恢复等待：");
  }

  if (text.startsWith("reward_risk 获取失败")) {
    return text.replace("reward_risk 获取失败，使用默认 1.0：", "reward/risk 估算失败，系统临时回退到默认值 1.0，错误详情：");
  }

  if (text.includes("Live trading monitor started")) {
    return text.replace(/🟢\s*Live trading monitor started \(daemon loop, poll_sec=(\d+)\)/, "实盘监控进程已启动，当前以守护模式运行，轮询间隔 $1 秒。");
  }

  if (text.includes("paper_ready_ok")) {
    return "测试盘交易环境校验通过，API、持仓模式和基础运行条件已准备完成。";
  }

  return text;
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

function buildPositionRow({
  symbol,
  direction,
  qty,
  entryPrice,
  currentPrice,
  leverage,
  holdBars,
  params,
}: {
  symbol: string;
  direction: "long" | "short";
  qty: number;
  entryPrice: number | null;
  currentPrice: number | null;
  leverage: number;
  holdBars: number;
  params: StrategyParams;
}): PositionRow {
  const unrealizedPnl = computeUnrealizedPnl(direction, qty, entryPrice, currentPrice);
  const riskBands = computeApproxRiskBands(direction, entryPrice, params);
  const holdingMinutes = holdBars * timeframeToMinutes(params.timeframe);

  return {
    symbol,
    direction: direction === "short" ? "Short" : "Long",
    entryPrice,
    currentPrice,
    positionSize: `${Math.abs(qty).toFixed(4)} ${extractBaseAsset(symbol)}`,
    leverage: `${leverage.toFixed(0)}x`,
    unrealizedPnl,
    stopLoss: riskBands.stopLoss,
    takeProfit: riskBands.takeProfit,
    holdingTime: humanizeMinutes(holdingMinutes),
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

function buildMarketCandles(
  history: ApiDashboardHistoryPoint[],
  fallbackCandles: MarketCandle[],
  currentPrice: number | null,
  updatedAt: string,
): MarketCandle[] {
  const normalizedHistory = history
    .map((point) => ({
      timestamp: toIsoString(point.bar_ts || point.timestamp, ""),
      price: toNumber(point.price),
      positionQty: toNumber(point.position_qty),
    }))
    .filter((point) => point.timestamp && point.price !== null);

  if (normalizedHistory.length === 0) {
    if (currentPrice !== null) {
      return [
        {
          timestamp: updatedAt,
          open: currentPrice,
          high: currentPrice * 1.003,
          low: currentPrice * 0.997,
          close: currentPrice,
          volume: 100000,
        },
      ];
    }
    return fallbackCandles;
  }

  const candles = normalizedHistory.map((point, index) => {
    const open = normalizedHistory[index - 1]?.price ?? point.price ?? 0;
    const close = point.price ?? open;
    const body = Math.abs(close - open);
    const wick = Math.max(close * 0.0012, body * 0.7, 0.02);
    return {
      timestamp: point.timestamp,
      open,
      high: Math.max(open, close) + wick,
      low: Math.max(0, Math.min(open, close) - wick),
      close,
      volume: Math.max(
        12000,
        Math.round(
          (body * Math.max(close, 1) * 280) +
          (Math.abs(point.positionQty ?? 0) * 320) +
          ((index % 5) + 1) * 1800,
        ),
      ),
    };
  });

  return candles.slice(-48);
}

function buildOrderBook(midPrice: number, signalDirection: SignalDirection, exposurePct: number) {
  const spreadBase = Math.max(midPrice * 0.00035, 0.01);
  const tilt = signalDirection === "Long" ? 1.08 : signalDirection === "Short" ? 0.94 : 1.0;
  const depthBase = Math.max(180, exposurePct * 11 + 320);

  const asks = Array.from({ length: 6 }, (_, index) => {
    const price = Number((midPrice + spreadBase * (index + 1.2)).toFixed(2));
    const size = Number((depthBase * (0.92 + index * 0.19) * (signalDirection === "Short" ? 1.08 : 1)).toFixed(1));
    return { price, size, total: 0 };
  });
  const bids = Array.from({ length: 6 }, (_, index) => {
    const price = Number((midPrice - spreadBase * (index + 0.8)).toFixed(2));
    const size = Number((depthBase * tilt * (0.88 + index * 0.17)).toFixed(1));
    return { price, size, total: 0 };
  });

  let askTotal = 0;
  let bidTotal = 0;
  asks.forEach((level) => {
    askTotal += level.size;
    level.total = Number(askTotal.toFixed(1));
  });
  bids.forEach((level) => {
    bidTotal += level.size;
    level.total = Number(bidTotal.toFixed(1));
  });

  return {
    midPrice,
    spread: Number((asks[0].price - bids[0].price).toFixed(2)),
    spreadPct: Number((((asks[0].price - bids[0].price) / Math.max(midPrice, 1e-9)) * 100).toFixed(4)),
    asks,
    bids,
  };
}

function buildSystemPulse(args: {
  pollSec: number;
  marginUsagePct: number;
  leverage: number;
  signalScore: number;
  runtimeStatus: StrategyStatus;
}): SystemPulse {
  const { pollSec, marginUsagePct, leverage, signalScore, runtimeStatus } = args;
  const cpu = Math.max(18, Math.min(92, Math.round(18 + signalScore * 0.34 + pollSec * 0.8)));
  const memory = Math.max(20, Math.min(94, Math.round(26 + marginUsagePct * 0.45)));
  const disk = Math.max(16, Math.min(88, Math.round(22 + leverage * 10 + marginUsagePct * 0.12)));
  const latencyPenalty = runtimeStatus === "Error" ? 55 : runtimeStatus === "Paused" ? 22 : 0;
  const latency = Math.max(12, Math.round(pollSec * 2.4 + marginUsagePct * 0.2 + latencyPenalty));
  return { cpu, memory, disk, latency };
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
  if (direction === "flat") {
    return [];
  }

  const currentPrice = toNumber(market?.last_price);
  const leverage = toNumber(market?.leverage) ?? params.maxLeverage;
  const holdBars = toNumber(position?.hold_bars) ?? 0;
  const entryPrice = toNumber(position?.entry_price);

  if (direction === "mixed") {
    const longQty = toNumber(position?.long_qty) ?? 0;
    const shortQty = toNumber(position?.short_qty) ?? 0;
    const rows: PositionRow[] = [];

    if (longQty > 0) {
      rows.push(
        buildPositionRow({
          symbol: market.symbol,
          direction: "long",
          qty: longQty,
          entryPrice: toNumber(position?.long_entry_price),
          currentPrice,
          leverage,
          holdBars,
          params,
        }),
      );
    }
    if (shortQty > 0) {
      rows.push(
        buildPositionRow({
          symbol: market.symbol,
          direction: "short",
          qty: shortQty,
          entryPrice: toNumber(position?.short_entry_price),
          currentPrice,
          leverage,
          holdBars,
          params,
        }),
      );
    }

    return rows;
  }

  if (qty === null || qty === 0) {
    return [];
  }

  return [
    buildPositionRow({
      symbol: market.symbol,
      direction: direction === "short" ? "short" : "long",
      qty,
      entryPrice,
      currentPrice,
      leverage,
      holdBars,
      params,
    }),
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
      entrySource: trade.entry_source || (toNumber(trade.entry) !== null ? "exchange_fill" : "not_recorded"),
      exit: toNumber(trade.exit),
      exitSource: trade.exit_source || (toNumber(trade.exit) !== null ? "exchange_fill" : "not_recorded"),
      pnl: toNumber(trade.pnl),
      pnlSource: trade.pnl_source || (toNumber(trade.pnl) !== null ? "exchange_fill" : "not_recorded"),
      fee: toNumber(trade.fee),
      feeSource: trade.fee_source || (toNumber(trade.fee) !== null ? "exchange_fill" : "not_recorded"),
      slippage: toNumber(trade.slippage),
      slippageSource: trade.slippage_source || (toNumber(trade.slippage) !== null ? "exchange_fill" : "not_recorded"),
      reason: humanizeTradeReason(trade.reason || "最近一次执行"),
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
        entrySource: toNumber(bundle.status?.position?.entry_price) !== null ? "position_snapshot" : "not_recorded",
        exit: /执行平仓/.test(message) ? toNumber(bundle.status?.market?.last_price) : null,
        exitSource: /执行平仓/.test(message) ? "market_snapshot" : "not_recorded",
        pnl: null,
        pnlSource: "not_recorded",
        fee: null,
        feeSource: "not_recorded",
        slippage: null,
        slippageSource: "not_recorded",
        reason: humanizeTradeReason(message, message),
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
        entrySource: toNumber(bundle.status?.position?.entry_price) !== null ? "position_snapshot" : "not_recorded",
        exit: lastExecution.action === "CLOSE" ? toNumber(bundle.status?.market?.last_price) : null,
        exitSource: lastExecution.action === "CLOSE" ? "market_snapshot" : "not_recorded",
        pnl: null,
        pnlSource: "not_recorded",
        fee: null,
        feeSource: "not_recorded",
        slippage: null,
        slippageSource: "not_recorded",
        reason: humanizeTradeReason(lastExecution.reason || bundle.status?.decision?.reason || "最后一次执行"),
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
      message: humanizeRuntimeLogMessage(event),
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
  const netPositionQty = toNumber(position.net_qty);
  const positionMode = typeof position.direction === "string" ? position.direction : null;
  const exposurePct = currentEquity > 0 && positionNotional !== null ? (positionNotional / currentEquity) * 100 : 0;
  const leverage = toNumber(liveRisk?.current_leverage) ?? toNumber(market.leverage) ?? params.maxLeverage;
  const runtimeStatus = normalizeStatus(runtime.last_status);
  const derivedRiskLevel = deriveRiskLevel(exposurePct, leverage, runtimeStatus);
  const signalDirection = liveSignal?.direction
    ? normalizeSignalDirection(liveSignal.direction)
    : deriveSignalDirection(toNumber(signal.long_prob), toNumber(signal.short_prob));
  const dailyPnl = toNumber(liveMetrics?.daily_pnl) ?? deriveDailyPnlFromCurve(equityCurve);
  const signalScore = toNumber(liveSignal?.score) ?? (
    toNumber(signal.long_prob) !== null && toNumber(signal.short_prob) !== null
      ? Math.round(Math.max(toNumber(signal.long_prob) || 0, toNumber(signal.short_prob) || 0) * 100)
      : mockSnapshot.signal.score
  );
  const marketSymbol = market.symbol || mockSnapshot.marketChart.symbol;
  const currentTimeframe = params.timeframe || mockSnapshot.marketChart.timeframe;
  const marketChartTimeframe = currentTimeframe === "1H" ? "1小时" : currentTimeframe === "15m" ? "15分钟" : currentTimeframe === "5m" ? "5分钟" : currentTimeframe;
  const marketCandles = buildMarketCandles(bundle.history || [], mockSnapshot.marketChart.candles, currentPrice, updatedAt);
  const orderBook = buildOrderBook(currentPrice ?? marketCandles[marketCandles.length - 1]?.close ?? mockSnapshot.orderBook.midPrice, signalDirection, exposurePct);
  const systemPulse = buildSystemPulse({
    pollSec: toNumber(runtime.poll_sec) ?? 10,
    marginUsagePct: toNumber(liveRisk?.margin_usage_pct) ?? exposurePct,
    leverage,
    signalScore,
    runtimeStatus,
  });

  return {
    productName: strategyMeta?.product_name || "Quant Alpha 控制台",
    strategyName: strategyMeta?.strategy_name || (market.symbol ? `${market.symbol} 实盘策略控制台` : mockSnapshot.strategyName),
    exchange: strategyMeta?.exchange || market.exchange || mockSnapshot.exchange,
    status: runtimeStatus,
    updatedAt,
    dataSource,
    equityCurve,
    marketChart: {
      symbol: marketSymbol.replace("-SWAP", "").replace("-", "/"),
      timeframe: marketChartTimeframe,
      venue: market.exchange || mockSnapshot.marketChart.venue,
      candles: marketCandles,
    },
    orderBook,
    systemPulse,
    metrics: {
      equity: currentEquity,
      dailyPnl,
      totalReturnPct: toNumber(liveMetrics?.total_return_pct) ?? mockSnapshot.metrics.totalReturnPct,
      maxDrawdownPct: toNumber(liveMetrics?.max_drawdown_pct) ?? Math.min(...equityCurve.map((point) => point.drawdown)),
      sharpeRatio: toNumber(liveMetrics?.sharpe_ratio) ?? mockSnapshot.metrics.sharpeRatio,
      winRatePct: toNumber(liveMetrics?.win_rate_pct) ?? mockSnapshot.metrics.winRatePct,
      openPositions: toNumber(liveMetrics?.open_positions) ?? positions.length,
      netPositionQty,
      positionNotional,
      positionMode,
      riskLevel: normalizeRiskLevel(liveMetrics?.risk_level, derivedRiskLevel),
    },
    signal: {
      direction: signalDirection,
      sources: buildSignalSources(bundle),
      score: signalScore,
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
