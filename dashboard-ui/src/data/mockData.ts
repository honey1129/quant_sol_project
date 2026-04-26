import type {
  CoreMetrics,
  DashboardSnapshot,
  EquityPoint,
  LogEntry,
  MarketCandle,
  OrderBookSnapshot,
  PositionRow,
  RiskSnapshot,
  StrategyParams,
  StrategySignal,
  SystemPulse,
  TradeRow,
} from "../types";

const SIX_HOURS = 6 * 60 * 60 * 1000;

function generateEquityCurve(): EquityPoint[] {
  const now = Date.now();
  let equity = 248_500;
  let benchmark = 248_500;
  let peak = equity;

  return Array.from({ length: 361 }, (_, index) => {
    const wave = Math.sin(index / 7) * 1_450 + Math.cos(index / 13) * 930;
    const drift = 120 + ((index % 9) - 4) * 62;
    const shock = index % 71 === 0 ? -3_800 : index % 97 === 0 ? 4_400 : 0;
    const benchWave = Math.sin(index / 10) * 1_000 + Math.cos(index / 17) * 620;
    const benchShock = index % 83 === 0 ? -2_300 : index % 111 === 0 ? 2_700 : 0;

    equity += drift + wave + shock;
    benchmark += 95 + benchWave + benchShock;
    peak = Math.max(peak, equity);

    return {
      timestamp: new Date(now - (360 - index) * SIX_HOURS).toISOString(),
      equity: Number(equity.toFixed(2)),
      benchmark: Number(benchmark.toFixed(2)),
      drawdown: Number((((equity - peak) / peak) * 100).toFixed(2)),
    };
  });
}

function generateMarketCandles(): MarketCandle[] {
  const now = Date.now();
  let price = 162.4;

  return Array.from({ length: 42 }, (_, index) => {
    const drift = Math.sin(index / 4.3) * 1.9 + Math.cos(index / 8.2) * 1.1 + 0.32;
    const open = price;
    const close = Number((open + drift).toFixed(2));
    const high = Number((Math.max(open, close) + Math.abs(drift) * 0.9 + 0.8).toFixed(2));
    const low = Number((Math.min(open, close) - Math.abs(drift) * 0.7 - 0.65).toFixed(2));
    const volume = Number((138_000 + Math.abs(drift) * 42_000 + ((index % 5) + 1) * 5_800).toFixed(0));
    price = close;

    return {
      timestamp: new Date(now - (41 - index) * 60 * 60 * 1000).toISOString(),
      open: Number(open.toFixed(2)),
      high,
      low,
      close,
      volume,
    };
  });
}

function buildOrderBook(midPrice: number): OrderBookSnapshot {
  const asks = Array.from({ length: 6 }, (_, index) => {
    const price = Number((midPrice + 0.06 + index * 0.04).toFixed(2));
    const size = Number((420 + (index + 1) * 138.5).toFixed(1));
    return { price, size, total: 0 };
  });
  const bids = Array.from({ length: 6 }, (_, index) => {
    const price = Number((midPrice - 0.02 - index * 0.04).toFixed(2));
    const size = Number((390 + (index + 1) * 121.3).toFixed(1));
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
    spreadPct: Number((((asks[0].price - bids[0].price) / midPrice) * 100).toFixed(4)),
    asks,
    bids,
  };
}

function buildMetrics(equityCurve: EquityPoint[], positions: PositionRow[], riskLevel: CoreMetrics["riskLevel"]): CoreMetrics {
  const latest = equityCurve[equityCurve.length - 1];
  const dailyAnchor = equityCurve[equityCurve.length - 5];
  const starting = equityCurve[0];
  const maxDrawdownPct = Math.min(...equityCurve.map((point) => point.drawdown));

  return {
    equity: latest.equity,
    dailyPnl: latest.equity - dailyAnchor.equity,
    totalReturnPct: ((latest.equity - starting.equity) / starting.equity) * 100,
    maxDrawdownPct,
    sharpeRatio: 1.84,
    winRatePct: 63.8,
    openPositions: positions.length,
    riskLevel,
  };
}

function minutesFromNow(minutes: number): string {
  return new Date(Date.now() + minutes * 60 * 1000).toISOString();
}

function minutesAgo(minutes: number): string {
  return new Date(Date.now() - minutes * 60 * 1000).toISOString();
}

const positions: PositionRow[] = [
  {
    symbol: "SOLUSDT.P",
    direction: "Long",
    entryPrice: 167.42,
    currentPrice: 172.94,
    positionSize: "18.50 SOL",
    leverage: "4x",
    unrealizedPnl: 102.12,
    stopLoss: 162.3,
    takeProfit: 178.6,
    holdingTime: "6小时 14分钟",
  },
  {
    symbol: "BTCUSDT.P",
    direction: "Short",
    entryPrice: 68_920,
    currentPrice: 68_102,
    positionSize: "0.72 BTC",
    leverage: "3x",
    unrealizedPnl: 588.96,
    stopLoss: 69_880,
    takeProfit: 66_900,
    holdingTime: "1天 3小时",
  },
  {
    symbol: "ETHUSDT.P",
    direction: "Long",
    entryPrice: 3_428,
    currentPrice: 3_396,
    positionSize: "11.00 ETH",
    leverage: "2x",
    unrealizedPnl: -352.0,
    stopLoss: 3_318,
    takeProfit: 3_565,
    holdingTime: "9小时 48分钟",
  },
];

const trades: TradeRow[] = [
  {
    time: minutesAgo(58),
    symbol: "SOLUSDT.P",
    side: "Long",
    entry: 166.8,
    exit: 171.2,
    pnl: 214.5,
    fee: 9.8,
    slippage: 4.3,
    reason: "ML 突破信号 + RSI 趋势确认",
    status: "Take Profit",
  },
  {
    time: minutesAgo(185),
    symbol: "BTCUSDT.P",
    side: "Short",
    entry: 69_150,
    exit: 68_760,
    pnl: 281.2,
    fee: 15.6,
    slippage: 8.2,
    reason: "MACD 反转动量",
    status: "Filled",
  },
  {
    time: minutesAgo(310),
    symbol: "ETHUSDT.P",
    side: "Long",
    entry: 3_455,
    exit: 3_398,
    pnl: -401.3,
    fee: 13.1,
    slippage: 6.4,
    reason: "ATR 止损触发执行",
    status: "Stopped",
  },
  {
    time: minutesAgo(545),
    symbol: "ARBUSDT.P",
    side: "Short",
    entry: 1.23,
    exit: 1.18,
    pnl: 146.4,
    fee: 5.1,
    slippage: 2.1,
    reason: "KDJ 超涨回落",
    status: "Filled",
  },
  {
    time: minutesAgo(728),
    symbol: "BTCUSDT.P",
    side: "Long",
    entry: 67_980,
    exit: 68_540,
    pnl: 402.8,
    fee: 16.2,
    slippage: 7.7,
    reason: "趋势延续 + 成交量结构切换",
    status: "Take Profit",
  },
  {
    time: minutesAgo(910),
    symbol: "SOLUSDT.P",
    side: "Short",
    entry: 170.4,
    exit: 171.1,
    pnl: -58.1,
    fee: 8.4,
    slippage: 3.4,
    reason: "逆势短线失效",
    status: "Canceled",
  },
];

const risk: RiskSnapshot = {
  currentLeverage: 3.2,
  maxLossPerTradePct: 1.2,
  marginUsagePct: 42.6,
  dailyLossLimitPct: 4.0,
  dailyLossUsedPct: 1.4,
  riskTriggered: false,
  apiStatus: "Connected",
  wsStatus: "Connected",
};

const params: StrategyParams = {
  timeframe: "15m",
  maPeriod: 34,
  rsiPeriod: 14,
  atrMultiplier: 2.4,
  stopLossPct: 1.1,
  takeProfitPct: 2.8,
  positionSizePct: 12,
  maxLeverage: 5,
};

const signal: StrategySignal = {
  direction: "Long",
  sources: ["MACD", "RSI", "ATR", "ML Model"],
  score: 82,
  lastTriggeredAt: minutesAgo(7),
  nextRunAt: minutesFromNow(3),
};

const marketCandles = generateMarketCandles();
const orderBook = buildOrderBook(marketCandles[marketCandles.length - 1]?.close ?? 168.92);

const systemPulse: SystemPulse = {
  cpu: 28,
  memory: 46,
  disk: 32,
  latency: 28,
};

const logs: LogEntry[] = [
  { id: "log-1", time: minutesAgo(1), level: "SUCCESS", message: "SOLUSDT.P 订单已成交，执行延迟 184ms。" },
  { id: "log-2", time: minutesAgo(2), level: "INFO", message: "信号引擎已重新计算 15m / 1H 周期过滤因子。" },
  { id: "log-3", time: minutesAgo(4), level: "INFO", message: "已收到来自 OKX WebSocket 的市场快照。" },
  { id: "log-4", time: minutesAgo(9), level: "WARN", message: "ETH 均值回归信号跌破阈值，保持原仓位不变。" },
  { id: "log-5", time: minutesAgo(13), level: "SUCCESS", message: "波动收缩后，BTC 空单止盈位已更新。" },
  { id: "log-6", time: minutesAgo(17), level: "INFO", message: "风控心跳正常，全部资金护栏均在阈值内。" },
  { id: "log-7", time: minutesAgo(22), level: "ERROR", message: "持仓接口发生短暂 REST 重试，1 次后恢复成功。" },
];

const equityCurve = generateEquityCurve();
const metrics = buildMetrics(equityCurve, positions, "Medium");

export const rollingLogMessages = [
  "策略心跳正常，等待下一根 K 线收盘。",
  "已收到衍生品盘口失衡的最新更新。",
  "ML 组合评分已叠加波动率状态重新刷新。",
  "当前 SOL 多单的移动止损已重新校准。",
  "资金费率监控完成，无需对冲再平衡。",
  "执行网关延迟稳定低于 220ms。",
  "风控检查通过，当前会话可继续交易。",
];

export const dashboardSnapshot: DashboardSnapshot = {
  productName: "Quant Alpha 控制台",
  strategyName: "加密多因子 Alpha v2.7",
  exchange: "OKX",
  status: "Running",
  updatedAt: new Date().toISOString(),
  dataSource: "mock",
  equityCurve,
  marketChart: {
    symbol: "SOL/USDT",
    timeframe: "1小时",
    venue: "OKX",
    candles: marketCandles,
  },
  orderBook,
  systemPulse,
  metrics,
  signal,
  positions,
  trades,
  risk,
  params,
  logs,
};
