import type {
  CoreMetrics,
  DashboardSnapshot,
  EquityPoint,
  LogEntry,
  PositionRow,
  RiskSnapshot,
  StrategyParams,
  StrategySignal,
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
    holdingTime: "6h 14m",
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
    holdingTime: "1d 3h",
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
    holdingTime: "9h 48m",
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
    reason: "ML breakout + RSI trend confirmation",
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
    reason: "MACD reversal momentum",
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
    reason: "ATR stop loss execution",
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
    reason: "KDJ overextension fade",
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
    reason: "Trend continuation with volume regime shift",
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
    reason: "Counter-trend scalp invalidated",
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

const logs: LogEntry[] = [
  { id: "log-1", time: minutesAgo(1), level: "SUCCESS", message: "Order filled on SOLUSDT.P, execution latency 184ms." },
  { id: "log-2", time: minutesAgo(2), level: "INFO", message: "Signal engine recalculated factors for 15m / 1h regime filters." },
  { id: "log-3", time: minutesAgo(4), level: "INFO", message: "Market data snapshot received from OKX WebSocket stream." },
  { id: "log-4", time: minutesAgo(9), level: "WARN", message: "ETH mean-reversion signal weakened below score threshold, holding unchanged." },
  { id: "log-5", time: minutesAgo(13), level: "SUCCESS", message: "Take-profit level updated for BTC short after volatility contraction." },
  { id: "log-6", time: minutesAgo(17), level: "INFO", message: "Risk engine heartbeat healthy, all capital guardrails within threshold." },
  { id: "log-7", time: minutesAgo(22), level: "ERROR", message: "Transient REST retry on positions endpoint resolved after 1 attempt." },
];

const equityCurve = generateEquityCurve();
const metrics = buildMetrics(equityCurve, positions, "Medium");

export const rollingLogMessages = [
  "Strategy heartbeat confirmed, awaiting next bar close.",
  "Received fresh order book imbalance update from derivatives feed.",
  "ML ensemble score refreshed with volatility regime overlay.",
  "Trailing stop recalibrated for active SOL long position.",
  "Funding-rate monitor completed, no hedge rebalance needed.",
  "Execution gateway latency stable below 220ms.",
  "Risk control review passed, session remains fully tradable.",
];

export const dashboardSnapshot: DashboardSnapshot = {
  productName: "Quant Alpha Dashboard",
  strategyName: "Crypto Multi-Factor Alpha v2.7",
  exchange: "OKX",
  status: "Running",
  updatedAt: new Date().toISOString(),
  dataSource: "mock",
  equityCurve,
  metrics,
  signal,
  positions,
  trades,
  risk,
  params,
  logs,
};
