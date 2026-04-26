export type ThemeMode = "dark" | "light";
export type StrategyStatus = "Running" | "Paused" | "Error";
export type ExchangeName = string;
export type SignalDirection = "Long" | "Short" | "Neutral";
export type SignalSource = "MACD" | "RSI" | "KDJ" | "ATR" | "ML Model" | string;
export type TimeRange = "1D" | "7D" | "30D" | "90D" | "All";
export type RiskLevel = "Low" | "Medium" | "High";
export type LogLevel = "INFO" | "SUCCESS" | "WARN" | "ERROR";
export type DataSource = "live" | "hybrid" | "mock";

export interface EquityPoint {
  timestamp: string;
  equity: number;
  benchmark: number;
  drawdown: number;
}

export interface CoreMetrics {
  equity: number;
  dailyPnl: number;
  totalReturnPct: number;
  maxDrawdownPct: number;
  sharpeRatio: number;
  winRatePct: number;
  openPositions: number;
  netPositionQty?: number | null;
  positionNotional?: number | null;
  positionMode?: string | null;
  riskLevel: RiskLevel;
}

export interface MarketCandle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketChartSnapshot {
  symbol: string;
  timeframe: string;
  venue: string;
  candles: MarketCandle[];
}

export interface OrderBookLevel {
  price: number;
  size: number;
  total: number;
}

export interface OrderBookSnapshot {
  midPrice: number;
  spread: number;
  spreadPct: number;
  asks: OrderBookLevel[];
  bids: OrderBookLevel[];
}

export interface SystemPulse {
  cpu: number;
  memory: number;
  disk: number;
  latency: number;
}

export interface StrategySignal {
  direction: SignalDirection;
  sources: SignalSource[];
  score: number;
  lastTriggeredAt: string;
  nextRunAt: string;
}

export interface PositionRow {
  symbol: string;
  direction: "Long" | "Short";
  entryPrice: number | null;
  currentPrice: number | null;
  positionSize: string;
  leverage: string;
  unrealizedPnl: number | null;
  stopLoss: number | null;
  takeProfit: number | null;
  holdingTime: string;
}

export interface TradeRow {
  time: string;
  symbol: string;
  side: "Long" | "Short";
  entry: number | null;
  entrySource?: string;
  exit: number | null;
  exitSource?: string;
  pnl: number | null;
  pnlSource?: string;
  fee: number | null;
  feeSource?: string;
  slippage: number | null;
  slippageSource?: string;
  reason: string;
  status: "Filled" | "Stopped" | "Take Profit" | "Canceled" | string;
}

export interface RiskSnapshot {
  currentLeverage: number;
  maxLossPerTradePct: number;
  marginUsagePct: number;
  dailyLossLimitPct: number;
  dailyLossUsedPct: number;
  riskTriggered: boolean;
  apiStatus: "Connected" | "Degraded" | "Disconnected";
  wsStatus: "Connected" | "Lagging" | "Disconnected";
}

export interface StrategyParams {
  timeframe: string;
  maPeriod: number;
  rsiPeriod: number;
  atrMultiplier: number;
  stopLossPct: number;
  takeProfitPct: number;
  positionSizePct: number;
  maxLeverage: number;
}

export interface LogEntry {
  id: string;
  time: string;
  level: LogLevel;
  message: string;
}

export interface DashboardSnapshot {
  productName: string;
  strategyName: string;
  exchange: ExchangeName;
  status: StrategyStatus;
  updatedAt: string;
  dataSource: DataSource;
  equityCurve: EquityPoint[];
  marketChart: MarketChartSnapshot;
  orderBook: OrderBookSnapshot;
  systemPulse: SystemPulse;
  metrics: CoreMetrics;
  signal: StrategySignal;
  positions: PositionRow[];
  trades: TradeRow[];
  risk: RiskSnapshot;
  params: StrategyParams;
  logs: LogEntry[];
}

export interface ApiDashboardRuntime {
  last_status?: string;
  loop_count?: number;
  same_bar_skip_count?: number;
  poll_sec?: number;
  heartbeat_interval_sec?: number;
  last_error?: string | null;
}

export interface ApiDashboardMarket {
  exchange?: string;
  symbol?: string;
  last_price?: number | null;
  leverage?: number | null;
  simulated?: boolean;
}

export interface ApiDashboardBar {
  last_processed_bar_ts?: string | null;
  latest_closed_bar_ts?: string | null;
}

export interface ApiDashboardSignal {
  long_prob?: number | null;
  short_prob?: number | null;
  money_flow_ratio?: number | null;
  volatility?: number | null;
  atr_ratio?: number | null;
}

export interface ApiDashboardAccount {
  total_eq?: number | null;
  avail_eq?: number | null;
  currency?: string;
}

export interface ApiDashboardPosition {
  direction?: string;
  net_qty?: number | null;
  entry_price?: number | null;
  long_qty?: number | null;
  short_qty?: number | null;
  long_entry_price?: number | null;
  short_entry_price?: number | null;
  hold_bars?: number | null;
  notional?: number | null;
  pending_orders?: number | null;
}

export interface ApiDashboardDecision {
  action?: string;
  reason?: string;
  target_ratio?: number | null;
  target_position?: number | null;
  delta_qty?: number | null;
}

export interface ApiDashboardExecution {
  action?: string;
  reason?: string;
  success?: boolean;
  timestamp?: string | null;
}

export interface ApiDashboardPerformance {
  baseline_total_eq?: number | null;
  current_total_eq?: number | null;
  peak_total_eq?: number | null;
  min_total_eq?: number | null;
  net_pnl?: number | null;
  return_pct?: number | null;
  drawdown_pct?: number | null;
  history_points?: number | null;
}

export interface ApiDashboardStrategyMeta {
  product_name?: string;
  strategy_name?: string;
  exchange?: string;
  symbol?: string;
  mode?: string;
  simulated?: boolean;
  intervals?: string[];
}

export interface ApiDashboardStrategyParams {
  timeframe?: string;
  intervals?: string[];
  ma_period?: number | null;
  rsi_period?: number | null;
  atr_multiplier?: number | null;
  stop_loss_pct?: number | null;
  take_profit_pct?: number | null;
  position_size_pct?: number | null;
  max_leverage?: number | null;
  adaptive_tp_sl_enabled?: boolean;
  threshold_long?: number | null;
  threshold_short?: number | null;
  atr_take_profit_multiplier?: number | null;
  atr_stop_loss_multiplier?: number | null;
  volatility_take_profit_multiplier?: number | null;
  volatility_stop_loss_multiplier?: number | null;
}

export interface ApiDashboardMetrics {
  equity?: number | null;
  daily_pnl?: number | null;
  total_return_pct?: number | null;
  max_drawdown_pct?: number | null;
  sharpe_ratio?: number | null;
  win_rate_pct?: number | null;
  open_positions?: number | null;
  risk_level?: RiskLevel | string | null;
  backtest_trade_count?: number | null;
  fees_paid?: number | null;
  slippage_cost?: number | null;
}

export interface ApiDashboardSignalSummary {
  direction?: SignalDirection | string;
  sources?: SignalSource[];
  score?: number | null;
  last_triggered_at?: string | null;
  next_run_at?: string | null;
}

export interface ApiDashboardRiskSnapshot {
  current_leverage?: number | null;
  max_loss_per_trade_pct?: number | null;
  margin_usage_pct?: number | null;
  daily_loss_limit_pct?: number | null;
  daily_loss_used_pct?: number | null;
  risk_triggered?: boolean;
  risk_level?: RiskLevel | string | null;
  api_status?: RiskSnapshot["apiStatus"] | string;
  ws_status?: RiskSnapshot["wsStatus"] | string;
  latest_error?: string | null;
}

export interface ApiDashboardTradeRow {
  time?: string;
  symbol?: string;
  side?: TradeRow["side"] | string;
  entry?: number | null;
  entry_source?: string;
  exit?: number | null;
  exit_source?: string;
  pnl?: number | null;
  pnl_source?: string;
  fee?: number | null;
  fee_source?: string;
  slippage?: number | null;
  slippage_source?: string;
  reason?: string;
  status?: TradeRow["status"] | string;
}

export interface ApiDashboardResearchMetrics {
  timestamp?: string | null;
  final_equity?: number | null;
  pnl?: number | null;
  return_pct?: number | null;
  max_drawdown_pct?: number | null;
  trade_count?: number | null;
  fees_paid?: number | null;
  slippage_cost?: number | null;
  source_path?: string | null;
  period_start?: string | null;
  period_end?: string | null;
  win_rate_pct?: number | null;
  sharpe_ratio?: number | null;
  trade_count_closed?: number | null;
}

export interface ApiDashboardStatus {
  updated_at?: string;
  runtime?: ApiDashboardRuntime;
  market?: ApiDashboardMarket;
  bar?: ApiDashboardBar;
  signal?: ApiDashboardSignal;
  account?: ApiDashboardAccount;
  position?: ApiDashboardPosition;
  decision?: ApiDashboardDecision;
  last_execution?: ApiDashboardExecution;
  performance?: ApiDashboardPerformance;
  baseline?: Record<string, unknown>;
}

export interface ApiDashboardHistoryPoint {
  bar_ts?: string;
  timestamp?: string;
  total_eq?: number | null;
  avail_eq?: number | null;
  price?: number | null;
  position_qty?: number | null;
  net_pnl?: number | null;
  return_pct?: number | null;
}

export interface ApiDashboardBundle {
  generated_at?: string;
  frontend_built?: boolean;
  status?: ApiDashboardStatus;
  history?: ApiDashboardHistoryPoint[];
  recent_events?: string[];
  strategy_meta?: ApiDashboardStrategyMeta;
  strategy_params?: ApiDashboardStrategyParams;
  metrics?: ApiDashboardMetrics;
  signal_summary?: ApiDashboardSignalSummary;
  risk_snapshot?: ApiDashboardRiskSnapshot;
  recent_trades?: ApiDashboardTradeRow[];
  research_metrics?: ApiDashboardResearchMetrics;
}

export interface ApiStrategyParamsSaveResponse {
  ok: boolean;
  error?: string;
  saved_at?: string;
  env_path?: string;
  restart_required?: boolean;
  message?: string;
  saved_params?: ApiDashboardStrategyParams;
  bundle?: ApiDashboardBundle;
}

export interface ApiStrategyRestartResponse {
  ok: boolean;
  error?: string;
  restarted_at?: string;
  command_mode?: "pm2" | "custom" | string;
  command?: string[];
  output?: string | null;
  message?: string;
  bundle?: ApiDashboardBundle;
}
