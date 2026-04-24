import { startTransition, useEffect, useState } from "react";

const POLL_MS = 15000;

const usdFormatter = new Intl.NumberFormat("zh-CN", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const compactFormatter = new Intl.NumberFormat("zh-CN", {
  notation: "compact",
  maximumFractionDigits: 2,
});

function toNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function clamp(value, min = 0, max = 100) {
  const numeric = toNumber(value);
  if (numeric === null) {
    return min;
  }
  return Math.min(max, Math.max(min, numeric));
}

function formatUsd(value) {
  const numeric = toNumber(value);
  return numeric === null ? "--" : usdFormatter.format(numeric);
}

function formatNumber(value, digits = 2) {
  const numeric = toNumber(value);
  return numeric === null ? "--" : numeric.toFixed(digits);
}

function formatCompact(value) {
  const numeric = toNumber(value);
  return numeric === null ? "--" : compactFormatter.format(numeric);
}

function formatPercent(value, digits = 2) {
  const numeric = toNumber(value);
  if (numeric === null) {
    return "--";
  }
  return `${numeric >= 0 ? "+" : ""}${numeric.toFixed(digits)}%`;
}

function formatRatioPercent(value, digits = 2) {
  const numeric = toNumber(value);
  return numeric === null ? "--" : formatPercent(numeric * 100, digits);
}

function formatTimestamp(value) {
  if (!value) {
    return "--";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatRelativeTime(value) {
  if (!value) {
    return "--";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "--";
  }

  const diffSeconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  if (diffSeconds < 60) {
    return `${diffSeconds}秒前`;
  }
  if (diffSeconds < 3600) {
    return `${Math.floor(diffSeconds / 60)}分钟前`;
  }
  if (diffSeconds < 86400) {
    return `${Math.floor(diffSeconds / 3600)}小时前`;
  }
  return `${Math.floor(diffSeconds / 86400)}天前`;
}

function directionLabel(direction, qty) {
  const normalizedQty = toNumber(qty);
  if (direction === "long") {
    return `多头 ${formatNumber(normalizedQty, 4)}`;
  }
  if (direction === "short") {
    return `空头 ${formatNumber(Math.abs(normalizedQty || 0), 4)}`;
  }
  return "空仓";
}

function buildPolyline(points, width, height, padding = 16) {
  if (!points || points.length < 2) {
    return { linePath: "", areaPath: "", min: null, max: null, first: null, last: null };
  }

  const numericPoints = points.map((point) => toNumber(point)).filter((point) => point !== null);
  if (numericPoints.length < 2) {
    return { linePath: "", areaPath: "", min: null, max: null, first: null, last: null };
  }

  const min = Math.min(...numericPoints);
  const max = Math.max(...numericPoints);
  const spread = max - min || 1;
  const innerWidth = width - padding * 2;
  const innerHeight = height - padding * 2;

  const chartPoints = numericPoints.map((value, index) => {
    const x = padding + (innerWidth * index) / Math.max(numericPoints.length - 1, 1);
    const y = padding + innerHeight - ((value - min) / spread) * innerHeight;
    return [x, y];
  });

  const linePath = chartPoints
    .map(([x, y], index) => `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`)
    .join(" ");

  const firstPoint = chartPoints[0];
  const lastPoint = chartPoints[chartPoints.length - 1];
  const areaPath = [
    `M ${firstPoint[0].toFixed(2)} ${(height - padding).toFixed(2)}`,
    ...chartPoints.map(([x, y]) => `L ${x.toFixed(2)} ${y.toFixed(2)}`),
    `L ${lastPoint[0].toFixed(2)} ${(height - padding).toFixed(2)}`,
    "Z",
  ].join(" ");

  return {
    linePath,
    areaPath,
    min,
    max,
    first: numericPoints[0],
    last: numericPoints[numericPoints.length - 1],
  };
}

function StatusBadge({ status }) {
  const normalized = status || "unknown";
  const tone =
    normalized === "running"
      ? "good"
      : normalized === "waiting_next_bar"
        ? "watch"
        : normalized === "error"
          ? "bad"
          : "neutral";

  const labelMap = {
    starting: "启动中",
    running: "运行中",
    waiting_next_bar: "等待下一根 Bar",
    error: "异常",
    unknown: "未知",
  };

  return <span className={`status-badge status-${tone}`}>{labelMap[normalized] || normalized}</span>;
}

function MetricCard({ eyebrow, value, delta, note, tone = "neutral" }) {
  return (
    <article className={`metric-card tone-${tone}`}>
      <p className="metric-eyebrow">{eyebrow}</p>
      <h3 className="metric-value">{value}</h3>
      <p className="metric-delta">{delta}</p>
      <p className="metric-note">{note}</p>
    </article>
  );
}

function SignalBar({ label, value, tone = "teal" }) {
  const ratio = toNumber(value) || 0;
  const width = clamp(ratio * 100);
  return (
    <div className="signal-row">
      <div className="signal-row-head">
        <span>{label}</span>
        <strong>{formatRatioPercent(ratio, 1)}</strong>
      </div>
      <div className="signal-track">
        <div className={`signal-fill signal-${tone}`} style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

function ProgressMeter({ label, value, note, tone = "teal" }) {
  const width = clamp(value);
  return (
    <div className="progress-meter">
      <div className="progress-head">
        <span>{label}</span>
        <strong>{toNumber(value) === null ? "--" : `${width.toFixed(1)}%`}</strong>
      </div>
      <div className="progress-track">
        <div className={`progress-fill progress-${tone}`} style={{ width: `${width}%` }} />
      </div>
      <p>{note}</p>
    </div>
  );
}

function CurvePanel({ title, subtitle, history, dataKey, formatter, strokeClass = "teal" }) {
  const series = history.map((item) => item?.[dataKey]);
  const { linePath, areaPath, min, max, first, last } = buildPolyline(series, 640, 240, 18);
  const firstLabel = history.length > 0 ? formatTimestamp(history[0].bar_ts || history[0].timestamp) : "--";
  const lastLabel =
    history.length > 0 ? formatTimestamp(history[history.length - 1].bar_ts || history[history.length - 1].timestamp) : "--";

  return (
    <section className="panel chart-panel">
      <div className="panel-head panel-head-wrap">
        <div>
          <p className="panel-eyebrow">{subtitle}</p>
          <h3>{title}</h3>
        </div>
        <div className="chart-range">
          <span>起点 {formatter(first)}</span>
          <span>最新 {formatter(last)}</span>
          <span>区间高 {formatter(max)}</span>
          <span>区间低 {formatter(min)}</span>
        </div>
      </div>

      {linePath ? (
        <div className="chart-shell">
          <svg viewBox="0 0 640 240" className="curve-chart" role="img" aria-label={title}>
            <defs>
              <linearGradient id={`${dataKey}-fill-teal`} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="rgba(20, 184, 166, 0.34)" />
                <stop offset="100%" stopColor="rgba(20, 184, 166, 0.02)" />
              </linearGradient>
              <linearGradient id={`${dataKey}-fill-warm`} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="rgba(245, 109, 75, 0.30)" />
                <stop offset="100%" stopColor="rgba(245, 109, 75, 0.02)" />
              </linearGradient>
              <linearGradient id={`${dataKey}-fill-ink`} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="rgba(56, 104, 179, 0.26)" />
                <stop offset="100%" stopColor="rgba(56, 104, 179, 0.02)" />
              </linearGradient>
            </defs>

            <path d={areaPath} fill={`url(#${dataKey}-fill-${strokeClass})`} />
            <path d={linePath} className={`curve-line curve-${strokeClass}`} />
          </svg>
          <div className="chart-footer">
            <span>{firstLabel}</span>
            <span>{lastLabel}</span>
          </div>
        </div>
      ) : (
        <div className="empty-chart">暂无足够历史数据绘制曲线</div>
      )}
    </section>
  );
}

function EventFeed({ events }) {
  return (
    <section className="panel feed-panel">
      <div className="panel-head">
        <div>
          <p className="panel-eyebrow">Recent Events</p>
          <h3>最近策略事件</h3>
        </div>
      </div>

      <div className="event-list">
        {events.length === 0 ? (
          <div className="empty-feed">暂无关键事件</div>
        ) : (
          events
            .slice()
            .reverse()
            .map((item) => (
              <article key={item} className="event-card">
                <span className="event-dot" />
                <p>{item}</p>
              </article>
            ))
        )}
      </div>
    </section>
  );
}

function App() {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    async function loadDashboard() {
      try {
        const response = await fetch("/api/dashboard", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Dashboard API returned ${response.status}`);
        }

        const json = await response.json();
        if (!active) {
          return;
        }

        startTransition(() => {
          setPayload(json);
          setError("");
          setLoading(false);
        });
      } catch (err) {
        if (!active) {
          return;
        }
        startTransition(() => {
          setError(err.message || "Failed to load dashboard");
          setLoading(false);
        });
      }
    }

    loadDashboard();
    const timer = setInterval(loadDashboard, POLL_MS);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  const status = payload?.status || {};
  const runtime = status.runtime || {};
  const market = status.market || {};
  const bar = status.bar || {};
  const signal = status.signal || {};
  const account = status.account || {};
  const position = status.position || {};
  const performance = status.performance || {};
  const decision = status.decision || {};
  const lastExecution = status.last_execution || {};
  const history = payload?.history || [];
  const events = payload?.recent_events || [];

  const totalEq = toNumber(account.total_eq);
  const availEq = toNumber(account.avail_eq);
  const lastPrice = toNumber(market.last_price);
  const entryPrice = toNumber(position.entry_price);
  const notional = toNumber(position.notional);
  const netQty = toNumber(position.net_qty);
  const drawdownPct = toNumber(performance.drawdown_pct);
  const netPnl = toNumber(performance.net_pnl);
  const signalSpreadPct =
    toNumber(signal.long_prob) !== null && toNumber(signal.short_prob) !== null
      ? (toNumber(signal.long_prob) - toNumber(signal.short_prob)) * 100
      : null;
  const freeRatioPct = totalEq !== null && totalEq > 0 && availEq !== null ? (availEq / totalEq) * 100 : null;
  const exposurePct = totalEq !== null && totalEq > 0 && notional !== null ? (notional / totalEq) * 100 : null;

  let floatingPnl = null;
  if (lastPrice !== null && entryPrice !== null && netQty !== null && netQty !== 0) {
    if ((position.direction || "flat") === "long") {
      floatingPnl = (lastPrice - entryPrice) * netQty;
    } else if ((position.direction || "flat") === "short") {
      floatingPnl = (entryPrice - lastPrice) * Math.abs(netQty);
    }
  }

  const signalBiasLabel =
    signalSpreadPct === null
      ? "--"
      : signalSpreadPct > 5
        ? `偏多 ${formatPercent(signalSpreadPct, 1)}`
        : signalSpreadPct < -5
          ? `偏空 ${formatPercent(signalSpreadPct, 1)}`
          : `中性 ${formatPercent(signalSpreadPct, 1)}`;

  const directionTone =
    position.direction === "long" ? "good" : position.direction === "short" ? "bad" : "neutral";

  return (
    <main className="dashboard-shell">
      <div className="dashboard-backdrop" />

      <section className="hero">
        <div className="hero-copy">
          <p className="hero-kicker">Quant Runtime Deck</p>
          <h1>交易运行状态与账户收益总览</h1>
          <p className="hero-text">
            一屏串起运行健康度、账户净值、仓位暴露、信号偏向和最近执行行为。盯盘时不再在 PM2、日志文件和交易所账户页之间来回切换。
          </p>
          <div className="hero-chip-row">
            <span className="hero-chip">{market.simulated ? "模拟盘" : "实盘"}</span>
            <span className="hero-chip">{market.exchange || "OKX"}</span>
            <span className="hero-chip">{market.symbol || "--"}</span>
            <span className="hero-chip">杠杆 {formatNumber(market.leverage, 0)}x</span>
          </div>
        </div>

        <div className="hero-status-card">
          <div className="hero-status-row">
            <StatusBadge status={runtime.last_status} />
            <span className="hero-update">更新时间 {formatTimestamp(payload?.generated_at || status.updated_at)}</span>
          </div>
          <div className="hero-status-grid hero-status-grid-wide">
            <div>
              <span>最近处理 Bar</span>
              <strong>{formatTimestamp(bar.last_processed_bar_ts)}</strong>
            </div>
            <div>
              <span>最新已收盘 Bar</span>
              <strong>{formatTimestamp(bar.latest_closed_bar_ts)}</strong>
            </div>
            <div>
              <span>Bar 延迟</span>
              <strong>{formatRelativeTime(bar.latest_closed_bar_ts)}</strong>
            </div>
            <div>
              <span>最近执行</span>
              <strong>{formatRelativeTime(lastExecution.timestamp)}</strong>
            </div>
            <div>
              <span>轮询频率</span>
              <strong>{formatNumber(runtime.poll_sec, 0)}s</strong>
            </div>
            <div>
              <span>连续跳过同 Bar</span>
              <strong>{formatNumber(runtime.same_bar_skip_count, 0)}</strong>
            </div>
          </div>
        </div>
      </section>

      {error ? <div className="error-banner">Dashboard API error: {error}</div> : null}

      <section className="metrics-grid metrics-grid-six">
        <MetricCard
          eyebrow="账户总权益"
          value={formatUsd(totalEq)}
          delta={`基线权益 ${formatUsd(performance.baseline_total_eq)}`}
          note={`历史点数 ${formatCompact(performance.history_points)}`}
          tone="teal"
        />
        <MetricCard
          eyebrow="净收益"
          value={formatUsd(netPnl)}
          delta={formatPercent(performance.return_pct)}
          note={`峰值权益 ${formatUsd(performance.peak_total_eq)}`}
          tone={netPnl === null ? "neutral" : netPnl >= 0 ? "good" : "bad"}
        />
        <MetricCard
          eyebrow="当前回撤"
          value={formatPercent(drawdownPct)}
          delta={`谷值权益 ${formatUsd(performance.min_total_eq)}`}
          note="从历史峰值回落的比例"
          tone={drawdownPct === null ? "neutral" : drawdownPct > -3 ? "ink" : "bad"}
        />
        <MetricCard
          eyebrow="当前仓位"
          value={directionLabel(position.direction, netQty)}
          delta={`名义敞口 ${formatUsd(notional)}`}
          note={`均价 ${formatUsd(entryPrice)}`}
          tone={directionTone}
        />
        <MetricCard
          eyebrow="可用保证金率"
          value={freeRatioPct === null ? "--" : `${formatNumber(freeRatioPct, 1)}%`}
          delta={`可用权益 ${formatUsd(availEq)}`}
          note={`挂单 ${formatNumber(position.pending_orders, 0)}`}
          tone="ink"
        />
        <MetricCard
          eyebrow="持仓浮盈估算"
          value={formatUsd(floatingPnl)}
          delta={`仓位利用率 ${exposurePct === null ? "--" : `${formatNumber(exposurePct, 1)}%`}`}
          note={`最新价 ${formatUsd(lastPrice)}`}
          tone={floatingPnl === null ? "neutral" : floatingPnl >= 0 ? "good" : "bad"}
        />
      </section>

      <section className="main-grid charts-grid">
        <CurvePanel
          title="净收益曲线"
          subtitle="Net PnL Timeline"
          history={history}
          dataKey="net_pnl"
          formatter={formatUsd}
        />
        <CurvePanel
          title="账户权益曲线"
          subtitle="Total Equity Timeline"
          history={history}
          dataKey="total_eq"
          formatter={formatUsd}
          strokeClass="warm"
        />
        <CurvePanel
          title="市场价格曲线"
          subtitle="Market Price Timeline"
          history={history}
          dataKey="price"
          formatter={formatUsd}
          strokeClass="ink"
        />
        <CurvePanel
          title="仓位规模曲线"
          subtitle="Position Quantity Timeline"
          history={history}
          dataKey="position_qty"
          formatter={(value) => formatNumber(value, 4)}
        />
      </section>

      <section className="main-grid">
        <div className="side-stack">
          <section className="panel signal-panel">
            <div className="panel-head">
              <div>
                <p className="panel-eyebrow">Signal Snapshot</p>
                <h3>信号强弱与波动状态</h3>
              </div>
              <div className="signal-mini-chip">
                <span>偏向</span>
                <strong>{signalBiasLabel}</strong>
              </div>
            </div>

            <SignalBar label="做多概率" value={signal.long_prob} tone="teal" />
            <SignalBar label="做空概率" value={signal.short_prob} tone="warm" />

            <div className="signal-metrics">
              <div className="signal-metric">
                <span>资金流比</span>
                <strong>{formatNumber(signal.money_flow_ratio, 3)}</strong>
              </div>
              <div className="signal-metric">
                <span>波动率</span>
                <strong>{formatNumber(signal.volatility, 6)}</strong>
              </div>
              <div className="signal-metric">
                <span>ATR 比例</span>
                <strong>{formatRatioPercent(signal.atr_ratio, 2)}</strong>
              </div>
              <div className="signal-metric">
                <span>概率差</span>
                <strong>{formatPercent(signalSpreadPct, 1)}</strong>
              </div>
            </div>
          </section>

          <section className="panel account-panel">
            <div className="panel-head">
              <div>
                <p className="panel-eyebrow">Account Structure</p>
                <h3>账户结构与仓位占用</h3>
              </div>
            </div>

            <div className="progress-stack">
              <ProgressMeter
                label="可用权益占比"
                value={freeRatioPct}
                note={`availEq ${formatUsd(availEq)} / totalEq ${formatUsd(totalEq)}`}
                tone="teal"
              />
              <ProgressMeter
                label="仓位利用率"
                value={exposurePct}
                note={`持仓名义 ${formatUsd(notional)} / 总权益 ${formatUsd(totalEq)}`}
                tone="warm"
              />
            </div>

            <div className="account-grid">
              <div className="runtime-card">
                <span>估算浮盈</span>
                <strong>{formatUsd(floatingPnl)}</strong>
                <p>基于最新价与均价的粗略估算</p>
              </div>
              <div className="runtime-card">
                <span>当前价格</span>
                <strong>{formatUsd(lastPrice)}</strong>
                <p>开仓均价 {formatUsd(entryPrice)}</p>
              </div>
            </div>
          </section>
        </div>

        <div className="side-stack">
          <section className="panel runtime-panel">
            <div className="panel-head">
              <div>
                <p className="panel-eyebrow">Runtime Health</p>
                <h3>运行状态摘要</h3>
              </div>
            </div>

            <div className="runtime-grid runtime-grid-three">
              <div className="runtime-card">
                <span>当前决策</span>
                <strong>{decision.action || "--"}</strong>
                <p>{decision.reason || "暂无"}</p>
              </div>
              <div className="runtime-card">
                <span>上次执行</span>
                <strong>{lastExecution.action || "--"}</strong>
                <p>{lastExecution.reason || "暂无"}</p>
              </div>
              <div className="runtime-card">
                <span>最近执行时间</span>
                <strong>{formatTimestamp(lastExecution.timestamp)}</strong>
                <p>{formatRelativeTime(lastExecution.timestamp)}</p>
              </div>
              <div className="runtime-card">
                <span>循环计数</span>
                <strong>{formatCompact(runtime.loop_count)}</strong>
                <p>心跳间隔 {formatNumber(runtime.heartbeat_interval_sec, 0)}s</p>
              </div>
              <div className="runtime-card">
                <span>账户模式</span>
                <strong>{market.simulated ? "模拟盘" : "实盘"}</strong>
                <p>
                  {market.exchange || "OKX"} · {market.symbol || "--"}
                </p>
              </div>
              <div className="runtime-card">
                <span>信号目标</span>
                <strong>{formatNumber(decision.target_ratio, 3)}</strong>
                <p>目标仓位 {formatNumber(decision.target_position, 4)}</p>
              </div>
            </div>

            {runtime.last_error ? (
              <div className="runtime-error">
                <span>最近异常</span>
                <p>{runtime.last_error}</p>
              </div>
            ) : (
              <div className="runtime-ok">
                <span>系统状态</span>
                <p>目前没有记录到最近异常，心跳与 Bar 推进信息正常可见。</p>
              </div>
            )}
          </section>

          <EventFeed events={events} />
        </div>
      </section>

      {loading ? <div className="loading-fog">正在拉取仪表盘数据...</div> : null}
    </main>
  );
}

export default App;
