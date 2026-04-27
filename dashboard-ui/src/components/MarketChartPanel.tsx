import { useEffect, useRef, useState } from "react";
import { formatNumber } from "../lib/format";
import type { MarketChartSnapshot } from "../types";
import { SegmentedControl } from "./SegmentedControl";

interface MarketChartPanelProps {
  marketChart: MarketChartSnapshot;
}

type MarketTimeframeLabel = "1分" | "15分" | "1小时" | "4小时" | "1日";

const MARKET_TIMEFRAME_OPTIONS: Array<{
  label: MarketTimeframeLabel;
  minutes: number;
  maxCandles: number;
}> = [
  { label: "1分", minutes: 1, maxCandles: 90 },
  { label: "15分", minutes: 15, maxCandles: 72 },
  { label: "1小时", minutes: 60, maxCandles: 48 },
  { label: "4小时", minutes: 240, maxCandles: 48 },
  { label: "1日", minutes: 1440, maxCandles: 30 },
];

const axisTimeFormatters = {
  intraday: new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Asia/Shanghai",
  }),
  swing: new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Asia/Shanghai",
  }),
  daily: new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    timeZone: "Asia/Shanghai",
  }),
};

function average(values: number[]) {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function movingAverage(values: number[], period: number) {
  return values.map((_, index) => {
    const start = Math.max(0, index - period + 1);
    return average(values.slice(start, index + 1));
  });
}

function roundPrice(value: number) {
  return Number(value.toFixed(2));
}

function roundVolume(value: number) {
  return Math.max(0, Math.round(value));
}

function timeframeToMinutes(timeframe: string): number | null {
  const normalized = timeframe.trim().toLowerCase();
  switch (normalized) {
    case "1分":
    case "1分钟":
    case "1m":
      return 1;
    case "5分":
    case "5分钟":
    case "5m":
      return 5;
    case "15分":
    case "15分钟":
    case "15m":
      return 15;
    case "1小时":
    case "1h":
      return 60;
    case "4小时":
    case "4h":
      return 240;
    case "1日":
    case "1天":
    case "1d":
      return 1440;
    default: {
      const match = normalized.match(/^(\d+)(m|h|d)$/i);
      if (!match) {
        return null;
      }
      const count = Number(match[1]);
      const unit = match[2].toLowerCase();
      if (unit === "m") {
        return count;
      }
      if (unit === "h") {
        return count * 60;
      }
      return count * 1440;
    }
  }
}

function normalizeTimeframeLabel(timeframe: string): MarketTimeframeLabel | null {
  const minutes = timeframeToMinutes(timeframe);
  if (minutes === 1) {
    return "1分";
  }
  if (minutes === 15) {
    return "15分";
  }
  if (minutes === 60) {
    return "1小时";
  }
  if (minutes === 240) {
    return "4小时";
  }
  if (minutes === 1440) {
    return "1日";
  }
  return null;
}

function pickClosestTimeframeLabel(timeframe: string): MarketTimeframeLabel {
  const exactMatch = normalizeTimeframeLabel(timeframe);
  if (exactMatch) {
    return exactMatch;
  }

  const minutes = timeframeToMinutes(timeframe) ?? 60;
  return MARKET_TIMEFRAME_OPTIONS.reduce((closest, option) => {
    return Math.abs(option.minutes - minutes) < Math.abs(closest.minutes - minutes) ? option : closest;
  }).label;
}

function inferBaseIntervalMinutes(candles: MarketChartSnapshot["candles"], timeframe: string): number {
  const diffs = candles
    .slice(1)
    .map((candle, index) => Date.parse(candle.timestamp) - Date.parse(candles[index].timestamp))
    .filter((diff) => Number.isFinite(diff) && diff > 0)
    .sort((a, b) => a - b);

  if (diffs.length > 0) {
    return Math.max(1, Math.round(diffs[Math.floor(diffs.length / 2)] / 60000));
  }

  return timeframeToMinutes(timeframe) ?? 60;
}

function aggregateCandles(candles: MarketChartSnapshot["candles"], targetMinutes: number) {
  const bucketMs = targetMinutes * 60 * 1000;
  const aggregated: MarketChartSnapshot["candles"] = [];

  candles.forEach((candle) => {
    const timestampMs = Date.parse(candle.timestamp);
    if (!Number.isFinite(timestampMs)) {
      aggregated.push(candle);
      return;
    }

    const bucketCloseMs = Math.ceil(timestampMs / bucketMs) * bucketMs;
    const currentBucket = aggregated[aggregated.length - 1];

    if (!currentBucket || Date.parse(currentBucket.timestamp) !== bucketCloseMs) {
      aggregated.push({
        ...candle,
        timestamp: new Date(bucketCloseMs).toISOString(),
      });
      return;
    }

    currentBucket.high = Math.max(currentBucket.high, candle.high);
    currentBucket.low = Math.min(currentBucket.low, candle.low);
    currentBucket.close = candle.close;
    currentBucket.volume = roundVolume(currentBucket.volume + candle.volume);
  });

  return aggregated;
}

function expandCandles(
  candles: MarketChartSnapshot["candles"],
  baseMinutes: number,
  targetMinutes: number,
) {
  const splitCount = Math.max(1, Math.round(baseMinutes / targetMinutes));
  if (splitCount === 1) {
    return candles;
  }

  return candles.flatMap((candle) => {
    const candleRange = Math.max(candle.high - candle.low, 0.02);
    const wickPadding = Math.max(candleRange / (splitCount * 4), 0.01);
    const volumePerSplit = candle.volume / splitCount;
    const endTimeMs = Date.parse(candle.timestamp);
    const peakIndex = candle.close >= candle.open
      ? splitCount - 1
      : Math.max(0, Math.floor(splitCount * 0.35));
    const troughIndex = candle.close >= candle.open
      ? Math.max(0, Math.floor(splitCount * 0.35))
      : splitCount - 1;

    return Array.from({ length: splitCount }, (_, index) => {
      const startProgress = index / splitCount;
      const endProgress = (index + 1) / splitCount;
      let open = candle.open + (candle.close - candle.open) * startProgress;
      let close = candle.open + (candle.close - candle.open) * endProgress;
      let high = Math.max(open, close) + wickPadding;
      let low = Math.max(0, Math.min(open, close) - wickPadding);

      if (index === 0) {
        open = candle.open;
      }
      if (index === splitCount - 1) {
        close = candle.close;
      }
      if (index === peakIndex) {
        high = Math.max(high, candle.high);
      }
      if (index === troughIndex) {
        low = Math.min(low, candle.low);
      }

      const segmentTimeMs = Number.isFinite(endTimeMs)
        ? endTimeMs - (splitCount - index - 1) * targetMinutes * 60 * 1000
        : Number.NaN;

      return {
        timestamp: Number.isFinite(segmentTimeMs) ? new Date(segmentTimeMs).toISOString() : candle.timestamp,
        open: roundPrice(open),
        high: roundPrice(high),
        low: roundPrice(low),
        close: roundPrice(close),
        volume: roundVolume(volumePerSplit),
      };
    });
  });
}

function buildChartCandles(
  candles: MarketChartSnapshot["candles"],
  sourceTimeframe: string,
  selectedTimeframe: MarketTimeframeLabel,
) {
  const option = MARKET_TIMEFRAME_OPTIONS.find((item) => item.label === selectedTimeframe)
    ?? MARKET_TIMEFRAME_OPTIONS[2];
  const sourceLabel = normalizeTimeframeLabel(sourceTimeframe) ?? sourceTimeframe;
  const sourceMinutes = inferBaseIntervalMinutes(candles, sourceTimeframe);
  let nextCandles = candles;
  let note = `源数据周期 ${sourceLabel}`;

  if (sourceMinutes < option.minutes) {
    nextCandles = aggregateCandles(candles, option.minutes);
    note = `由 ${sourceLabel} 源数据聚合为 ${selectedTimeframe} 视图`;
  } else if (sourceMinutes > option.minutes) {
    nextCandles = expandCandles(candles, sourceMinutes, option.minutes);
    note = `基于 ${sourceLabel} 源数据近似拆分为 ${selectedTimeframe} 视图`;
  }

  return {
    candles: nextCandles.slice(-option.maxCandles),
    note,
  };
}

function formatAxisTimestamp(timestamp: string, timeframe: MarketTimeframeLabel) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return "--";
  }

  if (timeframe === "1日") {
    return axisTimeFormatters.daily.format(date).replace(/\//g, "-");
  }
  if (timeframe === "1小时" || timeframe === "4小时") {
    return axisTimeFormatters.swing.format(date).replace(/\//g, "-");
  }
  return axisTimeFormatters.intraday.format(date);
}

export function MarketChartPanel({ marketChart }: MarketChartPanelProps) {
  const previousSourceTimeframeRef = useRef(marketChart.timeframe);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [stageWidth, setStageWidth] = useState(760);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [hoverY, setHoverY] = useState<number | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<MarketTimeframeLabel>(() =>
    pickClosestTimeframeLabel(marketChart.timeframe),
  );

  useEffect(() => {
    const previousDefault = pickClosestTimeframeLabel(previousSourceTimeframeRef.current);
    const nextDefault = pickClosestTimeframeLabel(marketChart.timeframe);

    setSelectedTimeframe((current) => (current === previousDefault ? nextDefault : current));
    previousSourceTimeframeRef.current = marketChart.timeframe;
  }, [marketChart.timeframe]);

  useEffect(() => {
    const node = stageRef.current;
    if (!node || typeof ResizeObserver === "undefined") {
      return;
    }
    const update = () => {
      const next = node.getBoundingClientRect().width;
      if (next > 0) {
        setStageWidth(Math.round(next));
      }
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const chartView = buildChartCandles(marketChart.candles, marketChart.timeframe, selectedTimeframe);
  const candles = chartView.candles.length > 0 ? chartView.candles : marketChart.candles.slice(-1);
  const closes = candles.map((candle) => candle.close);
  const highs = candles.map((candle) => candle.high);
  const lows = candles.map((candle) => candle.low);
  const maxPrice = Math.max(...highs);
  const minPrice = Math.min(...lows);
  const maxVolume = Math.max(...candles.map((candle) => candle.volume), 1);
  const width = Math.max(360, stageWidth - 32); // subtract market-stage padding (p-4 = 16px each side)
  const height = 360;
  const topPad = 28;
  const bottomPad = 44;
  const priceHeight = 208;
  const volumeTop = 252;
  const volumeHeight = 66;
  const leftPad = 18;
  const rightPad = 66;
  const innerWidth = width - leftPad - rightPad;
  const step = candles.length > 1 ? innerWidth / (candles.length - 1) : innerWidth;
  const candleWidth = Math.max(8, step * 0.54);
  const priceRange = Math.max(maxPrice - minPrice, 1e-6);
  const ma7 = movingAverage(closes, 7);
  const ma14 = movingAverage(closes, 14);
  const ma21 = movingAverage(closes, 21);
  const latest = candles[candles.length - 1];
  const previous = candles[candles.length - 2] ?? latest;
  const delta = latest ? latest.close - previous.close : 0;
  const deltaPct = previous?.close ? (delta / previous.close) * 100 : 0;

  const yForPrice = (price: number) => topPad + ((maxPrice - price) / priceRange) * priceHeight;
  const xForIndex = (index: number) => leftPad + index * step;
  const yForVolume = (volume: number) => volumeTop + volumeHeight - (volume / maxVolume) * volumeHeight;

  const maPath = (values: number[]) =>
    values.map((value, index) => `${index === 0 ? "M" : "L"} ${xForIndex(index)} ${yForPrice(value)}`).join(" ");

  const priceLabels = Array.from({ length: 5 }, (_, index) => {
    const ratio = index / 4;
    const value = maxPrice - priceRange * ratio;
    return {
      y: topPad + priceHeight * ratio,
      value,
    };
  });

  return (
    <section className="terminal-panel">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div>
          <h2 className="panel-title">{marketChart.symbol} · {selectedTimeframe} · {marketChart.venue}</h2>
          <p className="market-toolbar-note">{chartView.note}</p>
        </div>

        <SegmentedControl
          value={selectedTimeframe}
          options={MARKET_TIMEFRAME_OPTIONS.map((item) => ({ value: item.label, label: item.label }))}
          onChange={(value) => setSelectedTimeframe(value as MarketTimeframeLabel)}
          ariaLabel="K线周期"
        />
      </div>

      <div className="market-headline">
        <div>
          <p className="market-headline-value">{latest ? formatNumber(latest.close, 2) : "--"}</p>
          <p className={`market-headline-change ${delta >= 0 ? "is-up" : "is-down"}`}>
            {delta >= 0 ? "+" : ""}{formatNumber(delta, 2)} ({deltaPct >= 0 ? "+" : ""}{formatNumber(deltaPct, 2)}%)
          </p>
        </div>
        <div className="market-indicators">
          <span>MA(7) <strong>{formatNumber(ma7[ma7.length - 1] ?? latest?.close ?? 0, 2)}</strong></span>
          <span>MA(14) <strong>{formatNumber(ma14[ma14.length - 1] ?? latest?.close ?? 0, 2)}</strong></span>
          <span>MA(21) <strong>{formatNumber(ma21[ma21.length - 1] ?? latest?.close ?? 0, 2)}</strong></span>
          <span>成交量 <strong>{Math.round((latest?.volume ?? 0) / 1000)}K</strong></span>
        </div>
      </div>

      <div ref={stageRef} className="market-stage relative">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          width={width}
          height={height}
          preserveAspectRatio="xMidYMid meet"
          className="block"
          role="img"
          aria-label={`${marketChart.symbol} market chart`}
          onMouseMove={(event) => {
            const rect = event.currentTarget.getBoundingClientRect();
            const scaleX = width / rect.width;
            const scaleY = height / rect.height;
            const localX = (event.clientX - rect.left) * scaleX;
            const localY = (event.clientY - rect.top) * scaleY;
            if (localX < leftPad || localX > width - rightPad || candles.length === 0) {
              setHoveredIndex(null);
              setHoverY(null);
              return;
            }
            const rawIndex = Math.round((localX - leftPad) / Math.max(step, 1e-6));
            const idx = Math.max(0, Math.min(candles.length - 1, rawIndex));
            setHoveredIndex(idx);
            setHoverY(localY);
          }}
          onMouseLeave={() => {
            setHoveredIndex(null);
            setHoverY(null);
          }}
        >
          <defs>
            <linearGradient id="marketVolumeUp" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="var(--color-up)" stopOpacity="0.9" />
              <stop offset="100%" stopColor="var(--color-up)" stopOpacity="0.25" />
            </linearGradient>
            <linearGradient id="marketVolumeDown" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="var(--color-down)" stopOpacity="0.85" />
              <stop offset="100%" stopColor="var(--color-down)" stopOpacity="0.24" />
            </linearGradient>
          </defs>

          {priceLabels.map((label) => (
            <g key={label.value}>
              <line
                x1={leftPad}
                y1={label.y}
                x2={width - rightPad + 10}
                y2={label.y}
                stroke="rgba(129,148,188,0.12)"
                strokeDasharray="5 5"
              />
              <text x={width - rightPad + 18} y={label.y + 4} fill="#a1adc4" fontSize="11">
                {formatNumber(label.value, 2)}
              </text>
            </g>
          ))}

          {candles.map((candle, index) => {
            const x = xForIndex(index);
            const openY = yForPrice(candle.open);
            const closeY = yForPrice(candle.close);
            const highY = yForPrice(candle.high);
            const lowY = yForPrice(candle.low);
            const bodyY = Math.min(openY, closeY);
            const bodyHeight = Math.max(2, Math.abs(closeY - openY));
            const isUp = candle.close >= candle.open;
            const bodyColor = isUp ? "var(--color-up)" : "var(--color-down)";
            const bodyFill = isUp ? "rgba(47, 208, 147, 0.95)" : "rgba(255, 107, 122, 0.95)";

            return (
              <g key={candle.timestamp}>
                <line x1={x} y1={highY} x2={x} y2={lowY} stroke={bodyColor} strokeWidth="1.5" />
                <rect
                  x={x - candleWidth / 2}
                  y={bodyY}
                  width={candleWidth}
                  height={bodyHeight}
                  rx="2"
                  fill={bodyFill}
                />
                <rect
                  x={x - candleWidth / 2}
                  y={yForVolume(candle.volume)}
                  width={Math.max(3, candleWidth * 0.85)}
                  height={volumeTop + volumeHeight - yForVolume(candle.volume)}
                  rx="1.5"
                  fill={isUp ? "url(#marketVolumeUp)" : "url(#marketVolumeDown)"}
                />
              </g>
            );
          })}

          <path d={maPath(ma7)} fill="none" stroke="#facc15" strokeWidth="2" />
          <path d={maPath(ma14)} fill="none" stroke="#ec4899" strokeWidth="1.9" />
          <path d={maPath(ma21)} fill="none" stroke="#3b82f6" strokeWidth="1.9" />

          {(() => {
            const tickStep = Math.max(1, Math.ceil(candles.length / 7));
            return candles
              .map((candle, index) => ({ candle, index }))
              .filter(({ index }) => index % tickStep === 0 || index === candles.length - 1)
              .map(({ candle, index }) => (
                <text
                  key={`${candle.timestamp}-${index}`}
                  x={xForIndex(index)}
                  y={height - 12}
                  textAnchor="middle"
                  fill="#a1adc4"
                  fontSize="11"
                >
                  {formatAxisTimestamp(candle.timestamp, selectedTimeframe)}
                </text>
              ));
          })()}

          {hoveredIndex !== null && candles[hoveredIndex] ? (
            <g pointerEvents="none">
              <line
                x1={xForIndex(hoveredIndex)}
                y1={topPad}
                x2={xForIndex(hoveredIndex)}
                y2={volumeTop + volumeHeight}
                stroke="rgba(170,180,200,0.45)"
                strokeDasharray="3 4"
                strokeWidth="1"
              />
              {hoverY !== null && hoverY >= topPad && hoverY <= topPad + priceHeight ? (
                <>
                  <line
                    x1={leftPad}
                    y1={hoverY}
                    x2={width - rightPad}
                    y2={hoverY}
                    stroke="rgba(170,180,200,0.45)"
                    strokeDasharray="3 4"
                    strokeWidth="1"
                  />
                  <rect
                    x={width - rightPad + 2}
                    y={hoverY - 9}
                    width={56}
                    height={18}
                    rx={3}
                    fill="#10192a"
                    stroke="rgba(170,180,200,0.4)"
                  />
                  <text
                    x={width - rightPad + 30}
                    y={hoverY + 4}
                    fontSize="11"
                    fill="#e2e8f0"
                    textAnchor="middle"
                  >
                    {formatNumber(maxPrice - ((hoverY - topPad) / priceHeight) * priceRange, 2)}
                  </text>
                </>
              ) : null}
              <rect
                x={Math.min(width - 80, Math.max(leftPad, xForIndex(hoveredIndex) - 38))}
                y={height - 22}
                width={76}
                height={18}
                rx={3}
                fill="#10192a"
                stroke="rgba(170,180,200,0.4)"
              />
              <text
                x={Math.min(width - 42, Math.max(leftPad + 38, xForIndex(hoveredIndex)))}
                y={height - 9}
                fontSize="11"
                fill="#e2e8f0"
                textAnchor="middle"
              >
                {formatAxisTimestamp(candles[hoveredIndex].timestamp, selectedTimeframe)}
              </text>
            </g>
          ) : null}
        </svg>

        {hoveredIndex !== null && candles[hoveredIndex] ? (
          <div
            className="pointer-events-none absolute top-3 left-3 surface-1 px-3 py-2 text-xs font-mono leading-5 text-slate-200"
            style={{ minWidth: 168 }}
          >
            {(() => {
              const c = candles[hoveredIndex];
              const isUp = c.close >= c.open;
              return (
                <>
                  <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">
                    {formatAxisTimestamp(c.timestamp, selectedTimeframe)}
                  </div>
                  <div className="mt-1 grid grid-cols-2 gap-x-3">
                    <span className="text-slate-500">O</span><span className="text-right">{formatNumber(c.open, 2)}</span>
                    <span className="text-slate-500">H</span><span className="text-right">{formatNumber(c.high, 2)}</span>
                    <span className="text-slate-500">L</span><span className="text-right">{formatNumber(c.low, 2)}</span>
                    <span className="text-slate-500">C</span>
                    <span className={`text-right ${isUp ? "text-up" : "text-down"}`}>{formatNumber(c.close, 2)}</span>
                    <span className="text-slate-500">Vol</span><span className="text-right">{formatNumber(c.volume, 0)}</span>
                  </div>
                </>
              );
            })()}
          </div>
        ) : null}
      </div>
    </section>
  );
}
