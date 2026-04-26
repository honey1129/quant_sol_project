import { formatDateTime, formatNumber } from "../lib/format";
import type { MarketChartSnapshot } from "../types";

interface MarketChartPanelProps {
  marketChart: MarketChartSnapshot;
}

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

export function MarketChartPanel({ marketChart }: MarketChartPanelProps) {
  const candles = marketChart.candles.slice(-30);
  const closes = candles.map((candle) => candle.close);
  const highs = candles.map((candle) => candle.high);
  const lows = candles.map((candle) => candle.low);
  const maxPrice = Math.max(...highs);
  const minPrice = Math.min(...lows);
  const maxVolume = Math.max(...candles.map((candle) => candle.volume), 1);
  const width = 760;
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
          <p className="panel-kicker">市场主图</p>
          <h2 className="panel-title">{marketChart.symbol} · {marketChart.timeframe} · {marketChart.venue}</h2>
          <p className="panel-subtitle">
            结合近端价格结构、均线和成交量节奏，快速判断当前信号是否有趋势地基支撑。
          </p>
        </div>

        <div className="market-toolbar">
          {["1分", "15分", "1小时", "4小时", "1日"].map((item) => (
            <span key={item} className={`market-toolbar-pill ${item === marketChart.timeframe ? "is-active" : ""}`}>
              {item}
            </span>
          ))}
        </div>
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

      <div className="market-stage">
        <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full" role="img" aria-label={`${marketChart.symbol} market chart`}>
          <defs>
            <linearGradient id="marketVolumeUp" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#2ed29a" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#2ed29a" stopOpacity="0.25" />
            </linearGradient>
            <linearGradient id="marketVolumeDown" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#ff6b7a" stopOpacity="0.85" />
              <stop offset="100%" stopColor="#ff6b7a" stopOpacity="0.24" />
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
              <text x={width - rightPad + 18} y={label.y + 4} fill="#7f90ae" fontSize="11">
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
            const bodyColor = isUp ? "#2fd093" : "#ff6b7a";

            return (
              <g key={candle.timestamp}>
                <line x1={x} y1={highY} x2={x} y2={lowY} stroke={bodyColor} strokeWidth="1.5" />
                <rect
                  x={x - candleWidth / 2}
                  y={bodyY}
                  width={candleWidth}
                  height={bodyHeight}
                  rx="2"
                  fill={isUp ? "rgba(47, 208, 147, 0.95)" : "rgba(255, 107, 122, 0.95)"}
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

          {candles.filter((_, index) => index % 6 === 0 || index === candles.length - 1).map((candle, index) => {
            const candleIndex = candles.findIndex((item) => item.timestamp === candle.timestamp);
            return (
              <text
                key={`${candle.timestamp}-${index}`}
                x={xForIndex(candleIndex)}
                y={height - 12}
                textAnchor="middle"
                fill="#7f90ae"
                fontSize="11"
              >
                {formatDateTime(candle.timestamp)}
              </text>
            );
          })}
        </svg>
      </div>
    </section>
  );
}
