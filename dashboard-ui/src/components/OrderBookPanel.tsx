import { formatNumber } from "../lib/format";
import type { OrderBookSnapshot } from "../types";

interface OrderBookPanelProps {
  orderBook: OrderBookSnapshot;
  symbol: string;
}

function maxTotal(orderBook: OrderBookSnapshot) {
  return Math.max(
    ...orderBook.asks.map((level) => level.total),
    ...orderBook.bids.map((level) => level.total),
    1,
  );
}

function sumSizes(levels: OrderBookSnapshot["asks"]) {
  return levels.reduce((acc, level) => acc + level.size, 0);
}

export function OrderBookPanel({ orderBook, symbol }: OrderBookPanelProps) {
  const maxDepth = maxTotal(orderBook);
  const asks = [...orderBook.asks].reverse();
  const bids = orderBook.bids;
  const askVolume = sumSizes(orderBook.asks);
  const bidVolume = sumSizes(orderBook.bids);
  const totalVolume = askVolume + bidVolume;
  const bidPct = totalVolume > 0 ? (bidVolume / totalVolume) * 100 : 50;
  const askPct = 100 - bidPct;
  const imbalance = bidPct - askPct;
  const imbalanceTone = imbalance >= 0 ? "text-up" : "text-down";

  return (
    <section className="terminal-panel">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="panel-title">盘口深度 {symbol}</h2>
        <span className="panel-chip">L2</span>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-3">
        <div className="surface-2 px-4 py-3">
          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">中价</p>
          <p className="mt-1 font-mono text-base font-semibold text-white">{formatNumber(orderBook.midPrice, 2)}</p>
        </div>
        <div className="surface-2 px-4 py-3">
          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">价差</p>
          <p className="mt-1 font-mono text-base font-semibold text-white">
            {formatNumber(orderBook.spread, 2)}
            <span className="ml-2 text-xs font-normal text-slate-400">{formatNumber(orderBook.spreadPct, 4)}%</span>
          </p>
        </div>
        <div className="surface-2 px-4 py-3">
          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">买卖失衡</p>
          <p className={`mt-1 font-mono text-base font-semibold ${imbalanceTone}`}>
            {imbalance >= 0 ? "+" : ""}{formatNumber(imbalance, 1)}%
          </p>
        </div>
      </div>

      <div className="mb-3 flex h-1.5 overflow-hidden rounded-full bg-white/[0.04]" aria-label="买卖深度比例">
        <div className="h-full bg-up/70" style={{ width: `${bidPct}%` }} />
        <div className="h-full bg-down/70" style={{ width: `${askPct}%` }} />
      </div>

      <div className="orderbook-board">
        <div className="orderbook-board-header">
          <span>价格</span>
          <span className="text-right">数量</span>
          <span className="text-right">累计</span>
        </div>

        <div className="space-y-1">
          {asks.map((level) => (
            <div key={`ask-${level.price}`} className="orderbook-row orderbook-row-ask">
              <div className="orderbook-row-fill" style={{ width: `${(level.total / maxDepth) * 100}%` }} />
              <span className="text-down">{formatNumber(level.price, 2)}</span>
              <span className="text-right">{formatNumber(level.size, 1)}</span>
              <span className="text-right">{formatNumber(level.total, 1)}</span>
            </div>
          ))}
        </div>

        <div className="my-3 flex items-baseline justify-center gap-3 border-y border-white/10 py-2">
          <span className="font-mono text-xl font-semibold text-white">{formatNumber(orderBook.midPrice, 2)}</span>
          <span className="text-xs text-slate-400">中价</span>
        </div>

        <div className="space-y-1">
          {bids.map((level) => (
            <div key={`bid-${level.price}`} className="orderbook-row orderbook-row-bid">
              <div className="orderbook-row-fill" style={{ width: `${(level.total / maxDepth) * 100}%` }} />
              <span className="text-up">{formatNumber(level.price, 2)}</span>
              <span className="text-right">{formatNumber(level.size, 1)}</span>
              <span className="text-right">{formatNumber(level.total, 1)}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
