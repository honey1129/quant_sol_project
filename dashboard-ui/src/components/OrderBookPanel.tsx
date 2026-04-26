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

export function OrderBookPanel({ orderBook, symbol }: OrderBookPanelProps) {
  const maxDepth = maxTotal(orderBook);
  const asks = [...orderBook.asks].reverse();
  const bids = orderBook.bids;

  return (
    <section className="terminal-panel">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <p className="panel-kicker">订单簿快照</p>
          <h2 className="panel-title">盘口深度 {symbol}</h2>
        </div>
        <span className="panel-chip">L2</span>
      </div>

      <div className="orderbook-board">
        <div className="orderbook-board-header">
          <span>价格</span>
          <span>数量</span>
          <span>累计</span>
        </div>

        <div className="space-y-1.5">
          {asks.map((level) => (
            <div key={`ask-${level.price}`} className="orderbook-row orderbook-row-ask">
              <div className="orderbook-row-fill" style={{ width: `${(level.total / maxDepth) * 100}%` }} />
              <span className="text-rose-300">{formatNumber(level.price, 2)}</span>
              <span>{formatNumber(level.size, 1)}</span>
              <span>{formatNumber(level.total, 1)}</span>
            </div>
          ))}
        </div>

        <div className="orderbook-mid">
          <strong>{formatNumber(orderBook.midPrice, 2)}</strong>
          <span>价差 {formatNumber(orderBook.spread, 2)} / {formatNumber(orderBook.spreadPct, 4)}%</span>
        </div>

        <div className="space-y-1.5">
          {bids.map((level) => (
            <div key={`bid-${level.price}`} className="orderbook-row orderbook-row-bid">
              <div className="orderbook-row-fill" style={{ width: `${(level.total / maxDepth) * 100}%` }} />
              <span className="text-emerald-300">{formatNumber(level.price, 2)}</span>
              <span>{formatNumber(level.size, 1)}</span>
              <span>{formatNumber(level.total, 1)}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
