"""
简单规则策略回测 - 验证框架正确性

规则:
1. 只在 trend_long 时做多
2. 只在 trend_short 时做空
3. 不使用任何ML模型
4. 止盈止损使用配置值

目的: 验证回测框架、止盈止损逻辑、手续费计算是否正确
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config import config
from core.okx_api import OKXClient
from core.ml_feature_engineering import merge_multi_period_features, add_advanced_features
from core.trend_filter import derive_trend_context
from core.regime_filter import derive_market_regime
from utils.utils import log_info

def _row_atr_ratio(row):
    close_price = row.get("5m_close")
    atr_value = row.get("5m_atr")
    if pd.isna(close_price) or pd.isna(atr_value):
        return None
    close_price = float(close_price)
    if close_price <= 0:
        return None
    return float(atr_value) / close_price


def resolve_intrabar_tp_sl(position, entry_price, bar_open, bar_high, bar_low, take_profit, stop_loss):
    """检查bar内是否触发止盈止损，返回触发价格和原因"""
    if position == 0 or entry_price <= 0:
        return None

    if position > 0:  # 做多
        take_profit_price = entry_price * (1 + take_profit)
        stop_loss_price = entry_price * (1 - stop_loss)

        # 悲观假设：先看止损再看止盈
        hit_sl = bar_low <= stop_loss_price
        hit_tp = bar_high >= take_profit_price

        if hit_sl and hit_tp:
            # 都触发，悲观假设先止损
            return {"reason": "SL", "price": stop_loss_price}
        elif hit_sl:
            return {"reason": "SL", "price": stop_loss_price}
        elif hit_tp:
            return {"reason": "TP", "price": take_profit_price}

    else:  # 做空
        take_profit_price = entry_price * (1 - take_profit)
        stop_loss_price = entry_price * (1 + stop_loss)

        hit_sl = bar_high >= stop_loss_price
        hit_tp = bar_low <= take_profit_price

        if hit_sl and hit_tp:
            return {"reason": "SL", "price": stop_loss_price}
        elif hit_sl:
            return {"reason": "SL", "price": stop_loss_price}
        elif hit_tp:
            return {"reason": "TP", "price": take_profit_price}

    return None


def simple_rule_backtest():
    """
    简单规则策略回测

    规则:
    - trend_long -> 做多 0.15 倍仓位
    - trend_short -> 做空 0.15 倍仓位
    - neutral -> 平仓
    """
    log_info("=" * 80)
    log_info("简单规则策略回测 - 验证框架")
    log_info("=" * 80)

    # 拉取数据
    log_info("拉取数据...")
    client = OKXClient()
    data_dict = client.fetch_data()
    merged_df = merge_multi_period_features(data_dict)
    merged_df = add_advanced_features(merged_df)
    df = merged_df.dropna().copy()

    log_info(f"数据范围: {df.index[0]} 至 {df.index[-1]}, 共 {len(df)} 条")

    # 配置
    initial_balance = config.INITIAL_BALANCE
    position_size = 0.15  # 15% 仓位
    take_profit = config.TAKE_PROFIT
    stop_loss = config.STOP_LOSS
    fee_rate = config.FEE_RATE

    log_info(f"初始资金: {initial_balance} USDT")
    log_info(f"仓位大小: {position_size:.0%}")
    log_info(f"止盈: {take_profit:.1%}, 止损: {stop_loss:.1%}")
    log_info(f"手续费率: {fee_rate:.2%}")
    log_info("")

    # 回测状态
    balance = initial_balance
    position = 0.0  # 正数做多，负数做空
    entry_price = 0.0
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        bar_ts = row.name
        close = row['5m_close']
        high = row['5m_high']
        low = row['5m_low']
        open_price = row['5m_open']

        # 计算 trend 和 regime
        trend_context = derive_trend_context(
            row,
            interval=config.TREND_FILTER_INTERVAL,
            fast_col=config.TREND_FILTER_FAST_COL,
            slow_col=config.TREND_FILTER_SLOW_COL,
            min_gap=config.TREND_FILTER_MIN_GAP,
        )
        regime_context = derive_market_regime(
            trend_bias=trend_context.get("trend_bias"),
            trend_gap=trend_context.get("trend_gap"),
            volatility=row.get("volatility_15"),
            atr_ratio=_row_atr_ratio(row),
            money_flow_ratio=row.get("money_flow_ratio"),
            trend_gap_threshold=config.REGIME_TREND_GAP_THRESHOLD,
            high_vol_atr_threshold=config.REGIME_HIGH_VOL_ATR_THRESHOLD,
            high_volatility_threshold=config.REGIME_HIGH_VOLATILITY_THRESHOLD,
            money_flow_extreme_threshold=config.REGIME_MONEY_FLOW_EXTREME_THRESHOLD,
        )

        trend_bias = trend_context.get('trend_bias', 'neutral')
        regime = regime_context.get('regime', 'range')

        # 检查当前持仓是否触发止盈止损
        if position != 0:
            exit_signal = resolve_intrabar_tp_sl(position, entry_price, open_price, high, low, take_profit, stop_loss)

            if exit_signal:
                # 平仓
                exit_price = exit_signal['price']
                exit_reason = exit_signal['reason']

                # 计算PnL
                qty = abs(position)
                if position > 0:
                    pnl = (exit_price - entry_price) * qty
                else:
                    pnl = (entry_price - exit_price) * qty

                # 扣除手续费（开仓 + 平仓）
                entry_fee = entry_price * qty * fee_rate
                exit_fee = exit_price * qty * fee_rate
                total_fee = entry_fee + exit_fee
                net_pnl = pnl - total_fee

                balance += net_pnl

                trades.append({
                    'entry_time': bar_ts,
                    'entry_price': entry_price,
                    'exit_time': bar_ts,
                    'exit_price': exit_price,
                    'direction': 'long' if position > 0 else 'short',
                    'qty': qty,
                    'pnl': pnl,
                    'fee': total_fee,
                    'net_pnl': net_pnl,
                    'reason': exit_reason,
                    'balance_after': balance,
                })

                position = 0
                entry_price = 0

        # 决定是否开仓或调整仓位
        if position == 0:
            # 根据 trend 决定方向
            if trend_bias == 'long':
                # 做多
                position = (balance * position_size) / close
                entry_price = close

                # 扣除开仓手续费
                entry_fee = entry_price * position * fee_rate
                balance -= entry_fee

            elif trend_bias == 'short':
                # 做空
                position = -(balance * position_size) / close
                entry_price = close

                entry_fee = entry_price * abs(position) * fee_rate
                balance -= entry_fee

    # 统计
    log_info("=" * 80)
    log_info("回测结果")
    log_info("=" * 80)
    log_info(f"总交易次数: {len(trades)}")

    if len(trades) == 0:
        log_info("⚠️ 没有任何交易")
        return

    df_trades = pd.DataFrame(trades)

    tp_trades = df_trades[df_trades['reason'] == 'TP']
    sl_trades = df_trades[df_trades['reason'] == 'SL']

    win_rate = len(tp_trades) / len(trades) * 100 if len(trades) > 0 else 0

    avg_win = tp_trades['net_pnl'].mean() if len(tp_trades) > 0 else 0
    avg_loss = sl_trades['net_pnl'].mean() if len(sl_trades) > 0 else 0

    total_pnl = df_trades['net_pnl'].sum()
    total_fee = df_trades['fee'].sum()

    final_balance = balance
    roi = (final_balance - initial_balance) / initial_balance * 100

    log_info(f"止盈次数: {len(tp_trades)}")
    log_info(f"止损次数: {len(sl_trades)}")
    log_info(f"胜率: {win_rate:.1f}%")
    log_info("")
    log_info(f"平均止盈: {avg_win:.2f} USDT")
    log_info(f"平均止损: {avg_loss:.2f} USDT")
    log_info(f"盈亏比: {abs(avg_win / avg_loss):.2f}:1" if avg_loss != 0 else "N/A")
    log_info("")
    log_info(f"总手续费: {total_fee:.2f} USDT")
    log_info(f"净盈亏: {total_pnl:.2f} USDT")
    log_info(f"最终资金: {final_balance:.2f} USDT")
    log_info(f"ROI: {roi:+.2f}%")
    log_info("")

    # 按方向统计
    long_trades = df_trades[df_trades['direction'] == 'long']
    short_trades = df_trades[df_trades['direction'] == 'short']

    if len(long_trades) > 0:
        long_tp = len(long_trades[long_trades['reason'] == 'TP'])
        long_win_rate = long_tp / len(long_trades) * 100
        long_pnl = long_trades['net_pnl'].sum()
        log_info(f"做多: {len(long_trades)} 笔, 胜率 {long_win_rate:.1f}%, 净盈亏 {long_pnl:+.2f} USDT")

    if len(short_trades) > 0:
        short_tp = len(short_trades[short_trades['reason'] == 'TP'])
        short_win_rate = short_tp / len(short_trades) * 100
        short_pnl = short_trades['net_pnl'].sum()
        log_info(f"做空: {len(short_trades)} 笔, 胜率 {short_win_rate:.1f}%, 净盈亏 {short_pnl:+.2f} USDT")

    log_info("")
    log_info("最近10笔交易:")
    for _, trade in df_trades.tail(10).iterrows():
        log_info(f"  {trade['entry_time']} {trade['direction']:5s} "
                f"entry={trade['entry_price']:.2f} exit={trade['exit_price']:.2f} "
                f"{trade['reason']:2s} pnl={trade['net_pnl']:+7.2f}")

    return df_trades


if __name__ == '__main__':
    simple_rule_backtest()
