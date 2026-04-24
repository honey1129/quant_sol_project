import os
from datetime import datetime
import csv
import joblib
import traceback
import math
from core.strategy_core import StrategyCore
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from core import position_manager, okx_api, ml_feature_engineering, signal_engine
from config import config
from utils.utils import log_info, log_error,LOGS_DIR


def mark_to_market_equity(balance, position, entry_price, mark_price):
    equity = float(balance)
    position = float(position)
    entry_price = float(entry_price) if entry_price else 0.0

    if position == 0 or entry_price <= 0:
        return equity

    return equity + (float(mark_price) - entry_price) * position


def resolve_intrabar_tp_sl(position, entry_price, bar_open, bar_high, bar_low, take_profit, stop_loss, worst_case=True):
    if position == 0 or entry_price <= 0:
        return None

    if position > 0:
        take_profit_price = entry_price * (1 + take_profit)
        stop_loss_price = entry_price * (1 - stop_loss)
        hit_tp = bar_open >= take_profit_price or bar_high >= take_profit_price
        hit_sl = bar_open <= stop_loss_price or bar_low <= stop_loss_price
    else:
        take_profit_price = entry_price * (1 - take_profit)
        stop_loss_price = entry_price * (1 + stop_loss)
        hit_tp = bar_open <= take_profit_price or bar_low <= take_profit_price
        hit_sl = bar_open >= stop_loss_price or bar_high >= stop_loss_price

    if not hit_tp and not hit_sl:
        return None

    if hit_tp and hit_sl:
        exit_reason = "SL" if worst_case else "TP"
    elif hit_sl:
        exit_reason = "SL"
    else:
        exit_reason = "TP"

    if position > 0:
        if exit_reason == "SL":
            trigger_price = min(bar_open, stop_loss_price)
        else:
            trigger_price = max(bar_open, take_profit_price)
    else:
        if exit_reason == "SL":
            trigger_price = max(bar_open, stop_loss_price)
        else:
            trigger_price = min(bar_open, take_profit_price)

    return {
        "reason": exit_reason,
        "trigger_price": float(trigger_price),
        "tp_price": float(take_profit_price),
        "sl_price": float(stop_loss_price),
    }


class Backtester:
    def __init__(
        self,
        interval,
        window,
        *,
        data_dict=None,
        reward_risk=None,
        precomputed_data=None,
        feature_cols=None,
        models=None,
        model_weights=None,
        funding_history=None,
        enable_csv_dump=True,
        show_progress=True,
        emit_diagnostics=True,
    ):
        self.interval = interval
        self.window = window
        self.in_high_conf = False
        self.hold_bars = 0
        self.peak_price = None
        self.enable_csv_dump = bool(enable_csv_dump)
        self.show_progress = bool(show_progress)
        self.emit_diagnostics = bool(emit_diagnostics)

        # 拉取多周期数据以及计算reward_risk
        if data_dict is None:
            self.data_dict, self.reward_risk = self._load_data()
        else:
            self.data_dict = data_dict
            self.reward_risk = float(reward_risk if reward_risk is not None else config.KELLY_REWARD_RISK)

        # 特征工程
        if precomputed_data is None:
            merged_df = ml_feature_engineering.merge_multi_period_features(self.data_dict)
            merged_df = ml_feature_engineering.add_advanced_features(merged_df)
            self.data = merged_df.dropna().copy()
        else:
            self.data = precomputed_data.copy()

        # 读取训练时的特征列表
        self.feature_cols = feature_cols if feature_cols is not None else joblib.load(config.FEATURE_LIST_PATH)

        # 加载模型与权重
        self.models = models if models is not None else signal_engine.load_models(config.MODEL_PATHS)
        self.model_weights = model_weights if model_weights is not None else config.MODEL_WEIGHTS

        # 初始化仓位和资金
        self.position = 0
        self.entry_price = 0
        self.balance = config.INITIAL_BALANCE
        self.max_balance = self.balance
        self.max_drawdown = 0.0
        self.trade_log = []
        self.funding_log = []
        self.fee_rate = config.FEE_RATE
        self.slippage_bps = float(config.BACKTEST_SLIPPAGE_BPS)
        self.enable_funding = bool(config.BACKTEST_ENABLE_FUNDING)
        self.enable_intrabar_tp_sl = bool(config.BACKTEST_INTRABAR_TP_SL)
        self.worst_case_tp_sl = bool(config.BACKTEST_WORST_CASE_TP_SL)
        self.fee_paid_total = 0.0
        self.slippage_paid_total = 0.0
        self.funding_pnl_total = 0.0
        self.tp_exit_count = 0
        self.sl_exit_count = 0
        self.final_equity = self.balance

        # 初始化 position_manager
        self.position_manager = position_manager.PositionManager()
        # ✅ 统一核心策略（以回测为准）
        self.core = StrategyCore(
            self.position_manager,
            threshold_long=config.THRESHOLD_LONG,
            threshold_short=config.THRESHOLD_SHORT,
            take_profit=config.TAKE_PROFIT,
            stop_loss=config.STOP_LOSS,
            adaptive_tp_sl_enabled=config.ADAPTIVE_TP_SL_ENABLED,
            atr_take_profit_multiplier=config.ATR_TAKE_PROFIT_MULTIPLIER,
            atr_stop_loss_multiplier=config.ATR_STOP_LOSS_MULTIPLIER,
            volatility_take_profit_multiplier=config.VOLATILITY_TAKE_PROFIT_MULTIPLIER,
            volatility_stop_loss_multiplier=config.VOLATILITY_STOP_LOSS_MULTIPLIER,
            adaptive_take_profit_min=config.ADAPTIVE_TAKE_PROFIT_MIN,
            adaptive_take_profit_max=config.ADAPTIVE_TAKE_PROFIT_MAX,
            adaptive_stop_loss_min=config.ADAPTIVE_STOP_LOSS_MIN,
            adaptive_stop_loss_max=config.ADAPTIVE_STOP_LOSS_MAX,
            min_hold_bars=config.MIN_HOLD_BARS,
            add_threshold=config.ADD_THRESHOLD,
            max_rebalance_ratio=config.MAX_REBALANCE_RATIO,
            min_adjust_amount=float(config.MIN_ADJUST_AMOUNT),
            signal_min_prob_diff=config.SIGNAL_MIN_PROB_DIFF,
            min_signal_target_ratio=config.MIN_SIGNAL_TARGET_RATIO,
            reverse_signal_min_prob_diff=config.REVERSE_SIGNAL_MIN_PROB_DIFF,
            reverse_min_target_ratio=config.REVERSE_MIN_TARGET_RATIO,
            reward_risk=float(self.reward_risk),
        )
        if self.emit_diagnostics:
            self._log_intrabar_range_diagnostics()

        self.price_series = self.data['5m_close']
        self.funding_history = (
            funding_history.copy()
            if funding_history is not None
            else self._load_funding_history()
        )
        self.next_funding_idx = 0

    def _load_data(self):
        log_info(f"从OKX拉取历史数据: {self.interval}, {self.window}根K线")
        client = okx_api.OKXClient()
        all_data = client.fetch_data()
        reward_risk = float(config.KELLY_REWARD_RISK)
        log_info(f"回测使用固定 reward_risk={reward_risk:.4f}")
        return all_data,reward_risk

    def _load_funding_history(self):
        if not self.enable_funding or self.data.empty:
            return pd.DataFrame(columns=['funding_time', 'funding_rate'])

        start_ts = self.data.index.min()
        end_ts = self.data.index.max()
        funding_span = end_ts - start_ts
        estimated_records = max(16, math.ceil(funding_span.total_seconds() / (8 * 3600)) + 16)
        record_limit = max(int(config.BACKTEST_FUNDING_HISTORY_LIMIT), estimated_records)

        client = okx_api.OKXClient()
        funding_df = client.fetch_funding_rate_history(max_records=record_limit)
        if funding_df.empty:
            log_info("回测未获取到 funding 历史，按 0 处理")
            return funding_df

        funding_df = funding_df[
            (funding_df['funding_time'] >= start_ts) &
            (funding_df['funding_time'] <= end_ts)
        ].copy()

        log_info(f"回测加载 funding 记录: {len(funding_df)} 条")
        return funding_df.reset_index(drop=True)

    def _log_intrabar_range_diagnostics(self):
        required_cols = {'5m_open', '5m_high', '5m_low'}
        if self.data.empty or not required_cols.issubset(self.data.columns):
            return

        base_price = self.data['5m_open'].replace(0, np.nan)
        up_move = ((self.data['5m_high'] - base_price) / base_price).dropna()
        down_move = ((base_price - self.data['5m_low']) / base_price).dropna()

        if up_move.empty or down_move.empty:
            return

        up_p95 = float(up_move.quantile(0.95))
        down_p95 = float(down_move.quantile(0.95))
        log_info(f"5m振幅参考: 上行95分位={up_p95:.2%}, 下行95分位={down_p95:.2%}")

        if self.core.adaptive_tp_sl_enabled:
            required_adaptive_cols = ['5m_close', '5m_atr', 'volatility_15']
            if not set(required_adaptive_cols).issubset(self.data.columns):
                return

            risk_df = self.data[required_adaptive_cols].replace([np.inf, -np.inf], np.nan).dropna()
            if risk_df.empty:
                return

            tp_values = []
            sl_values = []
            for row in risk_df.itertuples(index=False):
                close_price = float(row[0])
                atr_value = float(row[1])
                volatility = float(row[2])
                atr_ratio = atr_value / close_price if close_price > 0 else None
                take_profit, stop_loss = self.core.resolve_risk_thresholds(
                    volatility=volatility,
                    atr_ratio=atr_ratio,
                )
                tp_values.append(take_profit)
                sl_values.append(stop_loss)

            if tp_values and sl_values:
                log_info(
                    "自适应TP/SL参考: "
                    f"TP中位={np.median(tp_values):.2%}, TP95分位={np.quantile(tp_values, 0.95):.2%}, "
                    f"SL中位={np.median(sl_values):.2%}, SL95分位={np.quantile(sl_values, 0.95):.2%}"
                )
            return

        if self.core.take_profit > up_p95 * 1.5:
            log_info(
                f"⚠ 当前 TAKE_PROFIT={self.core.take_profit:.2%} 明显高于常见单根5m振幅，"
                "止盈在 intrabar 模式下可能极少触发"
            )
        if self.core.stop_loss > down_p95 * 1.5:
            log_info(
                f"⚠ 当前 STOP_LOSS={self.core.stop_loss:.2%} 明显高于常见单根5m振幅，"
                "止损在 intrabar 模式下可能极少触发"
            )

    def _get_trade_side(self, current_pos, delta, action):
        if action == "CLOSE":
            return "sell" if current_pos > 0 else "buy"

        if current_pos > 0:
            return "buy" if delta > 0 else "sell"
        if current_pos < 0:
            return "sell" if delta < 0 else "buy"
        return "buy" if delta > 0 else "sell"

    def _apply_slippage(self, price, side, bar_low=None, bar_high=None):
        slip_ratio = self.slippage_bps / 10000.0
        if side == "buy":
            exec_price = price * (1 + slip_ratio)
            if bar_high is not None:
                exec_price = min(exec_price, bar_high)
            return exec_price

        exec_price = price * (1 - slip_ratio)
        if bar_low is not None:
            exec_price = max(exec_price, bar_low)
        return exec_price

    def _apply_trade_costs(self, qty, reference_price, exec_price):
        fee = abs(qty * exec_price * self.fee_rate)
        slippage_cost = abs(qty) * abs(exec_price - reference_price)
        self.balance -= fee
        self.fee_paid_total += fee
        self.slippage_paid_total += slippage_cost
        return fee, slippage_cost

    def _apply_funding_until(self, ts):
        if self.funding_history.empty:
            return

        while self.next_funding_idx < len(self.funding_history):
            event = self.funding_history.iloc[self.next_funding_idx]
            funding_time = event['funding_time']
            if funding_time > ts:
                break

            if self.position != 0:
                mark_price = float(self.price_series.asof(funding_time))
                funding_rate = float(event['funding_rate'])
                funding_pnl = -self.position * mark_price * funding_rate
                self.balance += funding_pnl
                self.funding_pnl_total += funding_pnl
                self.funding_log.append((funding_time, "资金费", mark_price, self.position, self.balance))

            self.next_funding_idx += 1

    def _mark_to_market_equity(self, mark_price):
        return mark_to_market_equity(self.balance, self.position, self.entry_price, mark_price)

    def _maybe_execute_intrabar_tp_sl(self, exec_row, take_profit=None, stop_loss=None):
        if not self.enable_intrabar_tp_sl or self.position == 0 or self.entry_price <= 0:
            return False

        bar_open = float(exec_row.get('5m_open', exec_row['5m_close']))
        bar_high = float(exec_row['5m_high'])
        bar_low = float(exec_row['5m_low'])
        if take_profit is None or stop_loss is None:
            take_profit, stop_loss = self.core.get_risk_thresholds()

        hit = resolve_intrabar_tp_sl(
            self.position,
            self.entry_price,
            bar_open,
            bar_high,
            bar_low,
            take_profit,
            stop_loss,
            worst_case=self.worst_case_tp_sl,
        )
        if hit is None:
            return False

        pos_to_close = self.position
        close_side = "sell" if pos_to_close > 0 else "buy"
        reference_price = float(hit["trigger_price"])
        exec_price = self._apply_slippage(reference_price, close_side, bar_low=bar_low, bar_high=bar_high)
        profit = (exec_price - self.entry_price) * pos_to_close
        self.balance += profit
        self._apply_trade_costs(pos_to_close, reference_price, exec_price)

        action = "止损" if hit["reason"] == "SL" else "止盈"
        if hit["reason"] == "SL":
            self.sl_exit_count += 1
        else:
            self.tp_exit_count += 1

        self.trade_log.append((exec_row.name, action, exec_price, pos_to_close, self.balance))

        self.position = 0.0
        self.entry_price = 0.0
        self.hold_bars = 0
        self.core.set_state(0.0, 0.0, 0)
        return True


    def run_backtest(self):

        # ========== 预计算信号 ==========
        self.data[['long_prob', 'short_prob']] = self.data.apply(
            self._predict_row, axis=1, result_type="expand"
        )

        if len(self.data) < 2:
            log_error("回测样本不足，无法使用已收盘信号 -> 下一根开盘成交的模式")
            return

        # 用上一根已收盘 bar 生成信号，在下一根 bar 开盘成交，避免同 bar 未来函数。
        for i in tqdm(range(1, len(self.data)), disable=not self.show_progress):
            signal_row = self.data.iloc[i - 1]
            exec_row = self.data.iloc[i]
            price = float(exec_row.get('5m_open', exec_row['5m_close']))
            bar_high = float(exec_row['5m_high'])
            bar_low = float(exec_row['5m_low'])
            bar_close = float(exec_row['5m_close'])
            long_prob = signal_row['long_prob']
            short_prob = signal_row['short_prob']
            money_flow_ratio = signal_row['money_flow_ratio']
            volatility = signal_row['volatility_15']
            close_price = float(signal_row['5m_close'])
            atr_value = signal_row.get('5m_atr')
            atr_ratio = None

            if pd.isna(volatility):
                continue
            volatility = float(volatility)
            if pd.notna(atr_value) and close_price > 0:
                atr_ratio = float(atr_value) / close_price

            self._apply_funding_until(exec_row.name)

            # ===== 将回测状态同步进 core =====
            self.core.set_state(self.position, self.entry_price, self.hold_bars)
            decision_equity = self._mark_to_market_equity(price)

            out = self.core.on_bar(
                price=price,
                equity=decision_equity,
                long_prob=long_prob,
                short_prob=short_prob,
                money_flow_ratio=money_flow_ratio,
                volatility=volatility,
                atr_ratio=atr_ratio,
            )

            action = out["action"]
            delta = float(out["delta_qty"])
            take_profit, stop_loss = self.core.get_risk_thresholds()

            if action == "CLOSE":
                pos_to_close = self.position
                entry_price = self.entry_price
                side = self._get_trade_side(pos_to_close, delta, action)
                exec_price = self._apply_slippage(price, side, bar_low=bar_low, bar_high=bar_high)

                profit = (exec_price - entry_price) * pos_to_close
                self.balance += profit
                self._apply_trade_costs(pos_to_close, price, exec_price)

                act = "平仓" if out.get("reason") == "TP/SL" else "反向平仓"
                self.trade_log.append((exec_row.name, act, exec_price, pos_to_close, self.balance))

            elif action == "OPEN":
                new_pos, _, _ = self.core.get_state()
                side = self._get_trade_side(self.position, delta, action)
                exec_price = self._apply_slippage(price, side, bar_low=bar_low, bar_high=bar_high)
                self._apply_trade_costs(new_pos, price, exec_price)
                self.core.set_state(new_pos, exec_price, self.core.get_state()[2])

                act = "开多" if new_pos > 0 else "开空"
                self.trade_log.append((exec_row.name, act, exec_price, new_pos, self.balance))

            elif action == "REBALANCE":
                old_pos = self.position
                old_entry = self.entry_price
                new_pos, _, _ = self.core.get_state()
                side = self._get_trade_side(old_pos, delta, action)
                exec_price = self._apply_slippage(price, side, bar_low=bar_low, bar_high=bar_high)

                if old_pos != 0 and np.sign(delta) != np.sign(old_pos):
                    reduced_qty = min(abs(delta), abs(old_pos))
                    closed_pos = math.copysign(reduced_qty, old_pos)
                    realized_profit = (exec_price - old_entry) * closed_pos
                    self.balance += realized_profit

                self._apply_trade_costs(delta, price, exec_price)

                if abs(new_pos) > abs(old_pos):
                    existing_qty = abs(old_pos)
                    added_qty = abs(delta)
                    new_entry = (
                        (existing_qty * old_entry) +
                        (added_qty * exec_price)
                    ) / max(existing_qty + added_qty, 1e-9)
                else:
                    new_entry = old_entry
                self.core.set_state(new_pos, new_entry, self.core.get_state()[2])

                if new_pos > 0:
                    act = "加多" if delta > 0 else "减多"
                else:
                    act = "减空" if delta > 0 else "加空"
                self.trade_log.append((exec_row.name, act, exec_price, new_pos, self.balance))

            self.position, self.entry_price, self.hold_bars = self.core.get_state()
            self._maybe_execute_intrabar_tp_sl(exec_row, take_profit=take_profit, stop_loss=stop_loss)
            self.position, self.entry_price, self.hold_bars = self.core.get_state()
            equity = self._mark_to_market_equity(bar_close)
            self.final_equity = equity

            self.max_balance = max(self.max_balance, equity)
            if self.max_balance > 0:
                drawdown = (equity - self.max_balance) / self.max_balance
                self.max_drawdown = min(self.max_drawdown, drawdown)


        return self._summary()

    def _predict_row(self, row):
        """
        复用实盘信号融合逻辑，保持一致性
        """
        X_row = row[self.feature_cols].values.reshape(1, -1).astype(float)
        X_row = pd.DataFrame(X_row, columns=self.feature_cols)

        weighted_sum = np.zeros(2)
        total_weight = sum(self.model_weights.values())

        for name, model in self.models.items():
            prob = model.predict_proba(X_row)[0]
            weight = self.model_weights.get(name, 1.0)
            weighted_sum += prob * weight

        avg_pred = weighted_sum / total_weight
        long_prob, short_prob = avg_pred[1], avg_pred[0]
        return long_prob, short_prob

    def _summary(self):
        pnl = self.final_equity - config.INITIAL_BALANCE
        drawdown = self.max_drawdown

        log_info("回测完成 ✅")
        log_info(f"期末净值: {self.final_equity:.2f} USDT")
        if abs(self.final_equity - self.balance) > 1e-9:
            log_info(f"期末已实现余额: {self.balance:.2f} USDT")
        if self.position != 0:
            log_info(f"期末持仓: {self.position:.4f} @ {self.entry_price:.4f}")
        log_info(f"累计收益: {pnl:.2f} USDT ({pnl / config.INITIAL_BALANCE * 100:.2f}%)")
        log_info(f"最大回撤: {drawdown * 100:.2f}%")
        log_info(f"交易次数: {len(self.trade_log)}")
        log_info(f"资金费事件数: {len(self.funding_log)}")
        log_info(f"止盈次数: {self.tp_exit_count}")
        log_info(f"止损次数: {self.sl_exit_count}")
        log_info(f"手续费合计: {self.fee_paid_total:.2f} USDT")
        log_info(f"滑点成本合计: {self.slippage_paid_total:.2f} USDT")
        log_info(f"资金费净额: {self.funding_pnl_total:.2f} USDT")
        log_info(f"交易记录示例: {self.trade_log[-5:]}")
        if self.enable_csv_dump:
            self.dump_trade_log_to_csv(pnl, drawdown)

        return {
            "final_equity": float(self.final_equity),
            "final_balance": float(self.balance),
            "pnl": float(pnl),
            "return_pct": float(pnl / config.INITIAL_BALANCE * 100),
            "max_drawdown_pct": float(drawdown * 100),
            "trade_count": int(len(self.trade_log)),
            "funding_event_count": int(len(self.funding_log)),
            "take_profit_count": int(self.tp_exit_count),
            "stop_loss_count": int(self.sl_exit_count),
            "fees_paid": float(self.fee_paid_total),
            "slippage_cost": float(self.slippage_paid_total),
            "funding_pnl": float(self.funding_pnl_total),
            "ending_position": float(self.position),
            "ending_entry_price": float(self.entry_price),
        }

    def dump_trade_log_to_csv(self,pnl, drawdown):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backtest_log_path = os.path.join(LOGS_DIR, f"backtest_{self.interval}_{ts}.csv")
        with open(backtest_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "action", "price",  "position","balance"])

            writer.writerows([
                (t, a, round(p, 4), round(pos, 4),round(b, 2))
                for (t, a, p,pos, b) in self.trade_log
            ])
            writer.writerow([])
            writer.writerow(["# Summary"])
            writer.writerow(["Final Equity", round(self.final_equity, 2)])
            writer.writerow(["Final Balance", round(self.balance, 2)])
            writer.writerow(["Ending Position", round(self.position, 4)])
            writer.writerow(["Ending Entry Price", round(self.entry_price, 4)])
            writer.writerow(["PnL (USDT)", round(pnl, 2)])
            writer.writerow([
                "Return (%)",
                round(pnl / config.INITIAL_BALANCE * 100, 2)
            ])
            writer.writerow([
                "Max Drawdown (%)",
                round(drawdown * 100, 2)
            ])
            writer.writerow([
                "Trade Count",
                len(self.trade_log)
            ])
            writer.writerow([
                "Funding Event Count",
                len(self.funding_log)
            ])
            writer.writerow([
                "Take Profit Count",
                self.tp_exit_count
            ])
            writer.writerow([
                "Stop Loss Count",
                self.sl_exit_count
            ])
            writer.writerow([
                "Fees Paid (USDT)",
                round(self.fee_paid_total, 2)
            ])
            writer.writerow([
                "Slippage Cost (USDT)",
                round(self.slippage_paid_total, 2)
            ])
            writer.writerow([
                "Funding PnL (USDT)",
                round(self.funding_pnl_total, 2)
            ])
            if self.funding_log:
                writer.writerow([])
                writer.writerow(["# Funding"])
                writer.writerow(["timestamp", "action", "price", "position", "balance"])
                writer.writerows([
                    (t, a, round(p, 4), round(pos, 4), round(b, 2))
                    for (t, a, p, pos, b) in self.funding_log
                ])



if __name__ == '__main__':
    try:
        base_interval = config.INTERVALS[0] if config.INTERVALS else "5m"
        window = config.WINDOWS.get(base_interval, 1000)
        log_info("\n==== 开始多周期融合回测 ====")
        backtester = Backtester("multi_period", window)
        backtester.run_backtest()
    except Exception as e:
        log_error(traceback.format_exc())
