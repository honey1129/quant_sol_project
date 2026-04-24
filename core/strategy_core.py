# core/strategy_core.py
import math
import numpy as np

class StrategyCore:

    def __init__(
        self,
        position_manager,
        *,
        threshold_long: float,
        threshold_short: float,
        take_profit: float,
        stop_loss: float,
        adaptive_tp_sl_enabled: bool = True,
        atr_take_profit_multiplier: float = 3.0,
        atr_stop_loss_multiplier: float = 1.5,
        volatility_take_profit_multiplier: float = 4.5,
        volatility_stop_loss_multiplier: float = 2.0,
        adaptive_take_profit_min: float = 0.006,
        adaptive_take_profit_max: float = 0.03,
        adaptive_stop_loss_min: float = 0.004,
        adaptive_stop_loss_max: float = 0.02,
        min_hold_bars: int = 8,
        add_threshold: float = 0.15,
        max_rebalance_ratio: float = 0.3,
        min_adjust_amount: float = 10.0,
        signal_min_prob_diff: float = 0.18,
        min_signal_target_ratio: float = 0.08,
        reverse_signal_min_prob_diff: float = 0.26,
        reverse_min_target_ratio: float = 0.12,
        reward_risk: float = 1.0,
    ):
        self.pm = position_manager

        self.threshold_long = float(threshold_long)
        self.threshold_short = float(threshold_short)
        self.take_profit = float(take_profit)
        self.stop_loss = float(stop_loss)
        self.adaptive_tp_sl_enabled = bool(adaptive_tp_sl_enabled)
        self.atr_take_profit_multiplier = float(atr_take_profit_multiplier)
        self.atr_stop_loss_multiplier = float(atr_stop_loss_multiplier)
        self.volatility_take_profit_multiplier = float(volatility_take_profit_multiplier)
        self.volatility_stop_loss_multiplier = float(volatility_stop_loss_multiplier)
        self.adaptive_take_profit_min = float(adaptive_take_profit_min)
        self.adaptive_take_profit_max = float(adaptive_take_profit_max)
        self.adaptive_stop_loss_min = float(adaptive_stop_loss_min)
        self.adaptive_stop_loss_max = float(adaptive_stop_loss_max)

        self.min_hold_bars = int(min_hold_bars)
        self.add_threshold = float(add_threshold)
        self.max_rebalance_ratio = float(max_rebalance_ratio)
        self.min_adjust_amount = float(min_adjust_amount)
        self.signal_min_prob_diff = float(signal_min_prob_diff)
        self.min_signal_target_ratio = float(min_signal_target_ratio)
        self.reverse_signal_min_prob_diff = float(reverse_signal_min_prob_diff)
        self.reverse_min_target_ratio = float(reverse_min_target_ratio)

        self.reward_risk = float(reward_risk)

        self.position = 0.0
        self.entry_price = 0.0
        self.hold_bars = 0
        self.current_take_profit = self.take_profit
        self.current_stop_loss = self.stop_loss

    def reset_risk_thresholds(self):
        self.current_take_profit = self.take_profit
        self.current_stop_loss = self.stop_loss

    def get_risk_thresholds(self):
        return self.current_take_profit, self.current_stop_loss

    def _clean_optional_ratio(self, value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        return value

    def resolve_risk_thresholds(self, *, volatility: float = None, atr_ratio: float = None):
        if not self.adaptive_tp_sl_enabled:
            return self.take_profit, self.stop_loss

        atr_ratio = self._clean_optional_ratio(atr_ratio)
        volatility = self._clean_optional_ratio(volatility)

        tp_candidates = []
        sl_candidates = []

        if atr_ratio is not None:
            tp_candidates.append(atr_ratio * self.atr_take_profit_multiplier)
            sl_candidates.append(atr_ratio * self.atr_stop_loss_multiplier)

        if volatility is not None:
            tp_candidates.append(volatility * self.volatility_take_profit_multiplier)
            sl_candidates.append(volatility * self.volatility_stop_loss_multiplier)

        take_profit = max(tp_candidates) if tp_candidates else self.take_profit
        stop_loss = max(sl_candidates) if sl_candidates else self.stop_loss

        take_profit = float(np.clip(
            take_profit,
            self.adaptive_take_profit_min,
            self.adaptive_take_profit_max,
        ))
        stop_loss = float(np.clip(
            stop_loss,
            self.adaptive_stop_loss_min,
            self.adaptive_stop_loss_max,
        ))
        return take_profit, stop_loss

    def update_risk_thresholds(self, *, volatility: float = None, atr_ratio: float = None):
        take_profit, stop_loss = self.resolve_risk_thresholds(
            volatility=volatility,
            atr_ratio=atr_ratio,
        )
        self.current_take_profit = take_profit
        self.current_stop_loss = stop_loss
        return take_profit, stop_loss

    def set_state(self, position: float, entry_price: float, hold_bars: int = None):
        self.position = float(position)
        self.entry_price = float(entry_price)
        if hold_bars is not None:
            self.hold_bars = int(hold_bars)
        if self.position == 0 or self.entry_price <= 0:
            self.reset_risk_thresholds()

    def get_state(self):
        return self.position, self.entry_price, self.hold_bars

    def _resolve_directional_target_ratio(
        self,
        *,
        long_prob: float,
        short_prob: float,
        money_flow_ratio: float,
        volatility: float,
    ):
        long_prob = float(long_prob)
        short_prob = float(short_prob)
        prob_gap = abs(long_prob - short_prob)

        if long_prob >= short_prob:
            dominant_prob = long_prob
            if dominant_prob <= self.threshold_long or prob_gap < self.signal_min_prob_diff:
                return 0.0, prob_gap, dominant_prob

            target_ratio = float(self.pm.calculate_target_ratio(
                long_prob,
                money_flow_ratio,
                volatility,
                self.reward_risk,
            ))
            if target_ratio < self.min_signal_target_ratio:
                return 0.0, prob_gap, dominant_prob
            return target_ratio, prob_gap, dominant_prob

        dominant_prob = short_prob
        if dominant_prob <= self.threshold_short or prob_gap < self.signal_min_prob_diff:
            return 0.0, prob_gap, dominant_prob

        target_ratio = float(self.pm.calculate_target_ratio(
            short_prob,
            money_flow_ratio,
            volatility,
            self.reward_risk,
        ))
        if target_ratio < self.min_signal_target_ratio:
            return 0.0, prob_gap, dominant_prob
        return -target_ratio, prob_gap, dominant_prob

    def on_bar(
        self,
        *,
        price: float,
        equity: float,
        long_prob: float,
        short_prob: float,
        money_flow_ratio: float,
        volatility: float,
        atr_ratio: float = None,
    ):
        """
        单根 5m bar 决策一次（与回测一致）
        """
        price = float(price)
        equity = float(equity)
        pos = float(self.position)
        take_profit, stop_loss = self.update_risk_thresholds(
            volatility=volatility,
            atr_ratio=atr_ratio,
        )

        # ======================
        # 1) 持仓 -> 止盈止损
        # ======================
        if pos != 0:
            pnl_pct = (price - self.entry_price) / self.entry_price if pos > 0 else (self.entry_price - price) / self.entry_price
            if pnl_pct >= take_profit or pnl_pct <= -stop_loss:
                delta_qty = -pos
                self.position = 0.0
                self.entry_price = 0.0
                self.hold_bars = 0
                self.reset_risk_thresholds()
                return {
                    "action": "CLOSE",
                    "delta_qty": delta_qty,
                    "target_ratio": 0.0,
                    "target_position": 0.0,
                    "reason": "TP/SL",
                }

        # ======================
        # 2) 计算目标仓位档位
        # ======================
        target_ratio, signal_prob_gap, dominant_prob = self._resolve_directional_target_ratio(
            long_prob=long_prob,
            short_prob=short_prob,
            money_flow_ratio=money_flow_ratio,
            volatility=volatility,
        )
        target_position = target_ratio * equity / price

        # ======================
        # 3) 空仓 -> 只允许开仓
        # ======================
        if pos == 0:
            if abs(target_position * price) >= self.min_adjust_amount and target_position != 0:
                self.position = target_position
                self.entry_price = price
                self.hold_bars = 0
                return {
                    "action": "OPEN",
                    "delta_qty": target_position,
                    "target_ratio": target_ratio,
                    "target_position": target_position,
                    "reason": "OpenFromFlat",
                }
            return {
                "action": "HOLD",
                "delta_qty": 0.0,
                "target_ratio": target_ratio,
                "target_position": target_position,
                "reason": "FlatNoSignal",
            }

        # ======================
        # 4) 持仓 -> 最小持有期
        # ======================
        self.hold_bars += 1
        if self.hold_bars < self.min_hold_bars:
            return {
                "action": "HOLD",
                "delta_qty": 0.0,
                "target_ratio": target_ratio,
                "target_position": target_position,
                "reason": f"MinHold({self.hold_bars}/{self.min_hold_bars})",
            }

        # ======================
        # 5) 同方向 -> 分段加 / 减仓
        # ======================
        same_direction = (pos > 0 and target_position > 0) or (pos < 0 and target_position < 0)

        if same_direction:
            raw_delta = target_position - pos
            diff_ratio = raw_delta / max(abs(pos), 1e-9)

            if abs(diff_ratio) >= self.add_threshold:
                delta = raw_delta
                delta = float(np.clip(
                    delta,
                    -self.max_rebalance_ratio * abs(pos),
                    self.max_rebalance_ratio * abs(pos)
                ))

                if abs(delta * price) >= self.min_adjust_amount:
                    new_position = pos + delta

                    # 同方向加仓时，更新持仓均价；减仓则保留剩余仓位原均价。
                    if np.sign(delta) == np.sign(pos):
                        existing_qty = abs(pos)
                        added_qty = abs(delta)
                        total_qty = existing_qty + added_qty
                        if total_qty > 0:
                            self.entry_price = (
                                (existing_qty * self.entry_price) +
                                (added_qty * price)
                            ) / total_qty

                    self.position = new_position
                    return {
                        "action": "REBALANCE",
                        "delta_qty": delta,
                        "target_ratio": target_ratio,
                        "target_position": target_position,
                        "reason": "SameDirRebalance",
                    }

            return {
                "action": "HOLD",
                "delta_qty": 0.0,
                "target_ratio": target_ratio,
                "target_position": target_position,
                "reason": "SameDirNoRebalance",
            }

        # ======================
        # 6) 反向信号 -> 必须先清仓
        # ======================
        if abs(target_position) > 0:
            reverse_signal_is_strong = (
                signal_prob_gap >= self.reverse_signal_min_prob_diff and
                abs(target_ratio) >= self.reverse_min_target_ratio
            )
            if reverse_signal_is_strong:
                delta_qty = -pos
                self.position = 0.0
                self.entry_price = 0.0
                self.hold_bars = 0
                self.reset_risk_thresholds()
                return {
                    "action": "CLOSE",
                    "delta_qty": delta_qty,
                    "target_ratio": target_ratio,
                    "target_position": target_position,
                    "reason": "ReverseClose",
                }

            return {
                "action": "HOLD",
                "delta_qty": 0.0,
                "target_ratio": target_ratio,
                "target_position": target_position,
                "reason": (
                    f"WeakReverseSignal(gap={signal_prob_gap:.3f},"
                    f"dominant={dominant_prob:.3f},ratio={abs(target_ratio):.3f})"
                ),
            }

        return {
            "action": "HOLD",
            "delta_qty": 0.0,
            "target_ratio": target_ratio,
            "target_position": target_position,
            "reason": "NoSignalKeep",
        }
