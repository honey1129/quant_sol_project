# core/strategy_core.py
import math
import numpy as np

from core.trend_filter import trend_allows_direction
from core.regime_filter import regime_allows_direction, regime_reason


def _risk_payload(risk_decision):
    if risk_decision is None:
        return None
    return {
        "enabled": bool(risk_decision.enabled),
        "signal_strength": float(risk_decision.signal_strength),
        "volatility_ratio": float(risk_decision.volatility_ratio),
        "atr_ratio": float(risk_decision.atr_ratio),
        "trend_aligned": bool(risk_decision.trend_aligned),
        "risk_multiplier": float(risk_decision.risk_multiplier),
        "effective_leverage": int(risk_decision.effective_leverage),
        "max_position_ratio": float(risk_decision.max_position_ratio),
        "reasons": list(getattr(risk_decision, "reasons", ())),
    }


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
        reverse_exit_consecutive_bars: int = 0,
        reverse_exit_min_prob_diff: float = None,
        reward_risk: float = 1.0,
        fee_rate: float = 0.0005,
        slippage_bps: float = 0.0,
        cost_buffer_multiplier: float = 2.0,
        min_expected_net_edge: float = 0.0,
        min_take_profit_to_stop_loss_ratio: float = 2.2,
        min_take_profit_cost_multiplier: float = 6.0,
        regime_high_vol_stop_loss_min: float = None,
        trade_cooldown_bars: int = 0,
        take_profit_cooldown_bars: int = None,
        stop_loss_cooldown_bars: int = None,
        trend_filter_enabled: bool = False,
        regime_filter_enabled: bool = False,
        regime_range_allow_trades: bool = True,
        regime_high_vol_allow_trades: bool = False,
        regime_range_threshold_bonus: float = 0.0,
        regime_high_vol_threshold_bonus: float = 0.0,
        regime_trend_against_block: bool = True,
        regime_range_target_multiplier: float = 1.0,
        regime_high_vol_target_multiplier: float = 1.0,
        regime_range_min_signal_target_ratio: float = None,
        regime_high_vol_min_signal_target_ratio: float = None,
        block_losing_position_adds: bool = True,
        dynamic_risk_controller=None,
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
        self.reverse_exit_consecutive_bars = max(0, int(reverse_exit_consecutive_bars))
        self.reverse_exit_min_prob_diff = (
            self.reverse_signal_min_prob_diff
            if reverse_exit_min_prob_diff is None
            else float(reverse_exit_min_prob_diff)
        )

        self.reward_risk = float(reward_risk)
        self.fee_rate = max(0.0, float(fee_rate))
        self.slippage_bps = max(0.0, float(slippage_bps))
        self.cost_buffer_multiplier = max(0.0, float(cost_buffer_multiplier))
        self.min_expected_net_edge = float(min_expected_net_edge)
        self.min_take_profit_to_stop_loss_ratio = max(0.0, float(min_take_profit_to_stop_loss_ratio))
        self.min_take_profit_cost_multiplier = max(0.0, float(min_take_profit_cost_multiplier))
        self.regime_high_vol_stop_loss_min = (
            None
            if regime_high_vol_stop_loss_min is None
            else max(0.0, float(regime_high_vol_stop_loss_min))
        )
        self.trade_cooldown_bars = max(0, int(trade_cooldown_bars))
        self.take_profit_cooldown_bars = (
            self.trade_cooldown_bars
            if take_profit_cooldown_bars is None
            else max(0, int(take_profit_cooldown_bars))
        )
        self.stop_loss_cooldown_bars = (
            max(self.trade_cooldown_bars, 36)
            if stop_loss_cooldown_bars is None
            else max(0, int(stop_loss_cooldown_bars))
        )
        self.trend_filter_enabled = bool(trend_filter_enabled)
        self.regime_filter_enabled = bool(regime_filter_enabled)
        self.regime_range_allow_trades = bool(regime_range_allow_trades)
        self.regime_high_vol_allow_trades = bool(regime_high_vol_allow_trades)
        self.regime_range_threshold_bonus = max(0.0, float(regime_range_threshold_bonus))
        self.regime_high_vol_threshold_bonus = max(0.0, float(regime_high_vol_threshold_bonus))
        self.regime_trend_against_block = bool(regime_trend_against_block)
        self.regime_range_target_multiplier = max(0.0, float(regime_range_target_multiplier))
        self.regime_high_vol_target_multiplier = max(0.0, float(regime_high_vol_target_multiplier))
        self.regime_range_min_signal_target_ratio = (
            None
            if regime_range_min_signal_target_ratio is None
            else max(0.0, float(regime_range_min_signal_target_ratio))
        )
        self.regime_high_vol_min_signal_target_ratio = (
            None
            if regime_high_vol_min_signal_target_ratio is None
            else max(0.0, float(regime_high_vol_min_signal_target_ratio))
        )
        self.block_losing_position_adds = bool(block_losing_position_adds)
        self.dynamic_risk_controller = dynamic_risk_controller

        self.position = 0.0
        self.entry_price = 0.0
        self.hold_bars = 0
        self.cooldown_bars_remaining = 0
        self.reverse_signal_bars = 0
        self.current_take_profit = self.take_profit
        self.current_stop_loss = self.stop_loss

    def reset_risk_thresholds(self):
        self.current_take_profit = self.take_profit
        self.current_stop_loss = self.stop_loss

    def get_risk_thresholds(self):
        return self.current_take_profit, self.current_stop_loss

    def get_cooldown_bars_remaining(self):
        return self.cooldown_bars_remaining

    def get_reverse_signal_bars(self):
        return self.reverse_signal_bars

    def estimated_round_trip_cost_ratio(self):
        slippage_ratio = self.slippage_bps / 10000.0
        return 2.0 * (self.fee_rate + slippage_ratio)

    def _expected_net_edge_ratio(self, dominant_prob, take_profit, stop_loss):
        dominant_prob = float(np.clip(float(dominant_prob), 0.0, 1.0))
        gross_edge = (
            dominant_prob * float(take_profit) -
            (1.0 - dominant_prob) * float(stop_loss)
        )
        cost_floor = self.estimated_round_trip_cost_ratio() * self.cost_buffer_multiplier
        return gross_edge - cost_floor

    def _next_hold_cooldown(self):
        return max(0, int(self.cooldown_bars_remaining) - 1)

    def _next_trade_cooldown(self, reason=None):
        if reason == "TakeProfit":
            return max(0, int(self.take_profit_cooldown_bars))
        if reason == "StopLoss":
            return max(0, int(self.stop_loss_cooldown_bars))
        return max(0, int(self.trade_cooldown_bars))

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

    def resolve_risk_thresholds(self, *, volatility: float = None, atr_ratio: float = None, market_regime: str = None):
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

        stop_loss_floor = self.adaptive_stop_loss_min
        if (
            self.regime_high_vol_stop_loss_min is not None
            and str(market_regime or "").lower() in {"high_vol", "range_high_vol"}
        ):
            stop_loss_floor = max(stop_loss_floor, self.regime_high_vol_stop_loss_min)

        stop_loss = float(np.clip(
            stop_loss,
            stop_loss_floor,
            self.adaptive_stop_loss_max,
        ))
        take_profit_floor = max(
            self.adaptive_take_profit_min,
            stop_loss * self.min_take_profit_to_stop_loss_ratio,
            self.estimated_round_trip_cost_ratio() * self.min_take_profit_cost_multiplier,
        )
        take_profit = float(np.clip(
            max(take_profit, take_profit_floor),
            self.adaptive_take_profit_min,
            self.adaptive_take_profit_max,
        ))
        return take_profit, stop_loss

    def update_risk_thresholds(self, *, volatility: float = None, atr_ratio: float = None, market_regime: str = None):
        take_profit, stop_loss = self.resolve_risk_thresholds(
            volatility=volatility,
            atr_ratio=atr_ratio,
            market_regime=market_regime,
        )
        self.current_take_profit = take_profit
        self.current_stop_loss = stop_loss
        return take_profit, stop_loss

    def set_state(
        self,
        position: float,
        entry_price: float,
        hold_bars: int = None,
        cooldown_bars_remaining: int = None,
        reverse_signal_bars: int = None,
    ):
        self.position = float(position)
        self.entry_price = float(entry_price)
        if hold_bars is not None:
            self.hold_bars = int(hold_bars)
        if cooldown_bars_remaining is not None:
            self.cooldown_bars_remaining = max(0, int(cooldown_bars_remaining))
        if reverse_signal_bars is not None:
            self.reverse_signal_bars = max(0, int(reverse_signal_bars))
        if self.position == 0 or self.entry_price <= 0:
            self.reverse_signal_bars = 0
            self.reset_risk_thresholds()

    def get_state(self):
        return self.position, self.entry_price, self.hold_bars

    def _build_close_action(self, *, pos, reason, target_ratio=0.0, target_position=0.0, risk_decision=None):
        delta_qty = -float(pos)
        return self._with_risk({
            "action": "CLOSE",
            "delta_qty": delta_qty,
            "target_ratio": float(target_ratio),
            "target_position": float(target_position),
            "reason": reason,
            "next_position": 0.0,
            "next_entry_price": 0.0,
            "next_hold_bars": 0,
            "next_cooldown_bars": self._next_trade_cooldown(reason),
            "next_reverse_signal_bars": 0,
            "next_reset_risk": True,
            "raw_target_ratio": 0.0,
            "expected_net_edge": None,
            "take_profit": float(self.take_profit),
            "stop_loss": float(self.stop_loss),
            "signal_prob_gap": 0.0,
            "dominant_prob": 0.0,
        }, risk_decision)

    def _with_risk(self, decision, risk_decision):
        payload = _risk_payload(risk_decision)
        if payload is not None:
            decision = dict(decision)
            decision["risk"] = payload
        return decision

    def apply_decision(self, decision):
        self.position = float(decision["next_position"])
        self.entry_price = float(decision["next_entry_price"])
        self.hold_bars = int(decision["next_hold_bars"])
        if "next_cooldown_bars" in decision:
            self.cooldown_bars_remaining = max(0, int(decision["next_cooldown_bars"]))
        if "next_reverse_signal_bars" in decision:
            self.reverse_signal_bars = max(0, int(decision["next_reverse_signal_bars"]))
        elif self.position == 0:
            self.reverse_signal_bars = 0
        if decision.get("next_reset_risk"):
            self.reset_risk_thresholds()

    def _trend_block_reason(self, direction, trend_bias):
        if (
            self.trend_filter_enabled
            and direction is not None
            and not trend_allows_direction(direction, trend_bias)
        ):
            return f"TrendFilter({trend_bias or 'neutral'})"
        return None

    def _regime_adjustments(self, regime):
        if not self.regime_filter_enabled:
            return 0.0, 1.0
        regime = str(regime or "unknown").lower()
        threshold_bonus = 0.0
        target_multiplier = 1.0
        if regime == "range":
            threshold_bonus = self.regime_range_threshold_bonus
            target_multiplier = self.regime_range_target_multiplier
        elif regime in {"high_vol", "range_high_vol"}:
            threshold_bonus = self.regime_high_vol_threshold_bonus
            target_multiplier = self.regime_high_vol_target_multiplier
        return threshold_bonus, target_multiplier

    def _regime_block_reason(self, direction, regime):
        if not self.regime_filter_enabled or direction is None:
            return None
        regime = str(regime or "unknown").lower()
        allow_range = bool(self.regime_range_allow_trades)
        allow_high_vol = bool(self.regime_high_vol_allow_trades)
        if not self.regime_trend_against_block and regime in {"trend_long", "trend_short"}:
            return None
        if not regime_allows_direction(regime, direction, allow_range=allow_range, allow_high_vol=allow_high_vol):
            return regime_reason(regime)
        return None

    def _min_signal_target_ratio_for_regime(self, regime):
        if not self.regime_filter_enabled:
            return self.min_signal_target_ratio
        regime = str(regime or "unknown").lower()
        if regime == "range" and self.regime_range_min_signal_target_ratio is not None:
            return self.regime_range_min_signal_target_ratio
        if regime in {"high_vol", "range_high_vol"} and self.regime_high_vol_min_signal_target_ratio is not None:
            return self.regime_high_vol_min_signal_target_ratio
        return self.min_signal_target_ratio

    def _dominant_signal_direction(self, long_prob, short_prob, *, min_prob_diff=None, regime=None):
        long_prob = float(long_prob)
        short_prob = float(short_prob)
        prob_gap = abs(long_prob - short_prob)
        min_prob_diff = self.signal_min_prob_diff if min_prob_diff is None else float(min_prob_diff)
        threshold_bonus, _ = self._regime_adjustments(regime)
        threshold_long = self.threshold_long + threshold_bonus
        threshold_short = self.threshold_short + threshold_bonus

        if long_prob >= short_prob:
            if long_prob <= threshold_long or prob_gap < min_prob_diff:
                return None, prob_gap, long_prob
            return "long", prob_gap, long_prob

        if short_prob <= threshold_short or prob_gap < min_prob_diff:
            return None, prob_gap, short_prob
        return "short", prob_gap, short_prob

    def _resolve_directional_target_ratio(
        self,
        *,
        long_prob: float,
        short_prob: float,
        money_flow_ratio: float,
        volatility: float,
        take_profit: float,
        stop_loss: float,
        trend_bias: str = None,
        market_regime: str = None,
        apply_trend_filter: bool = True,
    ):
        long_prob = float(long_prob)
        short_prob = float(short_prob)
        prob_gap = abs(long_prob - short_prob)
        blocked_reason = None
        threshold_bonus, target_multiplier = self._regime_adjustments(market_regime)
        min_signal_target_ratio = self._min_signal_target_ratio_for_regime(market_regime)
        threshold_long = self.threshold_long + threshold_bonus
        threshold_short = self.threshold_short + threshold_bonus

        def passes_cost_gate(dominant_prob):
            expected_net_edge = self._expected_net_edge_ratio(
                dominant_prob,
                take_profit,
                stop_loss,
            )
            if expected_net_edge <= self.min_expected_net_edge:
                return False, expected_net_edge
            return True, expected_net_edge

        if long_prob >= short_prob:
            direction = "long"
            dominant_prob = long_prob
            if dominant_prob <= threshold_long or prob_gap < self.signal_min_prob_diff:
                return 0.0, prob_gap, dominant_prob, "WeakSignal", None, 0.0

            cost_ok, expected_net_edge = passes_cost_gate(dominant_prob)
            if not cost_ok:
                blocked_reason = f"CostGate(edge={expected_net_edge:.4%})"
            if blocked_reason is None:
                blocked_reason = self._regime_block_reason(direction, market_regime)
            if (
                blocked_reason is None
                and apply_trend_filter
            ):
                blocked_reason = self._trend_block_reason(direction, trend_bias)

            target_ratio = float(self.pm.calculate_target_ratio(
                long_prob,
                money_flow_ratio,
                volatility,
                self.reward_risk,
            ))
            target_ratio *= target_multiplier
            if target_ratio < min_signal_target_ratio:
                return 0.0, prob_gap, dominant_prob, "SmallTarget", expected_net_edge, target_ratio
            if blocked_reason is not None:
                return 0.0, prob_gap, dominant_prob, blocked_reason, expected_net_edge, target_ratio
            return target_ratio, prob_gap, dominant_prob, None, expected_net_edge, target_ratio

        direction = "short"
        dominant_prob = short_prob
        if dominant_prob <= threshold_short or prob_gap < self.signal_min_prob_diff:
            return 0.0, prob_gap, dominant_prob, "WeakSignal", None, 0.0

        cost_ok, expected_net_edge = passes_cost_gate(dominant_prob)
        if not cost_ok:
            blocked_reason = f"CostGate(edge={expected_net_edge:.4%})"
        if blocked_reason is None:
            blocked_reason = self._regime_block_reason(direction, market_regime)
        if (
            blocked_reason is None
            and apply_trend_filter
        ):
            blocked_reason = self._trend_block_reason(direction, trend_bias)

        target_ratio = float(self.pm.calculate_target_ratio(
            short_prob,
            money_flow_ratio,
            volatility,
            self.reward_risk,
        ))
        target_ratio *= target_multiplier
        if target_ratio < min_signal_target_ratio:
            return 0.0, prob_gap, dominant_prob, "SmallTarget", expected_net_edge, target_ratio
        if blocked_reason is not None:
            return 0.0, prob_gap, dominant_prob, blocked_reason, expected_net_edge, target_ratio
        return -target_ratio, prob_gap, dominant_prob, None, expected_net_edge, -target_ratio

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
        trend_bias: str = None,
        market_regime: str = None,
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
            market_regime=market_regime,
        )
        cooldown_remaining = int(self.cooldown_bars_remaining)
        reverse_signal_bars = int(self.reverse_signal_bars)
        risk_decision = None

        # ======================
        # 1) 持仓 -> 止盈止损
        # ======================
        if pos != 0:
            pnl_pct = (price - self.entry_price) / self.entry_price if pos > 0 else (self.entry_price - price) / self.entry_price
            if pnl_pct >= take_profit:
                return self._build_close_action(pos=pos, reason="TakeProfit", risk_decision=risk_decision)
            if pnl_pct <= -stop_loss:
                return self._build_close_action(pos=pos, reason="StopLoss", risk_decision=risk_decision)

        if pos == 0 and cooldown_remaining > 0:
            return self._with_risk({
                "action": "HOLD",
                "delta_qty": 0.0,
                "target_ratio": 0.0,
                "target_position": 0.0,
                "reason": f"Cooldown({cooldown_remaining})",
                "next_position": 0.0,
                "next_entry_price": 0.0,
                "next_hold_bars": 0,
                "next_cooldown_bars": self._next_hold_cooldown(),
                "next_reverse_signal_bars": 0,
                "next_reset_risk": True,
                "raw_target_ratio": 0.0,
                "expected_net_edge": None,
                "take_profit": float(take_profit),
                "stop_loss": float(stop_loss),
                "signal_prob_gap": abs(float(long_prob) - float(short_prob)),
                "dominant_prob": max(float(long_prob), float(short_prob)),
            }, risk_decision)

        # ======================
        # 2) 计算目标仓位档位
        # ======================
        target_ratio, signal_prob_gap, dominant_prob, block_reason, expected_net_edge, raw_target_ratio = self._resolve_directional_target_ratio(
            long_prob=long_prob,
            short_prob=short_prob,
            money_flow_ratio=money_flow_ratio,
            volatility=volatility,
            take_profit=take_profit,
            stop_loss=stop_loss,
            trend_bias=trend_bias,
            market_regime=market_regime,
            apply_trend_filter=False,
        )
        raw_target_ratio = float(raw_target_ratio)
        target_position = target_ratio * equity / price

        def attach_signal_diagnostics(decision):
            decision["raw_target_ratio"] = float(raw_target_ratio)
            decision["expected_net_edge"] = None if expected_net_edge is None else float(expected_net_edge)
            decision["take_profit"] = float(take_profit)
            decision["stop_loss"] = float(stop_loss)
            decision["signal_prob_gap"] = float(signal_prob_gap)
            decision["dominant_prob"] = float(dominant_prob)
            return decision
        target_direction = None
        if target_ratio > 0:
            target_direction = "long"
        elif target_ratio < 0:
            target_direction = "short"
        risk_decision = None
        if self.dynamic_risk_controller is not None and target_direction is not None:
            risk_decision = self.dynamic_risk_controller.evaluate(
                long_prob=long_prob,
                short_prob=short_prob,
                volatility=volatility,
                atr_ratio=atr_ratio,
                trend_bias=trend_bias,
                target_direction=target_direction,
            )
            target_ratio = self.dynamic_risk_controller.apply_to_target_ratio(target_ratio, risk_decision)
            target_position = target_ratio * equity / price
        trend_block_reason = self._trend_block_reason(target_direction, trend_bias)
        regime_block_reason = self._regime_block_reason(target_direction, market_regime)
        entry_block_reason = regime_block_reason or trend_block_reason
        raw_signal_direction, raw_signal_prob_gap, raw_dominant_prob = self._dominant_signal_direction(
            long_prob,
            short_prob,
            min_prob_diff=self.reverse_exit_min_prob_diff,
            regime=market_regime,
        )
        same_direction = (pos > 0 and target_position > 0) or (pos < 0 and target_position < 0)
        raw_reverse_signal = (
            (pos > 0 and raw_signal_direction == "short") or
            (pos < 0 and raw_signal_direction == "long")
        )
        next_reverse_signal_bars = reverse_signal_bars + 1 if raw_reverse_signal else 0
        reverse_signal_is_strong = (
            pos != 0 and
            not same_direction and
            abs(raw_target_ratio) > 0 and
            signal_prob_gap >= self.reverse_signal_min_prob_diff and
            abs(raw_target_ratio) >= self.reverse_min_target_ratio
        )
        consecutive_reverse_exit = (
            pos != 0 and
            self.reverse_exit_consecutive_bars > 0 and
            raw_reverse_signal and
            next_reverse_signal_bars >= self.reverse_exit_consecutive_bars
        )

        # ======================
        # 3) 空仓 -> 只允许开仓
        # ======================
        if pos == 0:
            if (
                entry_block_reason is not None
                and target_position != 0
                and abs(target_position * price) >= self.min_adjust_amount
            ):
                return self._with_risk(attach_signal_diagnostics({
                    "action": "HOLD",
                    "delta_qty": 0.0,
                    "target_ratio": 0.0,
                    "target_position": 0.0,
                    "reason": entry_block_reason,
                    "next_position": 0.0,
                    "next_entry_price": 0.0,
                    "next_hold_bars": 0,
                    "next_cooldown_bars": self._next_hold_cooldown(),
                    "next_reverse_signal_bars": 0,
                    "next_reset_risk": True,
                }), risk_decision)
            if abs(target_position * price) >= self.min_adjust_amount and target_position != 0:
                return self._with_risk(attach_signal_diagnostics({
                    "action": "OPEN",
                    "delta_qty": target_position,
                    "target_ratio": target_ratio,
                    "target_position": target_position,
                    "reason": "OpenFromFlat",
                    "next_position": float(target_position),
                    "next_entry_price": float(price),
                    "next_hold_bars": 0,
                    "next_cooldown_bars": self._next_trade_cooldown(),
                    "next_reverse_signal_bars": 0,
                    "next_reset_risk": False,
                }), risk_decision)
            flat_reason = "FlatNoSignal"
            if block_reason is not None:
                flat_reason = f"{block_reason}"
            return self._with_risk(attach_signal_diagnostics({
                "action": "HOLD",
                "delta_qty": 0.0,
                "target_ratio": target_ratio,
                "target_position": target_position,
                "reason": flat_reason,
                "next_position": 0.0,
                "next_entry_price": 0.0,
                "next_hold_bars": 0,
                "next_cooldown_bars": self._next_hold_cooldown(),
                "next_reverse_signal_bars": 0,
                "next_reset_risk": True,
            }), risk_decision)

        # ======================
        # 4) 反向强信号 -> 强制先平仓
        # ======================
        if reverse_signal_is_strong:
            return self._build_close_action(
                pos=pos,
                reason="ReverseClose",
                target_ratio=target_ratio,
                target_position=target_position,
                risk_decision=risk_decision,
            )

        if consecutive_reverse_exit:
            return self._build_close_action(
                pos=pos,
                reason=f"ConsecutiveReverseClose({next_reverse_signal_bars}/{self.reverse_exit_consecutive_bars})",
                target_ratio=target_ratio,
                target_position=target_position,
                risk_decision=risk_decision,
            )

        # ======================
        # 5) 持仓 -> 最小持有期
        # ======================
        next_hold_bars = self.hold_bars + 1
        if next_hold_bars < self.min_hold_bars:
            return self._with_risk(attach_signal_diagnostics({
                "action": "HOLD",
                "delta_qty": 0.0,
                "target_ratio": target_ratio,
                "target_position": target_position,
                "reason": f"MinHold({next_hold_bars}/{self.min_hold_bars})",
                "next_position": float(pos),
                "next_entry_price": float(self.entry_price),
                "next_hold_bars": next_hold_bars,
                "next_cooldown_bars": self._next_hold_cooldown(),
                "next_reverse_signal_bars": next_reverse_signal_bars,
                "next_reset_risk": False,
            }), risk_decision)

        # ======================
        # 6) 同方向 -> 分段加 / 减仓
        # ======================
        if same_direction:
            if cooldown_remaining > 0:
                return self._with_risk(attach_signal_diagnostics({
                    "action": "HOLD",
                    "delta_qty": 0.0,
                    "target_ratio": target_ratio,
                    "target_position": target_position,
                    "reason": f"Cooldown({cooldown_remaining})",
                    "next_position": float(pos),
                    "next_entry_price": float(self.entry_price),
                    "next_hold_bars": next_hold_bars,
                    "next_cooldown_bars": self._next_hold_cooldown(),
                    "next_reverse_signal_bars": next_reverse_signal_bars,
                    "next_reset_risk": False,
                }), risk_decision)

            raw_delta = target_position - pos
            if (
                entry_block_reason is not None
                and np.sign(raw_delta) == np.sign(pos)
                and abs(raw_delta * price) >= self.min_adjust_amount
            ):
                return self._with_risk(attach_signal_diagnostics({
                    "action": "HOLD",
                    "delta_qty": 0.0,
                    "target_ratio": target_ratio,
                    "target_position": target_position,
                    "reason": entry_block_reason,
                    "next_position": float(pos),
                    "next_entry_price": float(self.entry_price),
                    "next_hold_bars": next_hold_bars,
                    "next_cooldown_bars": self._next_hold_cooldown(),
                    "next_reverse_signal_bars": next_reverse_signal_bars,
                    "next_reset_risk": False,
                }), risk_decision)
            diff_ratio = raw_delta / max(abs(pos), 1e-9)

            if abs(diff_ratio) >= self.add_threshold:
                delta = raw_delta
                delta = float(np.clip(
                    delta,
                    -self.max_rebalance_ratio * abs(pos),
                    self.max_rebalance_ratio * abs(pos)
                ))

                if abs(delta * price) >= self.min_adjust_amount:
                    if (
                        self.block_losing_position_adds
                        and np.sign(delta) == np.sign(pos)
                        and self.entry_price > 0
                    ):
                        pnl_pct = (
                            (price - self.entry_price) / self.entry_price
                            if pos > 0 else
                            (self.entry_price - price) / self.entry_price
                        )
                        if pnl_pct < 0:
                            return self._with_risk(attach_signal_diagnostics({
                                "action": "HOLD",
                                "delta_qty": 0.0,
                                "target_ratio": target_ratio,
                                "target_position": target_position,
                                "reason": f"NoAddToLosingPosition(pnl={pnl_pct:.4%})",
                                "next_position": float(pos),
                                "next_entry_price": float(self.entry_price),
                                "next_hold_bars": next_hold_bars,
                                "next_cooldown_bars": self._next_hold_cooldown(),
                                "next_reverse_signal_bars": next_reverse_signal_bars,
                                "next_reset_risk": False,
                            }), risk_decision)

                    new_position = pos + delta

                    # 同方向加仓时，更新持仓均价；减仓则保留剩余仓位原均价。
                    if np.sign(delta) == np.sign(pos):
                        existing_qty = abs(pos)
                        added_qty = abs(delta)
                        total_qty = existing_qty + added_qty
                        if total_qty > 0:
                            next_entry_price = (
                                (existing_qty * self.entry_price) +
                                (added_qty * price)
                            ) / total_qty
                        else:
                            next_entry_price = self.entry_price
                    else:
                        next_entry_price = self.entry_price

                    return self._with_risk(attach_signal_diagnostics({
                        "action": "REBALANCE",
                        "delta_qty": delta,
                        "target_ratio": target_ratio,
                        "target_position": target_position,
                        "reason": "SameDirRebalance",
                        "next_position": float(new_position),
                        "next_entry_price": float(next_entry_price),
                        "next_hold_bars": next_hold_bars,
                        "next_cooldown_bars": self._next_trade_cooldown(),
                        "next_reverse_signal_bars": next_reverse_signal_bars,
                        "next_reset_risk": False,
                    }), risk_decision)

            return self._with_risk(attach_signal_diagnostics({
                "action": "HOLD",
                "delta_qty": 0.0,
                "target_ratio": target_ratio,
                "target_position": target_position,
                "reason": "SameDirNoRebalance",
                "next_position": float(pos),
                "next_entry_price": float(self.entry_price),
                "next_hold_bars": next_hold_bars,
                "next_cooldown_bars": self._next_hold_cooldown(),
                "next_reverse_signal_bars": next_reverse_signal_bars,
                "next_reset_risk": False,
            }), risk_decision)

        # ======================
        # 7) 反向弱信号 -> 持仓等待
        # ======================
        if block_reason is not None:
            return self._with_risk(attach_signal_diagnostics({
                "action": "HOLD",
                "delta_qty": 0.0,
                "target_ratio": target_ratio,
                "target_position": target_position,
                "reason": block_reason,
                "next_position": float(pos),
                "next_entry_price": float(self.entry_price),
                "next_hold_bars": next_hold_bars,
                "next_cooldown_bars": self._next_hold_cooldown(),
                "next_reverse_signal_bars": next_reverse_signal_bars,
                "next_reset_risk": False,
            }), risk_decision)

        if abs(target_position) > 0:
            return self._with_risk(attach_signal_diagnostics({
                "action": "HOLD",
                "delta_qty": 0.0,
                "target_ratio": target_ratio,
                "target_position": target_position,
                "reason": (
                    f"WeakReverseSignal(gap={signal_prob_gap:.3f},"
                    f"dominant={dominant_prob:.3f},ratio={abs(target_ratio):.3f},"
                    f"edge={expected_net_edge:.4f},"
                    f"reverse_bars={next_reverse_signal_bars}/{self.reverse_exit_consecutive_bars})"
                ),
                "next_position": float(pos),
                "next_entry_price": float(self.entry_price),
                "next_hold_bars": next_hold_bars,
                "next_cooldown_bars": self._next_hold_cooldown(),
                "next_reverse_signal_bars": next_reverse_signal_bars,
                "next_reset_risk": False,
            }), risk_decision)

        return self._with_risk(attach_signal_diagnostics({
            "action": "HOLD",
            "delta_qty": 0.0,
            "target_ratio": target_ratio,
            "target_position": target_position,
            "reason": "NoSignalKeep",
            "next_position": float(pos),
            "next_entry_price": float(self.entry_price),
            "next_hold_bars": next_hold_bars,
            "next_cooldown_bars": self._next_hold_cooldown(),
            "next_reverse_signal_bars": next_reverse_signal_bars,
            "next_reset_risk": False,
        }), risk_decision)
