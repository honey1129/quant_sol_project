import unittest
import sys
import types

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    fake_numpy = types.ModuleType("numpy")

    def clip(value, lower, upper):
        return max(lower, min(value, upper))

    def sign(value):
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    fake_numpy.clip = clip
    fake_numpy.sign = sign
    sys.modules["numpy"] = fake_numpy

from core.strategy_core import StrategyCore


class StubPositionManager:
    def __init__(self, target_ratio):
        self.target_ratio = target_ratio

    def calculate_target_ratio(self, prob, money_flow_ratio, volatility, reward_risk):
        return self.target_ratio


class StrategyCoreRebalanceTests(unittest.TestCase):
    def build_core(self, target_ratio, **kwargs):
        return StrategyCore(
            StubPositionManager(target_ratio),
            threshold_long=0.55,
            threshold_short=0.55,
            take_profit=kwargs.pop("take_profit", 0.5),
            stop_loss=kwargs.pop("stop_loss", 0.5),
            adaptive_tp_sl_enabled=kwargs.pop("adaptive_tp_sl_enabled", False),
            min_hold_bars=kwargs.pop("min_hold_bars", 0),
            add_threshold=kwargs.pop("add_threshold", 0.0),
            max_rebalance_ratio=kwargs.pop("max_rebalance_ratio", 1.0),
            min_adjust_amount=kwargs.pop("min_adjust_amount", 0.0),
            reward_risk=1.0,
            **kwargs,
        )

    def test_short_rebalance_reducing_position_returns_positive_delta(self):
        core = self.build_core(target_ratio=0.5)
        core.set_state(position=-10.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.1,
            short_prob=0.9,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "REBALANCE")
        self.assertAlmostEqual(out["delta_qty"], 5.0)

    def test_short_rebalance_adding_position_returns_negative_delta(self):
        core = self.build_core(target_ratio=1.5, block_losing_position_adds=False)
        core.set_state(position=-10.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.1,
            short_prob=0.9,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "REBALANCE")
        self.assertAlmostEqual(out["delta_qty"], -5.0)

    def test_same_direction_add_to_losing_long_is_blocked(self):
        core = self.build_core(target_ratio=1.5, block_losing_position_adds=True)
        core.set_state(position=10.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=99.0,
            equity=1000.0,
            long_prob=0.9,
            short_prob=0.1,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "HOLD")
        self.assertTrue(out["reason"].startswith("NoAddToLosingPosition"))

    def test_same_direction_reduction_of_losing_long_is_allowed(self):
        core = self.build_core(target_ratio=0.5, block_losing_position_adds=True)
        core.set_state(position=10.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=99.0,
            equity=1000.0,
            long_prob=0.9,
            short_prob=0.1,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "REBALANCE")
        self.assertLess(out["delta_qty"], 0)

    def test_same_direction_add_to_winning_short_is_allowed(self):
        core = self.build_core(target_ratio=1.5, block_losing_position_adds=True)
        core.set_state(position=-10.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=99.0,
            equity=1000.0,
            long_prob=0.1,
            short_prob=0.9,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "REBALANCE")
        self.assertLess(out["delta_qty"], 0)

    def test_adaptive_thresholds_override_fixed_tp_sl(self):
        core = self.build_core(
            target_ratio=0.0,
            adaptive_tp_sl_enabled=True,
            atr_take_profit_multiplier=2.5,
            atr_stop_loss_multiplier=1.2,
            volatility_take_profit_multiplier=4.0,
            volatility_stop_loss_multiplier=2.0,
            adaptive_take_profit_min=0.006,
            adaptive_take_profit_max=0.03,
            adaptive_stop_loss_min=0.004,
            adaptive_stop_loss_max=0.02,
        )
        core.set_state(position=1.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=101.1,
            equity=1000.0,
            long_prob=0.1,
            short_prob=0.1,
            money_flow_ratio=1.0,
            volatility=0.002,
            atr_ratio=0.004,
        )

        self.assertEqual(out["action"], "CLOSE")
        self.assertEqual(out["reason"], "TakeProfit")
        core.apply_decision(out)
        take_profit, stop_loss = core.get_risk_thresholds()
        self.assertAlmostEqual(take_profit, 0.5)
        self.assertAlmostEqual(stop_loss, 0.5)

    def test_resolve_risk_thresholds_clamps_adaptive_values(self):
        core = self.build_core(
            target_ratio=0.0,
            adaptive_tp_sl_enabled=True,
            atr_take_profit_multiplier=10.0,
            atr_stop_loss_multiplier=8.0,
            volatility_take_profit_multiplier=10.0,
            volatility_stop_loss_multiplier=8.0,
            adaptive_take_profit_min=0.006,
            adaptive_take_profit_max=0.03,
            adaptive_stop_loss_min=0.004,
            adaptive_stop_loss_max=0.02,
        )

        take_profit, stop_loss = core.resolve_risk_thresholds(
            volatility=0.01,
            atr_ratio=0.01,
        )

        self.assertAlmostEqual(take_profit, 0.03)
        self.assertAlmostEqual(stop_loss, 0.02)


    def test_range_high_vol_uses_regime_stop_loss_floor(self):
        core = self.build_core(
            target_ratio=0.0,
            adaptive_tp_sl_enabled=True,
            adaptive_stop_loss_min=0.0065,
            adaptive_stop_loss_max=0.022,
            regime_high_vol_stop_loss_min=0.009,
            atr_stop_loss_multiplier=2.4,
            volatility_stop_loss_multiplier=2.8,
            atr_take_profit_multiplier=6.0,
            volatility_take_profit_multiplier=8.0,
            min_take_profit_to_stop_loss_ratio=2.2,
        )

        _, normal_stop_loss = core.resolve_risk_thresholds(
            volatility=0.0008,
            atr_ratio=0.0014,
            market_regime="trend_long",
        )
        take_profit, high_vol_stop_loss = core.resolve_risk_thresholds(
            volatility=0.0008,
            atr_ratio=0.0014,
            market_regime="range_high_vol",
        )

        self.assertAlmostEqual(normal_stop_loss, 0.0065)
        self.assertAlmostEqual(high_vol_stop_loss, 0.009)
        self.assertGreaterEqual(take_profit, high_vol_stop_loss * 2.2)

    def test_range_high_vol_stop_loss_floor_delays_noise_stopout(self):
        core = self.build_core(
            target_ratio=0.0,
            adaptive_tp_sl_enabled=True,
            adaptive_stop_loss_min=0.0065,
            adaptive_stop_loss_max=0.022,
            regime_high_vol_stop_loss_min=0.009,
            atr_stop_loss_multiplier=2.4,
            volatility_stop_loss_multiplier=2.8,
            min_hold_bars=0,
        )
        core.set_state(position=1.0, entry_price=100.0, hold_bars=10)

        out = core.on_bar(
            price=99.2,
            equity=1000.0,
            long_prob=0.75,
            short_prob=0.25,
            money_flow_ratio=1.0,
            volatility=0.0008,
            atr_ratio=0.0014,
            market_regime="range_high_vol",
        )

        self.assertEqual(out["action"], "HOLD")
        self.assertNotEqual(out["reason"], "StopLoss")

    def test_flat_position_stays_flat_on_weak_signal_gap(self):
        core = self.build_core(target_ratio=0.2, signal_min_prob_diff=0.12)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.54,
            short_prob=0.46,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "HOLD")
        self.assertEqual(out["reason"], "WeakSignal")

    def test_existing_position_ignores_weak_reverse_signal(self):
        core = self.build_core(
            target_ratio=0.09,
            signal_min_prob_diff=0.12,
            reverse_signal_min_prob_diff=0.18,
            reverse_min_target_ratio=0.1,
        )
        core.set_state(position=10.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.40,
            short_prob=0.60,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "HOLD")
        self.assertTrue(out["reason"].startswith("WeakReverseSignal"))

    def test_consecutive_reverse_signal_closes_even_when_target_ratio_is_small(self):
        core = self.build_core(
            target_ratio=0.04,
            signal_min_prob_diff=0.12,
            reverse_signal_min_prob_diff=0.18,
            reverse_min_target_ratio=0.1,
            reverse_exit_consecutive_bars=2,
            reverse_exit_min_prob_diff=0.18,
            min_hold_bars=10,
        )
        core.set_state(position=10.0, entry_price=100.0, hold_bars=0)

        first = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.35,
            short_prob=0.65,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(first["action"], "HOLD")
        self.assertEqual(first["next_reverse_signal_bars"], 1)
        core.apply_decision(first)

        second = core.on_bar(
            price=99.8,
            equity=1000.0,
            long_prob=0.34,
            short_prob=0.66,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(second["action"], "CLOSE")
        self.assertEqual(second["reason"], "ConsecutiveReverseClose(2/2)")

    def test_consecutive_reverse_signal_resets_on_aligned_signal(self):
        core = self.build_core(
            target_ratio=0.04,
            signal_min_prob_diff=0.12,
            reverse_signal_min_prob_diff=0.18,
            reverse_min_target_ratio=0.1,
            reverse_exit_consecutive_bars=2,
            reverse_exit_min_prob_diff=0.18,
        )
        core.set_state(position=10.0, entry_price=100.0, hold_bars=0)

        first = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.35,
            short_prob=0.65,
            money_flow_ratio=1.0,
            volatility=0.01,
        )
        self.assertEqual(first["next_reverse_signal_bars"], 1)
        core.apply_decision(first)

        aligned = core.on_bar(
            price=100.1,
            equity=1000.0,
            long_prob=0.70,
            short_prob=0.30,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(aligned["action"], "HOLD")
        self.assertEqual(aligned["next_reverse_signal_bars"], 0)

    def test_existing_position_closes_on_strong_reverse_signal(self):
        core = self.build_core(
            target_ratio=0.2,
            signal_min_prob_diff=0.12,
            reverse_signal_min_prob_diff=0.18,
            reverse_min_target_ratio=0.1,
        )
        core.set_state(position=10.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.35,
            short_prob=0.65,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "CLOSE")
        self.assertEqual(out["reason"], "ReverseClose")

    def test_strong_reverse_signal_bypasses_trend_filter(self):
        core = self.build_core(
            target_ratio=0.2,
            signal_min_prob_diff=0.12,
            reverse_signal_min_prob_diff=0.18,
            reverse_min_target_ratio=0.1,
            trend_filter_enabled=True,
        )
        core.set_state(position=10.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.35,
            short_prob=0.65,
            money_flow_ratio=1.0,
            volatility=0.01,
            trend_bias="long",
        )

        self.assertEqual(out["action"], "CLOSE")
        self.assertEqual(out["reason"], "ReverseClose")

    def test_strong_reverse_signal_bypasses_min_hold(self):
        core = self.build_core(
            target_ratio=0.2,
            signal_min_prob_diff=0.12,
            reverse_signal_min_prob_diff=0.18,
            reverse_min_target_ratio=0.1,
            min_hold_bars=10,
        )
        core.set_state(position=10.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.35,
            short_prob=0.65,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "CLOSE")
        self.assertEqual(out["reason"], "ReverseClose")

    def test_open_does_not_mutate_core_state(self):
        core = self.build_core(target_ratio=0.5)
        core.set_state(position=0.0, entry_price=0.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.9,
            short_prob=0.1,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "OPEN")
        self.assertNotEqual(out["next_position"], 0.0)
        position, entry_price, hold_bars = core.get_state()
        self.assertEqual(position, 0.0)
        self.assertEqual(entry_price, 0.0)
        self.assertEqual(hold_bars, 0)

    def test_close_does_not_mutate_core_state(self):
        core = self.build_core(
            target_ratio=0.2,
            signal_min_prob_diff=0.12,
            reverse_signal_min_prob_diff=0.18,
            reverse_min_target_ratio=0.1,
        )
        core.set_state(position=10.0, entry_price=100.0, hold_bars=5)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.35,
            short_prob=0.65,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "CLOSE")
        self.assertEqual(out["next_position"], 0.0)
        position, entry_price, hold_bars = core.get_state()
        self.assertEqual(position, 10.0)
        self.assertEqual(entry_price, 100.0)
        self.assertEqual(hold_bars, 5)

    def test_apply_decision_writes_state(self):
        core = self.build_core(
            target_ratio=0.2,
            signal_min_prob_diff=0.12,
            reverse_signal_min_prob_diff=0.18,
            reverse_min_target_ratio=0.1,
        )
        core.set_state(position=10.0, entry_price=100.0, hold_bars=5)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.35,
            short_prob=0.65,
            money_flow_ratio=1.0,
            volatility=0.01,
        )
        self.assertEqual(out["action"], "CLOSE")

        core.apply_decision(out)
        position, entry_price, hold_bars = core.get_state()
        self.assertEqual(position, 0.0)
        self.assertEqual(entry_price, 0.0)
        self.assertEqual(hold_bars, 0)

    def test_cost_gate_blocks_low_net_edge_open(self):
        core = self.build_core(
            target_ratio=0.3,
            take_profit=0.01,
            stop_loss=0.01,
            fee_rate=0.002,
            cost_buffer_multiplier=2.0,
        )
        core.set_state(position=0.0, entry_price=0.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.65,
            short_prob=0.35,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "HOLD")
        self.assertTrue(out["reason"].startswith("CostGate"))
        self.assertAlmostEqual(out["required_trade_prob"], 0.9)
        self.assertAlmostEqual(out["prob_edge_margin"], -0.25)
        self.assertAlmostEqual(out["round_trip_cost"], 0.004)
        self.assertAlmostEqual(out["cost_floor"], 0.008)

    def test_required_probability_for_edge_includes_cost_buffer(self):
        core = self.build_core(
            target_ratio=0.3,
            take_profit=0.03,
            stop_loss=0.01,
            fee_rate=0.0005,
            slippage_bps=3.0,
            cost_buffer_multiplier=2.0,
            min_expected_net_edge=0.001,
        )

        required = core.required_probability_for_edge(0.03, 0.01)

        self.assertAlmostEqual(required, (0.01 + 0.0032 + 0.001) / 0.04)

    def test_flat_cooldown_blocks_new_open_and_counts_down(self):
        core = self.build_core(target_ratio=0.5, trade_cooldown_bars=3)
        core.set_state(
            position=0.0,
            entry_price=0.0,
            hold_bars=0,
            cooldown_bars_remaining=2,
        )

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.9,
            short_prob=0.1,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "HOLD")
        self.assertEqual(out["reason"], "Cooldown(2)")
        self.assertEqual(out["next_cooldown_bars"], 1)

    def test_same_direction_cooldown_blocks_rebalance(self):
        core = self.build_core(target_ratio=0.5, trade_cooldown_bars=3)
        core.set_state(
            position=1.0,
            entry_price=100.0,
            hold_bars=0,
            cooldown_bars_remaining=2,
        )

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.9,
            short_prob=0.1,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "HOLD")
        self.assertEqual(out["reason"], "Cooldown(2)")


    def test_stop_loss_uses_dedicated_cooldown(self):
        core = self.build_core(
            target_ratio=0.0,
            take_profit=0.5,
            stop_loss=0.01,
            trade_cooldown_bars=3,
            stop_loss_cooldown_bars=9,
        )
        core.set_state(position=1.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=98.9,
            equity=1000.0,
            long_prob=0.1,
            short_prob=0.1,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "CLOSE")
        self.assertEqual(out["reason"], "StopLoss")
        self.assertEqual(out["next_cooldown_bars"], 9)

    def test_apply_open_decision_sets_trade_cooldown(self):
        core = self.build_core(target_ratio=0.5, trade_cooldown_bars=3)
        core.set_state(position=0.0, entry_price=0.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.9,
            short_prob=0.1,
            money_flow_ratio=1.0,
            volatility=0.01,
        )

        self.assertEqual(out["action"], "OPEN")
        core.apply_decision(out)
        self.assertEqual(core.get_cooldown_bars_remaining(), 3)

    def test_trend_filter_blocks_countertrend_open(self):
        core = self.build_core(target_ratio=0.3, trend_filter_enabled=True)
        core.set_state(position=0.0, entry_price=0.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.8,
            short_prob=0.2,
            money_flow_ratio=1.0,
            volatility=0.01,
            trend_bias="short",
        )

        self.assertEqual(out["action"], "HOLD")
        self.assertEqual(out["reason"], "TrendFilter(short)")

    def test_trend_filter_blocks_countertrend_add(self):
        core = self.build_core(target_ratio=1.5, trend_filter_enabled=True)
        core.set_state(position=10.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.8,
            short_prob=0.2,
            money_flow_ratio=1.0,
            volatility=0.01,
            trend_bias="short",
        )

        self.assertEqual(out["action"], "HOLD")
        self.assertEqual(out["reason"], "TrendFilter(short)")

    def test_trend_filter_allows_countertrend_position_reduction(self):
        core = self.build_core(target_ratio=0.5, trend_filter_enabled=True)
        core.set_state(position=10.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.8,
            short_prob=0.2,
            money_flow_ratio=1.0,
            volatility=0.01,
            trend_bias="short",
        )

        self.assertEqual(out["action"], "REBALANCE")
        self.assertAlmostEqual(out["delta_qty"], -5.0)

    def test_trend_filter_allows_aligned_open(self):
        core = self.build_core(target_ratio=0.3, trend_filter_enabled=True)
        core.set_state(position=0.0, entry_price=0.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.8,
            short_prob=0.2,
            money_flow_ratio=1.0,
            volatility=0.01,
            trend_bias="long",
        )

        self.assertEqual(out["action"], "OPEN")
        self.assertGreater(out["delta_qty"], 0)


if __name__ == "__main__":
    unittest.main()

class StrategyCoreRegimeTests(unittest.TestCase):
    def build_core(self, target_ratio=0.2, **kwargs):
        defaults = dict(
            threshold_long=0.60,
            threshold_short=0.60,
            take_profit=0.5,
            stop_loss=0.5,
            adaptive_tp_sl_enabled=False,
            min_hold_bars=0,
            add_threshold=0.0,
            max_rebalance_ratio=1.0,
            min_adjust_amount=0.0,
            signal_min_prob_diff=0.10,
            min_signal_target_ratio=0.05,
            reward_risk=1.0,
            regime_filter_enabled=True,
            regime_range_allow_trades=True,
            regime_high_vol_allow_trades=False,
            regime_range_threshold_bonus=0.04,
            regime_high_vol_threshold_bonus=0.06,
            regime_range_target_multiplier=0.5,
            regime_high_vol_target_multiplier=0.25,
        )
        defaults.update(kwargs)
        return StrategyCore(
            StubPositionManager(target_ratio),
            **defaults,
        )

    def test_regime_blocks_against_trend_direction(self):
        core = self.build_core(target_ratio=0.2)
        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.20,
            short_prob=0.90,
            money_flow_ratio=1.0,
            volatility=0.001,
            market_regime="trend_long",
        )
        self.assertEqual(out["action"], "HOLD")
        self.assertEqual(out["reason"], "RegimeFilter(trend_long)")

    def test_range_regime_requires_stronger_signal(self):
        core = self.build_core(target_ratio=0.2)
        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.62,
            short_prob=0.38,
            money_flow_ratio=1.0,
            volatility=0.001,
            market_regime="range",
        )
        self.assertEqual(out["action"], "HOLD")
        self.assertEqual(out["reason"], "WeakSignal")

    def test_range_regime_scales_target_ratio(self):
        core = self.build_core(target_ratio=0.4)
        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.80,
            short_prob=0.20,
            money_flow_ratio=1.0,
            volatility=0.001,
            market_regime="range",
        )
        self.assertEqual(out["action"], "OPEN")
        self.assertAlmostEqual(out["target_ratio"], 0.2)


    def test_range_regime_can_use_lower_min_target_ratio(self):
        core = self.build_core(
            target_ratio=0.10,
            regime_range_target_multiplier=0.5,
            min_signal_target_ratio=0.08,
            regime_range_min_signal_target_ratio=0.05,
        )
        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.80,
            short_prob=0.20,
            money_flow_ratio=1.0,
            volatility=0.001,
            market_regime="range",
        )
        self.assertEqual(out["action"], "OPEN")
        self.assertAlmostEqual(out["target_ratio"], 0.05)

    def test_high_vol_regime_can_use_lower_min_target_ratio_when_allowed(self):
        core = self.build_core(
            target_ratio=0.16,
            regime_high_vol_allow_trades=True,
            regime_high_vol_target_multiplier=0.35,
            min_signal_target_ratio=0.08,
            regime_high_vol_min_signal_target_ratio=0.05,
        )
        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.90,
            short_prob=0.10,
            money_flow_ratio=1.0,
            volatility=0.003,
            market_regime="range_high_vol",
        )
        self.assertEqual(out["action"], "OPEN")
        self.assertAlmostEqual(out["target_ratio"], 0.056)

    def test_high_vol_regime_blocks_new_trades_by_default(self):
        core = self.build_core(target_ratio=0.4)
        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.90,
            short_prob=0.10,
            money_flow_ratio=1.0,
            volatility=0.003,
            market_regime="range_high_vol",
        )
        self.assertEqual(out["action"], "HOLD")
        self.assertEqual(out["reason"], "RegimeFilter(range_high_vol)")
