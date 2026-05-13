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
        self.assertEqual(out["reason"], "TP/SL")
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
        self.assertEqual(out["reason"], "FlatNoSignal")

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
