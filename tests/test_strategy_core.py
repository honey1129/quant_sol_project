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
            take_profit=0.5,
            stop_loss=0.5,
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
        core = self.build_core(target_ratio=1.5)
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


if __name__ == "__main__":
    unittest.main()
