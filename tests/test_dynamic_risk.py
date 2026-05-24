import unittest

from core.dynamic_risk import DynamicRiskController
from core.position_manager import PositionManager
from core.strategy_core import StrategyCore


class DynamicRiskControllerTests(unittest.TestCase):
    def test_disabled_keeps_base_risk(self):
        controller = DynamicRiskController(
            enabled=False,
            base_leverage=3,
            min_leverage=1,
            max_leverage=5,
            base_position_ratio=0.45,
            max_position_ratio=0.3,
            target_vol=0.02,
        )

        decision = controller.evaluate(
            long_prob=0.75,
            short_prob=0.25,
            volatility=0.01,
            atr_ratio=0.01,
            trend_bias="long",
            target_direction="long",
        )

        self.assertFalse(decision.enabled)
        self.assertEqual(decision.effective_leverage, 3)
        self.assertAlmostEqual(decision.max_position_ratio, 0.45)
        self.assertAlmostEqual(decision.risk_multiplier, 1.0)
        self.assertAlmostEqual(controller.apply_to_target_ratio(0.4, decision), 0.4)

    def test_high_volatility_and_trend_mismatch_reduce_risk(self):
        controller = DynamicRiskController(
            enabled=True,
            base_leverage=4,
            min_leverage=1,
            max_leverage=4,
            base_position_ratio=0.5,
            min_position_ratio=0.05,
            max_position_ratio=0.6,
            target_vol=0.01,
            high_vol_multiplier=0.5,
            low_signal_multiplier=0.5,
            trend_mismatch_multiplier=0.5,
            weak_signal_threshold=0.2,
            strong_signal_threshold=0.4,
        )

        decision = controller.evaluate(
            long_prob=0.58,
            short_prob=0.42,
            volatility=0.03,
            atr_ratio=0.02,
            trend_bias="short",
            target_direction="long",
        )

        self.assertTrue(decision.enabled)
        self.assertFalse(decision.trend_aligned)
        self.assertLess(decision.risk_multiplier, 0.5)
        self.assertEqual(decision.effective_leverage, 1)
        self.assertLess(controller.apply_to_target_ratio(0.5, decision), 0.25)

    def test_strong_aligned_signal_cannot_exceed_base_leverage(self):
        controller = DynamicRiskController(
            enabled=True,
            base_leverage=3,
            min_leverage=1,
            max_leverage=10,
            base_position_ratio=0.4,
            max_position_ratio=0.5,
            target_vol=0.02,
            strong_signal_threshold=0.25,
        )

        decision = controller.evaluate(
            long_prob=0.82,
            short_prob=0.18,
            volatility=0.006,
            atr_ratio=0.006,
            trend_bias="long",
            target_direction="long",
        )

        self.assertEqual(decision.effective_leverage, 3)
        self.assertLessEqual(decision.max_position_ratio, 0.5)

    def test_leverage_bounds_cannot_exceed_base_when_min_is_misconfigured(self):
        controller = DynamicRiskController(
            enabled=True,
            base_leverage=2,
            min_leverage=5,
            max_leverage=10,
            target_vol=0.02,
            strong_signal_threshold=0.25,
        )

        decision = controller.evaluate(
            long_prob=0.82,
            short_prob=0.18,
            volatility=0.006,
            atr_ratio=0.006,
            trend_bias="long",
            target_direction="long",
        )

        self.assertEqual(decision.effective_leverage, 2)


class DynamicRiskStrategyCoreTests(unittest.TestCase):
    def test_strategy_core_scales_open_target_ratio(self):
        controller = DynamicRiskController(
            enabled=True,
            base_leverage=3,
            min_leverage=1,
            max_leverage=3,
            base_position_ratio=0.45,
            min_position_ratio=0.05,
            max_position_ratio=0.45,
            target_vol=0.01,
            high_vol_multiplier=0.5,
            trend_mismatch_multiplier=0.5,
            strong_signal_threshold=0.4,
        )
        pm = PositionManager(min_ratio=0.45, max_ratio=0.45)
        core = StrategyCore(
            pm,
            threshold_long=0.55,
            threshold_short=0.55,
            take_profit=0.5,
            stop_loss=0.5,
            adaptive_tp_sl_enabled=False,
            min_hold_bars=0,
            min_adjust_amount=0.0,
            signal_min_prob_diff=0.05,
            min_signal_target_ratio=0.01,
            reward_risk=3.0,
            dynamic_risk_controller=controller,
        )

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.75,
            short_prob=0.25,
            money_flow_ratio=1.0,
            volatility=0.03,
            atr_ratio=0.02,
            trend_bias="short",
        )

        self.assertEqual(out["action"], "OPEN")
        self.assertIn("risk", out)
        self.assertLess(out["target_ratio"], 0.45)
        self.assertLess(out["target_position"], 4.5)

    def test_dynamic_risk_does_not_hide_strong_reverse_exit(self):
        controller = DynamicRiskController(
            enabled=True,
            base_leverage=3,
            min_leverage=1,
            max_leverage=3,
            base_position_ratio=0.45,
            min_position_ratio=0.05,
            max_position_ratio=0.45,
            target_vol=0.01,
            high_vol_multiplier=0.5,
            trend_mismatch_multiplier=0.5,
            strong_signal_threshold=0.4,
        )
        pm = PositionManager(min_ratio=0.45, max_ratio=0.45)
        core = StrategyCore(
            pm,
            threshold_long=0.55,
            threshold_short=0.55,
            take_profit=0.5,
            stop_loss=0.5,
            adaptive_tp_sl_enabled=False,
            min_hold_bars=10,
            min_adjust_amount=0.0,
            signal_min_prob_diff=0.05,
            min_signal_target_ratio=0.01,
            reverse_signal_min_prob_diff=0.18,
            reverse_min_target_ratio=0.12,
            reward_risk=3.0,
            dynamic_risk_controller=controller,
        )
        core.set_state(position=2.0, entry_price=100.0, hold_bars=0)

        out = core.on_bar(
            price=100.0,
            equity=1000.0,
            long_prob=0.25,
            short_prob=0.75,
            money_flow_ratio=1.0,
            volatility=0.03,
            atr_ratio=0.02,
            trend_bias="long",
        )

        self.assertEqual(out["action"], "CLOSE")
        self.assertEqual(out["reason"], "ReverseClose")
        self.assertIn("risk", out)
        self.assertLess(abs(out["target_ratio"]), 0.12)


if __name__ == "__main__":
    unittest.main()
