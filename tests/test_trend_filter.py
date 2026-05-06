import unittest

from core.trend_filter import derive_trend_context, trend_allows_direction


class TrendFilterTests(unittest.TestCase):
    def test_derives_long_bias_from_ema_stack_and_price(self):
        context = derive_trend_context(
            {
                "5m_close": 101.0,
                "1H_ema_20": 100.0,
                "1H_ema_60": 99.0,
            },
            min_gap=0.001,
        )

        self.assertEqual(context["trend_bias"], "long")
        self.assertGreater(context["trend_gap"], 0)

    def test_derives_short_bias_from_ema_stack_and_price(self):
        context = derive_trend_context(
            {
                "5m_close": 99.0,
                "1H_ema_20": 100.0,
                "1H_ema_60": 101.0,
            },
            min_gap=0.001,
        )

        self.assertEqual(context["trend_bias"], "short")
        self.assertLess(context["trend_gap"], 0)

    def test_returns_neutral_on_pullback_against_fast_ema(self):
        context = derive_trend_context(
            {
                "5m_close": 98.0,
                "1H_ema_20": 100.0,
                "1H_ema_60": 99.0,
            },
            min_gap=0.001,
        )

        self.assertEqual(context["trend_bias"], "neutral")

    def test_direction_gate_blocks_only_opposite_trend(self):
        self.assertTrue(trend_allows_direction("long", "long"))
        self.assertTrue(trend_allows_direction("short", "neutral"))
        self.assertFalse(trend_allows_direction("short", "long"))


if __name__ == "__main__":
    unittest.main()
