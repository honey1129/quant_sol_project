import unittest

from core.regime_filter import derive_market_regime, regime_allows_direction


class RegimeFilterTests(unittest.TestCase):
    def test_derives_trend_regime_when_gap_is_large(self):
        regime = derive_market_regime(
            trend_bias="long",
            trend_gap=0.004,
            volatility=0.0008,
            atr_ratio=0.001,
            money_flow_ratio=1.0,
            trend_gap_threshold=0.003,
        )
        self.assertEqual(regime["regime"], "trend_long")
        self.assertTrue(regime["is_trending"])

    def test_derives_range_high_vol_without_trend(self):
        regime = derive_market_regime(
            trend_bias="neutral",
            trend_gap=0.001,
            volatility=0.002,
            atr_ratio=0.0018,
            money_flow_ratio=1.0,
        )
        self.assertEqual(regime["regime"], "range_high_vol")
        self.assertTrue(regime["is_high_vol"])

    def test_regime_allows_only_aligned_trend_direction(self):
        self.assertTrue(regime_allows_direction("trend_short", "short"))
        self.assertFalse(regime_allows_direction("trend_short", "long"))


if __name__ == "__main__":
    unittest.main()
