import unittest

from backtest.backtest import resolve_intrabar_tp_sl


class ResolveIntrabarTpSlTests(unittest.TestCase):
    def test_long_prefers_stop_on_same_bar_conflict(self):
        hit = resolve_intrabar_tp_sl(
            position=1.0,
            entry_price=100.0,
            bar_open=100.0,
            bar_high=103.0,
            bar_low=98.0,
            take_profit=0.02,
            stop_loss=0.01,
            worst_case=True,
        )

        self.assertIsNotNone(hit)
        self.assertEqual(hit["reason"], "SL")
        self.assertAlmostEqual(hit["trigger_price"], 99.0)

    def test_short_take_profit_gap_uses_open_price(self):
        hit = resolve_intrabar_tp_sl(
            position=-1.0,
            entry_price=100.0,
            bar_open=97.5,
            bar_high=98.5,
            bar_low=97.0,
            take_profit=0.02,
            stop_loss=0.01,
            worst_case=True,
        )

        self.assertIsNotNone(hit)
        self.assertEqual(hit["reason"], "TP")
        self.assertAlmostEqual(hit["trigger_price"], 97.5)

    def test_returns_none_when_thresholds_not_hit(self):
        hit = resolve_intrabar_tp_sl(
            position=1.0,
            entry_price=100.0,
            bar_open=100.2,
            bar_high=100.7,
            bar_low=99.4,
            take_profit=0.02,
            stop_loss=0.01,
            worst_case=True,
        )

        self.assertIsNone(hit)


if __name__ == "__main__":
    unittest.main()
