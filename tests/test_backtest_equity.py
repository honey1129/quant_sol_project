import unittest

from backtest.backtest import mark_to_market_equity


class MarkToMarketEquityTests(unittest.TestCase):
    def test_flat_position_returns_cash_balance(self):
        self.assertAlmostEqual(
            mark_to_market_equity(1000.0, 0.0, 0.0, 88.0),
            1000.0,
        )

    def test_long_position_adds_unrealized_profit(self):
        self.assertAlmostEqual(
            mark_to_market_equity(1000.0, 2.0, 80.0, 85.0),
            1010.0,
        )

    def test_short_position_adds_unrealized_profit_when_price_falls(self):
        self.assertAlmostEqual(
            mark_to_market_equity(1000.0, -2.0, 80.0, 75.0),
            1010.0,
        )


if __name__ == "__main__":
    unittest.main()
