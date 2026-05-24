import unittest
from unittest.mock import patch

import pandas as pd

from backtest.backtest import Backtester, mark_to_market_equity


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


class BacktestSizingTests(unittest.TestCase):
    @patch("backtest.backtest.config.BACKTEST_MIN_ADJUST_AMOUNT", 40.0)
    def test_backtester_uses_backtest_min_adjust_amount(self):
        data = pd.DataFrame(
            {"5m_close": [100.0]},
            index=pd.date_range("2026-01-01", periods=1, freq="5min"),
        )
        with patch.object(Backtester, "_load_data", return_value=({}, 2.8)):
            with patch.object(Backtester, "_load_funding_history", return_value=pd.DataFrame()):
                backtester = Backtester(
                    "multi_period",
                    10,
                    data_dict={},
                    reward_risk=2.8,
                    precomputed_data=data,
                    feature_cols=[],
                    models={},
                    model_weights={},
                    enable_csv_dump=False,
                    show_progress=False,
                    emit_diagnostics=False,
                )

        self.assertEqual(backtester.core.min_adjust_amount, 40.0)


if __name__ == "__main__":
    unittest.main()
