import unittest

import pandas as pd

from core.ml_feature_engineering import keep_confirmed_bars, merge_multi_period_features


def build_ohlcv(index, base_price, step):
    values = []
    for i, ts in enumerate(index):
        close = base_price + step * i
        values.append(
            {
                "timestamp": ts,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000 + i,
            }
        )

    df = pd.DataFrame(values)
    df.set_index("timestamp", inplace=True)
    return df


class FeatureAlignmentTests(unittest.TestCase):
    def test_keep_confirmed_bars_drops_unconfirmed_tail(self):
        index = pd.date_range("2026-04-24 00:00:00", periods=3, freq="5min")
        df = build_ohlcv(index, base_price=100.0, step=1.0)
        df["confirm"] = ["1", "1", "0"]

        confirmed = keep_confirmed_bars(
            df,
            "5m",
            now_ts=pd.Timestamp("2026-04-24 01:00:00", tz="UTC"),
        )

        self.assertEqual(list(confirmed.index), list(index[:2]))

    def test_merge_multi_period_features_shifts_higher_timeframe_by_one_bar(self):
        base_index = pd.date_range("2026-04-24 00:00:00", periods=72, freq="5min")
        high_index = pd.date_range("2026-04-24 00:00:00", periods=24, freq="15min")

        base_df = build_ohlcv(base_index, base_price=100.0, step=1.0)
        base_df["confirm"] = "1"

        high_df = build_ohlcv(high_index, base_price=1000.0, step=100.0)
        high_df["confirm"] = "1"

        merged = merge_multi_period_features(
            {
                "5m": base_df,
                "15m": high_df,
            },
            base_interval="5m",
        )

        # 05:40 这根 5m bar 时，05:30-05:44 的 15m bar 尚未收盘，
        # 应该仍然只能看到上一根 15m bar（05:15）的 close。
        self.assertAlmostEqual(
            merged.loc[pd.Timestamp("2026-04-24 05:40:00"), "15m_close"],
            high_df.loc[pd.Timestamp("2026-04-24 05:15:00"), "close"],
        )

        # 到 05:45 时，05:30 的 15m bar 已经确认收盘，才能出现在特征里。
        self.assertAlmostEqual(
            merged.loc[pd.Timestamp("2026-04-24 05:45:00"), "15m_close"],
            high_df.loc[pd.Timestamp("2026-04-24 05:30:00"), "close"],
        )


if __name__ == "__main__":
    unittest.main()
