import unittest

import numpy as np
import pandas as pd

from core.ml_feature_engineering import (
    RUBIK_FEATURE_COLUMNS,
    add_rubik_features,
    model_feature_columns,
)


def _base_5m(n=240):
    idx = pd.date_range("2026-05-01 00:00:00", periods=n, freq="5min", tz="UTC")
    close = 100.0 + np.linspace(0, 5, n)
    return pd.DataFrame({"5m_close": close}, index=idx)


def _rubik_1h(n=24, start="2026-05-01 00:00:00"):
    ts = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return {
        "open_interest": pd.DataFrame({"ts": ts, "open_interest": np.linspace(1e8, 1.1e8, n),
                                       "oi_volume": np.linspace(1e6, 2e6, n)}),
        "taker_volume": pd.DataFrame({"ts": ts, "taker_sell_vol": np.full(n, 100.0),
                                      "taker_buy_vol": np.linspace(50.0, 300.0, n)}),
        "long_short_ratio": pd.DataFrame({"ts": ts, "long_short_ratio": np.linspace(0.8, 1.5, n)}),
    }


class RubikFeatureTests(unittest.TestCase):
    def test_disabled_when_no_data(self):
        df = _base_5m()
        out = add_rubik_features(df, None)
        # 关闭时不应引入任何 rubik 列
        self.assertEqual([c for c in out.columns if c.startswith("rubik")], [])

    def test_enabled_adds_stationary_columns(self):
        df = _base_5m()
        out = add_rubik_features(df, _rubik_1h())
        for col in RUBIK_FEATURE_COLUMNS:
            self.assertIn(col, out.columns)
        # taker imbalance 应在 [-1,1],是无量纲平稳量
        imb = out["rubik_taker_imbalance"]
        self.assertTrue((imb.abs() <= 1.0 + 1e-9).all())

    def test_no_lookahead_first_hour_is_zero(self):
        """第一个 1H bar(00:00)要到 01:00 收盘后才可见。

        00:00~00:55 的 5m bar 不应看到任何 rubik 值(填 0),否则就是未来函数。
        """
        df = _base_5m()
        out = add_rubik_features(df, _rubik_1h())
        first_hour = out.loc[out.index < pd.Timestamp("2026-05-01 01:00:00", tz="UTC")]
        # 这些 bar 对应的 rubik 派生值应全为 0(尚无已收盘的 1H 统计)
        self.assertTrue((first_hour["rubik_oi_vol_ratio"] == 0.0).all())
        # 01:00 之后应开始出现非零(00:00 的 bar 已收盘可见)
        later = out.loc[out.index >= pd.Timestamp("2026-05-01 01:05:00", tz="UTC")]
        self.assertGreater((later["rubik_oi_vol_ratio"] != 0.0).sum(), 0)

    def test_rubik_cols_enter_model_features_when_present(self):
        df = _base_5m()
        out = add_rubik_features(df, _rubik_1h())
        mf = set(model_feature_columns(out))
        for col in RUBIK_FEATURE_COLUMNS:
            self.assertIn(col, mf)


if __name__ == "__main__":
    unittest.main()
