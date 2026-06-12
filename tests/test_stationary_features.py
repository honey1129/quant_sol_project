import unittest

import numpy as np
import pandas as pd

from core.ml_feature_engineering import (
    add_advanced_features,
    add_stationary_features,
    merge_multi_period_features,
    model_feature_columns,
    _is_excluded_model_feature,
)


def build_ohlcv(index, base_price, step, volume_base=1000.0):
    values = []
    for i, ts in enumerate(index):
        close = base_price + step * i
        values.append(
            {
                "timestamp": ts,
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": volume_base + i,
            }
        )
    df = pd.DataFrame(values).set_index("timestamp")
    df["confirm"] = "1"
    return df


def build_merged(base_price, step):
    base_index = pd.date_range("2026-01-01 00:00:00", periods=600, freq="5min")
    i15 = pd.date_range("2026-01-01 00:00:00", periods=200, freq="15min")
    i1h = pd.date_range("2026-01-01 00:00:00", periods=60, freq="1h")
    data = {
        "5m": build_ohlcv(base_index, base_price, step),
        "15m": build_ohlcv(i15, base_price, step * 3),
        "1H": build_ohlcv(i1h, base_price, step * 12),
    }
    merged = merge_multi_period_features(data)
    return add_advanced_features(merged).dropna()


class StationaryFeatureTests(unittest.TestCase):
    def test_model_features_exclude_absolute_price_and_confirm_cols(self):
        merged = build_merged(base_price=100.0, step=0.05)
        feature_cols = model_feature_columns(merged)

        absolute_suffixes = (
            "_open", "_high", "_low", "_close", "_vwap",
            "_boll_mid", "_boll_upper", "_boll_lower",
            "_ema_10", "_ema_20", "_ema_30", "_ema_60",
            "_macd", "_macd_signal", "_macd_hist",
            "_tr", "_atr", "_atr_14", "_obv", "_momentum_10",
            "_confirm", "_is_confirmed", "_volume",
        )
        leaked = [
            c for c in feature_cols
            if c.endswith(absolute_suffixes)
        ]
        self.assertEqual(leaked, [], f"绝对量级列泄漏进模型特征: {leaked}")

        # 标签与原始绝对列不应进入模型特征
        for col in ("future_return", "target", "ema_12", "ema_26",
                    "money_flow", "money_flow_ma", "volume_ma"):
            self.assertNotIn(col, feature_cols)

    def test_raw_columns_preserved_in_dataframe(self):
        merged = build_merged(base_price=100.0, step=0.05)
        # 下游撮合/趋势判断依赖这些原始列，必须保留在 DataFrame 中
        for col in ("5m_close", "5m_open", "5m_high", "5m_low", "5m_atr",
                    "volatility_15", "money_flow_ratio", "15m_ema_20", "15m_ema_60"):
            self.assertIn(col, merged.columns)

    def test_stationary_features_present(self):
        merged = build_merged(base_price=100.0, step=0.05)
        feature_cols = set(model_feature_columns(merged))
        # 每个周期都应有相对价距 / 归一化 / obv zscore 派生列
        for col in ("5m_ema_60_rel", "5m_macd_norm", "5m_atr_norm", "5m_obv_zscore"):
            self.assertIn(col, feature_cols)
        # 同周期 close 的 _rel 恒为 0，不应生成
        self.assertNotIn("5m_close_rel", merged.columns)

    def test_stationary_features_are_scale_invariant(self):
        """核心回归保护:相同形态、不同价位段,平稳特征分布应几乎一致。

        这正是原模型失效的根因——绝对特征在不同价位段分布漂移。平稳特征
        必须跨价位带稳定,否则模型仍会记忆训练期价位。
        """
        low = build_merged(base_price=50.0, step=0.025)    # 低价位段
        high = build_merged(base_price=200.0, step=0.10)   # 高价位段(同比例斜率)

        feature_cols = [
            c for c in model_feature_columns(low)
            if c in model_feature_columns(high)
        ]
        stationary = [
            c for c in feature_cols
            if c.endswith(("_rel", "_norm", "_zscore"))
        ]
        self.assertGreater(len(stationary), 10)

        n = min(len(low), len(high), 200)
        for col in stationary:
            lo_mean = float(low[col].iloc[-n:].mean())
            hi_mean = float(high[col].iloc[-n:].mean())
            # 平稳特征跨价位段均值差应很小(绝对差 < 0.02)
            self.assertLess(
                abs(lo_mean - hi_mean), 0.02,
                f"{col} 在不同价位段分布漂移: low={lo_mean:.5f} high={hi_mean:.5f}",
            )

    def test_excluded_helper_matches_absolute_columns(self):
        self.assertTrue(_is_excluded_model_feature("5m_close"))
        self.assertTrue(_is_excluded_model_feature("1H_ema_60"))
        self.assertTrue(_is_excluded_model_feature("15m_confirm"))
        self.assertTrue(_is_excluded_model_feature("money_flow"))
        # 平稳派生列与已平稳列不应被排除
        self.assertFalse(_is_excluded_model_feature("5m_ema_60_rel"))
        self.assertFalse(_is_excluded_model_feature("5m_atr_norm"))
        self.assertFalse(_is_excluded_model_feature("5m_rsi"))
        self.assertFalse(_is_excluded_model_feature("money_flow_ratio"))
        self.assertFalse(_is_excluded_model_feature("volume_ratio"))


if __name__ == "__main__":
    unittest.main()
