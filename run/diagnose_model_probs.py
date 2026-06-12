"""模型概率诊断脚本。

用真实行情过一遍当前集成模型,量化:
- long/short 概率分布(是否被"卡死"成常数,这是非平稳特征导致的典型故障)
- 方向命中率(预测方向 vs 未来收益符号,>50% 才优于抛硬币)
- 通过做多/做空门槛的 bar 数量是否大致均衡

用法:
    PYTHONPATH=. python -m run.diagnose_model_probs [--bars 1500]

这是评估特征/标签改动是否真正改善模型方向能力的核心工具,
配合 OOS 回测一起看(回测看 PnL,本脚本看概率质量)。
"""
import argparse
import os

import joblib
import numpy as np

from config import config
from core import signal_engine
from core.ml_feature_engineering import (
    add_advanced_features,
    merge_multi_period_features,
)
from core.okx_api import OKXClient
from utils.utils import BASE_DIR


def _quantiles(arr):
    return [round(float(np.quantile(arr, q)), 3) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]


def diagnose(bars=1500):
    client = OKXClient()
    merged = add_advanced_features(merge_multi_period_features(client.fetch_data()))
    merged = merged.dropna().copy()
    print(
        f"rows={len(merged)} "
        f"price={float(merged['5m_close'].min()):.2f}->{float(merged['5m_close'].max()):.2f}"
    )

    feature_cols = joblib.load(os.path.join(BASE_DIR, config.FEATURE_LIST_PATH))
    models = signal_engine.load_models(config.MODEL_PATHS)
    X = merged[feature_cols].astype(float)

    n = min(int(bars), len(X) - config.MODEL_LABEL_FUTURE_WINDOW)
    longs, shorts = [], []
    for i in range(len(X) - n, len(X)):
        probs = signal_engine.weighted_predict_proba(
            models, X.iloc[i:i + 1], config.MODEL_WEIGHTS
        )
        shorts.append(probs[0])
        longs.append(probs[1])
    longs = np.array(longs)
    shorts = np.array(shorts)
    gap = np.abs(longs - shorts)

    print(f"\nlast {n} bars ensemble (directional probs, no_trade mass ignored):")
    print("  long_prob  q0/25/50/75/100:", _quantiles(longs), "mean", round(longs.mean(), 3))
    print("  short_prob q0/25/50/75/100:", _quantiles(shorts), "mean", round(shorts.mean(), 3))
    print("  |gap|      q0/25/50/75/100:", _quantiles(gap))

    # 方向命中率:argmax(long,short) 是否预测对 future_window 后收益符号
    close = merged["5m_close"].to_numpy()
    window = int(config.MODEL_LABEL_FUTURE_WINDOW)
    fwd = np.full(len(close), np.nan)
    fwd[:-window] = close[window:] / close[:-window] - 1
    fwd = fwd[len(X) - n:]
    threshold = float(config.MODEL_LABEL_THRESHOLD)
    pred_long = longs > shorts
    mask = np.abs(fwd) > threshold
    hit = ((pred_long & (fwd > 0)) | (~pred_long & (fwd < 0)))[mask]
    if mask.sum():
        print(
            f"\n  directional hit-rate on moves>{threshold:.1%}: "
            f"{hit.mean():.1%} (n={int(mask.sum())}) [50%=coin flip]"
        )
    print(
        f"  predicted long share: {pred_long.mean():.1%} "
        f"short share: {(~pred_long).mean():.1%}"
    )

    pass_long = int(((longs > config.THRESHOLD_LONG) & (gap >= config.SIGNAL_MIN_PROB_DIFF)).sum())
    pass_short = int(((shorts > config.THRESHOLD_SHORT) & (gap >= config.SIGNAL_MIN_PROB_DIFF)).sum())
    print(
        f"\n  THRESHOLD_LONG={config.THRESHOLD_LONG} SHORT={config.THRESHOLD_SHORT} "
        f"MIN_DIFF={config.SIGNAL_MIN_PROB_DIFF}"
    )
    print(f"  bars passing LONG gate: {pass_long}/{n}  SHORT gate: {pass_short}/{n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="诊断集成模型的概率质量与方向命中率")
    parser.add_argument("--bars", type=int, default=1500, help="回看的最近 bar 数")
    args = parser.parse_args()
    diagnose(bars=args.bars)
