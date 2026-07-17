# core/regime_filter.py
import math


TREND_REGIMES = {"trend_long", "trend_short"}
RANGE_REGIMES = {"range", "range_high_vol"}
# 注意：HIGH_VOL_REGIMES 中的 "high_vol" 是兼容性保留值；
# derive_market_regime 不会返回它（无趋势+高波动统一返回 "range_high_vol"）。
# 如果在 LOSS_GUARD_BLOCK_NEW_REGIMES 等配置中写入 "high_vol"，该条件永远不会被触发。
HIGH_VOL_REGIMES = {"high_vol", "range_high_vol"}
SUPPORTED_REGIMES = {"trend_long", "trend_short", "range", "high_vol", "range_high_vol", "unknown"}


def _clean_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def derive_market_regime(
    *,
    trend_bias=None,
    trend_gap=None,
    volatility=None,
    atr_ratio=None,
    money_flow_ratio=None,
    trend_gap_threshold=0.003,
    high_vol_atr_threshold=0.0016,
    high_volatility_threshold=0.0012,
    money_flow_extreme_threshold=1.8,
):
    """Classify the latest bar into a coarse market regime.

    This is intentionally rule-based and lightweight: it lets one shared model be
    gated differently in trending/ranging/high-vol conditions without retraining
    separate model artifacts yet.
    """
    trend_bias = str(trend_bias or "neutral").lower()
    trend_gap = abs(_clean_float(trend_gap, 0.0) or 0.0)
    volatility = _clean_float(volatility, None)
    atr_ratio = _clean_float(atr_ratio, None)
    money_flow_ratio = _clean_float(money_flow_ratio, None)

    is_trending = trend_bias in {"long", "short"} and trend_gap >= max(0.0, float(trend_gap_threshold))
    is_high_vol = False
    if atr_ratio is not None and atr_ratio >= float(high_vol_atr_threshold):
        is_high_vol = True
    if volatility is not None and volatility >= float(high_volatility_threshold):
        is_high_vol = True
    if money_flow_ratio is not None and money_flow_ratio >= float(money_flow_extreme_threshold):
        is_high_vol = True

    if is_trending:
        return {
            "regime": "trend_long" if trend_bias == "long" else "trend_short",
            "regime_reason": "TrendHighVol" if is_high_vol else "Trend",
            "is_trending": True,
            "is_high_vol": bool(is_high_vol),
            "trend_gap": trend_gap,
        }
    if is_high_vol:
        return {
            "regime": "range_high_vol",
            "regime_reason": "HighVolNoTrend",
            "is_trending": False,
            "is_high_vol": True,
            "trend_gap": trend_gap,
        }
    return {
        "regime": "range",
        "regime_reason": "LowTrendGap",
        "is_trending": False,
        "is_high_vol": False,
        "trend_gap": trend_gap,
    }


def regime_allows_direction(regime, direction, *, allow_range=True, allow_high_vol=True):
    regime = str(regime or "unknown").lower()
    direction = str(direction or "").lower()
    if regime == "trend_long":
        return direction == "long"
    if regime == "trend_short":
        return direction == "short"
    if regime == "range":
        return bool(allow_range)
    if regime in {"high_vol", "range_high_vol"}:
        return bool(allow_high_vol)
    return True


def regime_reason(regime):
    return f"RegimeFilter({regime or 'unknown'})"
