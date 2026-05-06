import math


def _safe_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _row_get(row, key):
    if hasattr(row, "get"):
        return row.get(key)
    try:
        return row[key]
    except Exception:
        return None


def derive_trend_context(
    row,
    *,
    interval="1H",
    fast_col="ema_20",
    slow_col="ema_60",
    min_gap=0.001,
    price_col="5m_close",
):
    prefix = str(interval)
    fast_key = f"{prefix}_{fast_col}"
    slow_key = f"{prefix}_{slow_col}"

    fast = _safe_float(_row_get(row, fast_key))
    slow = _safe_float(_row_get(row, slow_key))
    price = _safe_float(_row_get(row, price_col))
    min_gap = max(0.0, float(min_gap))

    context = {
        "trend_bias": "neutral",
        "trend_gap": None,
        "trend_fast": fast,
        "trend_slow": slow,
        "trend_price": price,
        "trend_source": f"{prefix}:{fast_col}>{slow_col}",
        "trend_reason": "MissingTrendData",
    }

    if fast is None or slow is None or slow <= 0:
        return context

    gap = (fast - slow) / slow
    context["trend_gap"] = float(gap)

    if price is None or price <= 0:
        if gap > min_gap:
            context["trend_bias"] = "long"
            context["trend_reason"] = "EmaFastAboveSlow"
        elif gap < -min_gap:
            context["trend_bias"] = "short"
            context["trend_reason"] = "EmaFastBelowSlow"
        else:
            context["trend_reason"] = "TrendGapNeutral"
        return context

    if gap > min_gap and price >= fast * (1.0 - min_gap):
        context["trend_bias"] = "long"
        context["trend_reason"] = "PriceAboveFastAndFastAboveSlow"
    elif gap < -min_gap and price <= fast * (1.0 + min_gap):
        context["trend_bias"] = "short"
        context["trend_reason"] = "PriceBelowFastAndFastBelowSlow"
    else:
        context["trend_reason"] = "TrendNeutralOrPullback"

    return context


def trend_allows_direction(direction, trend_bias):
    direction = str(direction or "").lower()
    trend_bias = str(trend_bias or "neutral").lower()
    if direction not in {"long", "short"}:
        return True
    if trend_bias not in {"long", "short"}:
        return True
    return direction == trend_bias
