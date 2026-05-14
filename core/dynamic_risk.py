import math
from dataclasses import dataclass

from config import config


@dataclass(frozen=True)
class DynamicRiskDecision:
    enabled: bool
    signal_strength: float
    volatility_ratio: float
    atr_ratio: float
    trend_aligned: bool
    risk_multiplier: float
    effective_leverage: int
    max_position_ratio: float


class DynamicRiskController:
    def __init__(
        self,
        *,
        enabled=None,
        base_leverage=None,
        min_leverage=None,
        max_leverage=None,
        base_position_ratio=None,
        min_position_ratio=None,
        max_position_ratio=None,
        target_vol=None,
        high_vol_multiplier=None,
        low_signal_multiplier=None,
        trend_mismatch_multiplier=None,
        strong_signal_threshold=None,
        weak_signal_threshold=None,
    ):
        self.enabled = bool(config.DYNAMIC_RISK_ENABLED if enabled is None else enabled)
        self.base_leverage = max(1, int(base_leverage if base_leverage is not None else config.LEVERAGE))
        configured_min_leverage = max(
            1,
            int(min_leverage if min_leverage is not None else config.DYNAMIC_LEVERAGE_MIN),
        )
        self.min_leverage = min(self.base_leverage, configured_min_leverage)
        self.max_leverage = min(self.base_leverage, max(
            self.min_leverage,
            int(max_leverage if max_leverage is not None else config.DYNAMIC_LEVERAGE_MAX),
        ))
        self.base_position_ratio = float(
            base_position_ratio if base_position_ratio is not None else config.MAX_POSITION_RATIO
        )
        self.min_position_ratio = max(
            0.0,
            float(min_position_ratio if min_position_ratio is not None else config.POSITION_MIN),
        )
        self.max_position_ratio = max(
            self.min_position_ratio,
            float(max_position_ratio if max_position_ratio is not None else config.DYNAMIC_POSITION_MAX),
        )
        self.target_vol = max(1e-9, float(target_vol if target_vol is not None else config.TARGET_VOL))
        self.high_vol_multiplier = _clip_ratio(
            high_vol_multiplier if high_vol_multiplier is not None else config.DYNAMIC_RISK_HIGH_VOL_MULTIPLIER
        )
        self.low_signal_multiplier = _clip_ratio(
            low_signal_multiplier if low_signal_multiplier is not None else config.DYNAMIC_RISK_LOW_SIGNAL_MULTIPLIER
        )
        self.trend_mismatch_multiplier = _clip_ratio(
            trend_mismatch_multiplier if trend_mismatch_multiplier is not None else config.DYNAMIC_RISK_TREND_MISMATCH_MULTIPLIER
        )
        self.strong_signal_threshold = float(
            strong_signal_threshold
            if strong_signal_threshold is not None
            else config.DYNAMIC_RISK_STRONG_SIGNAL_THRESHOLD
        )
        self.weak_signal_threshold = float(
            weak_signal_threshold
            if weak_signal_threshold is not None
            else config.DYNAMIC_RISK_WEAK_SIGNAL_THRESHOLD
        )

    def evaluate(
        self,
        *,
        long_prob,
        short_prob,
        volatility,
        atr_ratio=None,
        trend_bias=None,
        target_direction=None,
    ):
        signal_strength = _clean_float(abs(float(long_prob) - float(short_prob)), default=0.0)
        volatility = _clean_float(volatility, default=self.target_vol)
        atr_ratio = _clean_float(atr_ratio, default=volatility)
        realized_risk = max(volatility, atr_ratio, 1e-9)
        volatility_ratio = realized_risk / self.target_vol
        trend_aligned = _trend_aligned(target_direction, trend_bias)

        risk_multiplier = 1.0
        if not self.enabled:
            return DynamicRiskDecision(
                enabled=False,
                signal_strength=signal_strength,
                volatility_ratio=volatility_ratio,
                atr_ratio=atr_ratio,
                trend_aligned=trend_aligned,
                risk_multiplier=risk_multiplier,
                effective_leverage=self._clip_leverage(self.base_leverage),
                max_position_ratio=self.base_position_ratio,
            )

        if volatility_ratio > 1.0:
            risk_multiplier *= max(self.high_vol_multiplier, 1.0 / volatility_ratio)
        elif volatility_ratio < 0.65:
            risk_multiplier *= min(1.15, 1.0 / max(volatility_ratio, 0.5))

        if signal_strength < self.weak_signal_threshold:
            risk_multiplier *= self.low_signal_multiplier
        elif signal_strength >= self.strong_signal_threshold:
            risk_multiplier *= 1.15

        if not trend_aligned:
            risk_multiplier *= self.trend_mismatch_multiplier

        risk_multiplier = max(0.25, min(1.25, risk_multiplier))
        effective_leverage = self._clip_leverage(round(self.base_leverage * risk_multiplier))
        max_position_ratio = min(
            self.max_position_ratio,
            max(self.min_position_ratio, self.base_position_ratio * risk_multiplier),
        )
        return DynamicRiskDecision(
            enabled=True,
            signal_strength=signal_strength,
            volatility_ratio=volatility_ratio,
            atr_ratio=atr_ratio,
            trend_aligned=trend_aligned,
            risk_multiplier=float(risk_multiplier),
            effective_leverage=int(effective_leverage),
            max_position_ratio=float(max_position_ratio),
        )

    def apply_to_target_ratio(self, target_ratio, decision):
        if not decision.enabled:
            return float(target_ratio)
        target_ratio = float(target_ratio) * float(decision.risk_multiplier)
        cap = max(0.0, float(decision.max_position_ratio))
        return max(-cap, min(cap, target_ratio))

    def _clip_leverage(self, leverage):
        leverage = int(round(float(leverage)))
        leverage = max(self.min_leverage, leverage)
        return min(self.max_leverage, leverage)


def _clip_ratio(value):
    value = _clean_float(value, default=1.0)
    return max(0.0, min(1.0, value))


def _clean_float(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def _trend_aligned(target_direction, trend_bias):
    target_direction = str(target_direction or "").lower()
    trend_bias = str(trend_bias or "neutral").lower()
    if target_direction not in {"long", "short"}:
        return True
    if trend_bias in {"", "neutral", "none"}:
        return True
    return target_direction == trend_bias
