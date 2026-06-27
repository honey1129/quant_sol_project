# predict_engine.py

import json
import os
import joblib
import pandas as pd
from config import config
from core.ml_feature_engineering import merge_multi_period_features, add_advanced_features
from core.okx_api import OKXClient
from core.position_manager import PositionManager
from core.strategy_core import StrategyCore
from core.dynamic_risk import DynamicRiskController
from core.trend_filter import derive_trend_context
from core.regime_filter import derive_market_regime
from core import signal_engine
from utils.utils import BASE_DIR

class MultiPeriodSignalPredictor:
    def __init__(self):
        self.fetcher = OKXClient()
        self.model_paths = {name: os.path.join(BASE_DIR, path) for name, path in config.MODEL_PATHS.items()}
        self.models = {name: joblib.load(path) for name, path in self.model_paths.items()}
        self.model_weights = config.MODEL_WEIGHTS
        self.direction_model_weights = getattr(config, "MODEL_DIRECTION_MODEL_WEIGHTS", {})
        metadata_path = os.path.join(BASE_DIR, config.TRAINING_METADATA_PATH)
        self.model_metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as file:
                    self.model_metadata = json.load(file)
            except Exception:
                self.model_metadata = {}
        self.core = StrategyCore(
            PositionManager(),
            threshold_long=config.THRESHOLD_LONG,
            threshold_short=config.THRESHOLD_SHORT,
            take_profit=config.TAKE_PROFIT,
            stop_loss=config.STOP_LOSS,
            adaptive_tp_sl_enabled=config.ADAPTIVE_TP_SL_ENABLED,
            atr_take_profit_multiplier=config.ATR_TAKE_PROFIT_MULTIPLIER,
            atr_stop_loss_multiplier=config.ATR_STOP_LOSS_MULTIPLIER,
            volatility_take_profit_multiplier=config.VOLATILITY_TAKE_PROFIT_MULTIPLIER,
            volatility_stop_loss_multiplier=config.VOLATILITY_STOP_LOSS_MULTIPLIER,
            adaptive_take_profit_min=config.ADAPTIVE_TAKE_PROFIT_MIN,
            adaptive_take_profit_max=config.ADAPTIVE_TAKE_PROFIT_MAX,
            adaptive_stop_loss_min=config.ADAPTIVE_STOP_LOSS_MIN,
            adaptive_stop_loss_max=config.ADAPTIVE_STOP_LOSS_MAX,
            min_hold_bars=config.MIN_HOLD_BARS,
            add_threshold=config.ADD_THRESHOLD,
            max_rebalance_ratio=config.MAX_REBALANCE_RATIO,
            min_adjust_amount=float(config.MIN_ADJUST_AMOUNT),
            signal_min_prob_diff=config.SIGNAL_MIN_PROB_DIFF,
            min_signal_target_ratio=config.MIN_SIGNAL_TARGET_RATIO,
            reverse_signal_min_prob_diff=config.REVERSE_SIGNAL_MIN_PROB_DIFF,
            reverse_min_target_ratio=config.REVERSE_MIN_TARGET_RATIO,
            reverse_exit_consecutive_bars=config.REVERSE_EXIT_CONSECUTIVE_BARS,
            reverse_exit_min_prob_diff=config.REVERSE_EXIT_MIN_PROB_DIFF,
            reward_risk=float(config.KELLY_REWARD_RISK),
            fee_rate=float(config.FEE_RATE),
            slippage_bps=float(config.ESTIMATED_SLIPPAGE_BPS),
            cost_buffer_multiplier=float(config.COST_BUFFER_MULTIPLIER),
            min_expected_net_edge=float(config.MIN_EXPECTED_NET_EDGE),
            min_take_profit_to_stop_loss_ratio=float(config.MIN_TAKE_PROFIT_TO_STOP_LOSS_RATIO),
            min_take_profit_cost_multiplier=float(config.MIN_TAKE_PROFIT_COST_MULTIPLIER),
            regime_high_vol_stop_loss_min=float(config.REGIME_HIGH_VOL_STOP_LOSS_MIN),
            trade_cooldown_bars=int(config.TRADE_COOLDOWN_BARS),
            take_profit_cooldown_bars=int(config.TAKE_PROFIT_COOLDOWN_BARS),
            stop_loss_cooldown_bars=int(config.STOP_LOSS_COOLDOWN_BARS),
            trend_filter_enabled=bool(config.TREND_FILTER_ENABLED),
            regime_filter_enabled=bool(config.REGIME_FILTER_ENABLED),
            regime_range_allow_trades=bool(config.REGIME_RANGE_ALLOW_TRADES),
            regime_high_vol_allow_trades=bool(config.REGIME_HIGH_VOL_ALLOW_TRADES),
            regime_range_threshold_bonus=float(config.REGIME_RANGE_THRESHOLD_BONUS),
            regime_high_vol_threshold_bonus=float(config.REGIME_HIGH_VOL_THRESHOLD_BONUS),
            regime_trend_against_block=bool(config.REGIME_TREND_AGAINST_BLOCK),
            regime_range_target_multiplier=float(config.REGIME_RANGE_TARGET_MULTIPLIER),
            regime_high_vol_target_multiplier=float(config.REGIME_HIGH_VOL_TARGET_MULTIPLIER),
            regime_range_min_signal_target_ratio=float(config.REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO),
            regime_high_vol_min_signal_target_ratio=float(config.REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO),
            block_losing_position_adds=bool(config.BLOCK_LOSING_POSITION_ADDS),
            loss_condition_guard_enabled=bool(config.LOSS_CONDITION_GUARD_ENABLED),
            loss_guard_block_new_regimes=config.LOSS_GUARD_BLOCK_NEW_REGIMES,
            loss_guard_block_directions=config.LOSS_GUARD_BLOCK_DIRECTIONS,
            loss_guard_exit_regimes=config.LOSS_GUARD_EXIT_REGIMES,
            loss_guard_exit_min_hold_bars=int(config.LOSS_GUARD_EXIT_MIN_HOLD_BARS),
            loss_guard_exit_only_when_unprofitable=bool(config.LOSS_GUARD_EXIT_ONLY_WHEN_UNPROFITABLE),
            loss_guard_exit_min_unrealized_loss=float(config.LOSS_GUARD_EXIT_MIN_UNREALIZED_LOSS),
            loss_guard_exit_confirm_bars=int(config.LOSS_GUARD_EXIT_CONFIRM_BARS),
            long_entry_guard_enabled=bool(config.LONG_ENTRY_GUARD_ENABLED),
            long_entry_min_trend_gap=float(config.LONG_ENTRY_MIN_TREND_GAP),
            long_entry_high_vol_gap_buffer=float(config.LONG_ENTRY_HIGH_VOL_GAP_BUFFER),
            long_entry_high_vol_min_trend_gap=float(config.LONG_ENTRY_HIGH_VOL_MIN_TREND_GAP),
            long_entry_block_high_vol=bool(config.LONG_ENTRY_BLOCK_HIGH_VOL),
            dynamic_risk_controller=DynamicRiskController(),
        )

    def get_latest_signal(self):
        # 多周期拉取数据
        data_dict = self.fetcher.fetch_data()
        merged_df = merge_multi_period_features(data_dict)
        merged_df = add_advanced_features(merged_df)
        merged_df = merged_df.dropna().copy()

        if merged_df.empty:
            raise ValueError("特征数据不足，暂时无法生成已收盘 bar 信号")

        # 统一使用最新一根已确认收盘 bar，和实盘监控逻辑保持一致。
        feature_cols = joblib.load(os.path.join(BASE_DIR, config.FEATURE_LIST_PATH))
        X_live = merged_df[feature_cols].iloc[-1:].astype(float)
        X_live = pd.DataFrame(X_live, columns=feature_cols)

        price = float(merged_df["5m_close"].iloc[-1])
        money_flow_ratio = float(merged_df["money_flow_ratio"].iloc[-1])
        volatility = float(merged_df["volatility_15"].iloc[-1])
        atr_value = merged_df["5m_atr"].iloc[-1] if "5m_atr" in merged_df.columns else None
        atr_ratio = None
        if pd.notna(atr_value) and price > 0:
            atr_ratio = float(atr_value) / price
        trend_context = derive_trend_context(
            merged_df.iloc[-1],
            interval=config.TREND_FILTER_INTERVAL,
            fast_col=config.TREND_FILTER_FAST_COL,
            slow_col=config.TREND_FILTER_SLOW_COL,
            min_gap=config.TREND_FILTER_MIN_GAP,
        )
        regime_context = derive_market_regime(
            trend_bias=trend_context.get("trend_bias"),
            trend_gap=trend_context.get("trend_gap"),
            volatility=volatility,
            atr_ratio=atr_ratio,
            money_flow_ratio=money_flow_ratio,
            trend_gap_threshold=config.REGIME_TREND_GAP_THRESHOLD,
            high_vol_atr_threshold=config.REGIME_HIGH_VOL_ATR_THRESHOLD,
            high_volatility_threshold=config.REGIME_HIGH_VOLATILITY_THRESHOLD,
            money_flow_extreme_threshold=config.REGIME_MONEY_FLOW_EXTREME_THRESHOLD,
        )
        # 多模型融合预测。二分类质量模型需要当前 trend_bias 才能映射到多/空方向。
        avg_prob = signal_engine.weighted_predict_proba(
            self.models,
            X_live,
            self.model_weights,
            trend_bias=trend_context.get("trend_bias"),
            model_metadata=self.model_metadata,
            direction_model_weights=self.direction_model_weights,
        )
        long_prob, short_prob = avg_prob[1], avg_prob[0]

        print(
            f"实时预测概率 => 多头: {long_prob:.3f} | 空头: {short_prob:.3f} | "
            f"趋势: {trend_context.get('trend_bias')} | regime: {regime_context.get('regime')}"
        )

        # 统一复用空仓状态下的真实策略开仓判定，避免和交易引擎口径不一致。
        self.core.set_state(position=0.0, entry_price=0.0, hold_bars=0)
        out = self.core.on_bar(
            price=price,
            equity=float(config.INITIAL_BALANCE),
            long_prob=float(long_prob),
            short_prob=float(short_prob),
            money_flow_ratio=money_flow_ratio,
            volatility=volatility,
            atr_ratio=atr_ratio,
            trend_bias=trend_context.get("trend_bias"),
            trend_gap=trend_context.get("trend_gap"),
            is_high_vol=bool(regime_context.get("is_high_vol")),
            market_regime=regime_context.get("regime"),
        )

        if out["action"] == "OPEN" and out["delta_qty"] > 0:
            return 'long'
        elif out["action"] == "OPEN" and out["delta_qty"] < 0:
            return 'short'
        else:
            return 'neutral'


if __name__ == '__main__':
    predictor = MultiPeriodSignalPredictor()
    signal = predictor.get_latest_signal()
    print(f"✅ 当前信号: {signal.upper()}")
