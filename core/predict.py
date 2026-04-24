# predict_engine.py

import os
import joblib
import numpy as np
import pandas as pd
from config import config
from core.ml_feature_engineering import merge_multi_period_features, add_advanced_features
from core.okx_api import OKXClient
from core.position_manager import PositionManager
from core.strategy_core import StrategyCore
from utils.utils import BASE_DIR

class MultiPeriodSignalPredictor:
    def __init__(self):
        self.fetcher = OKXClient()
        self.model_paths = {name: os.path.join(BASE_DIR, path) for name, path in config.MODEL_PATHS.items()}
        self.models = {name: joblib.load(path) for name, path in self.model_paths.items()}
        self.model_weights = config.MODEL_WEIGHTS
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
            reward_risk=float(config.KELLY_REWARD_RISK),
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

        # 多模型融合预测
        weighted_sum = np.zeros(2)
        total_weight = sum(self.model_weights.values())

        for name, model in self.models.items():
            prob = model.predict_proba(X_live)[0]
            weight = self.model_weights.get(name, 1.0)
            weighted_sum += prob * weight

        avg_prob = weighted_sum / total_weight
        long_prob, short_prob = avg_prob[1], avg_prob[0]
        price = float(merged_df["5m_close"].iloc[-1])
        money_flow_ratio = float(merged_df["money_flow_ratio"].iloc[-1])
        volatility = float(merged_df["volatility_15"].iloc[-1])
        atr_value = merged_df["5m_atr"].iloc[-1] if "5m_atr" in merged_df.columns else None
        atr_ratio = None
        if pd.notna(atr_value) and price > 0:
            atr_ratio = float(atr_value) / price

        print(f"实时预测概率 => 多头: {long_prob:.3f} | 空头: {short_prob:.3f}")

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
