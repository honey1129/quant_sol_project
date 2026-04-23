# live_trading_monitor.py
import os
import time
import json
import joblib
import traceback
import numpy as np
import pandas as pd
from core import ml_feature_engineering, signal_engine
from core.reward_risk import RewardRiskEstimator
from core.strategy_core import StrategyCore
from utils.utils import log_info, log_error, BASE_DIR
from config import config
from core.okx_api import OKXClient
from core.position_manager import PositionManager


LIVE_STATE_PATH = os.path.join(BASE_DIR, "logs", "live_trading_state.json")


def load_last_bar_ts(state_path):
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        raw_value = payload.get("last_bar_ts")
        if not raw_value:
            return None
        return pd.Timestamp(raw_value)
    except Exception:
        return None


def persist_last_bar_ts(state_path, bar_ts):
    if bar_ts is None:
        return
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    payload = {"last_bar_ts": pd.Timestamp(bar_ts).isoformat()}
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)
    os.replace(tmp_path, state_path)


class LiveTrader:
    def __init__(self, client):
        self.client = client
        self.position_manager = PositionManager()

        self.MIN_HOLD_BARS = config.MIN_HOLD_BARS
        self.ADD_THRESHOLD = config.ADD_THRESHOLD
        self.MAX_REBALANCE_RATIO = config.MAX_REBALANCE_RATIO
        self.MIN_ADJUST_AMOUNT = float(config.MIN_ADJUST_AMOUNT)

        # ===== 实盘状态 =====
        self.hold_bars = 0
        self.state_path = LIVE_STATE_PATH
        self.last_bar_ts = load_last_bar_ts(self.state_path) if bool(config.LIVE_PERSIST_LAST_BAR) else None

        # ===== 模型/特征=====
        feature_path = os.path.join(BASE_DIR, config.FEATURE_LIST_PATH) if "BASE_DIR" in globals() else config.FEATURE_LIST_PATH
        self.feature_cols = joblib.load(feature_path)

        model_paths = {n: os.path.join(BASE_DIR, p) for n, p in config.MODEL_PATHS.items()} if "BASE_DIR" in globals() else config.MODEL_PATHS
        self.models = signal_engine.load_models(model_paths)
        self.model_weights = config.MODEL_WEIGHTS


        self.reward_risk = self._load_reward_risk()
        self.core = StrategyCore(
            self.position_manager,
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
            min_hold_bars=self.MIN_HOLD_BARS,
            add_threshold=self.ADD_THRESHOLD,
            max_rebalance_ratio=self.MAX_REBALANCE_RATIO,
            min_adjust_amount=self.MIN_ADJUST_AMOUNT,
            reward_risk=float(self.reward_risk),
        )

        if self.last_bar_ts is not None:
            log_info(f"已恢复最近处理 bar: {self.last_bar_ts}")

    def ensure_runtime_ready(self):
        self.client.ensure_trading_ready()

    def _load_reward_risk(self):
        try:
            trades = self.client.fetch_recent_closed_trades()
            rr = RewardRiskEstimator()
            rr.batch_update(trades)
            val = float(rr.estimate())
            log_info(f"reward_risk={val:.4f}")
            return val
        except Exception as e:
            log_error(f"reward_risk 获取失败，使用默认 1.0：{e}")
            return 1.0

    def _predict_latest_probs(self, row: pd.Series):
        X = row[self.feature_cols].values.reshape(1, -1).astype(float)
        X = pd.DataFrame(X, columns=self.feature_cols)

        weighted_sum = np.zeros(2)
        total_weight = float(sum(self.model_weights.values()))

        for name, model in self.models.items():
            prob = model.predict_proba(X)[0]
            w = float(self.model_weights.get(name, 1.0))
            weighted_sum += prob * w

        avg = weighted_sum / max(total_weight, 1e-9)
        long_prob, short_prob = float(avg[1]), float(avg[0])
        return long_prob, short_prob

    def _get_latest_features(self):
        data_dict = self.client.fetch_data()
        merged_df = ml_feature_engineering.merge_multi_period_features(data_dict)
        merged_df = ml_feature_engineering.add_advanced_features(merged_df)
        merged_df = merged_df.dropna().copy()

        if len(merged_df) < 2:
            raise RuntimeError("特征数据不足，暂时无法生成已收盘 bar 信号")

        # 最后一根通常是尚未收盘的 bar，实盘统一使用上一根已收盘 bar。
        row = merged_df.iloc[-2]
        bar_ts = merged_df.index[-2]
        price = float(row["5m_close"])
        money_flow_ratio = float(row["money_flow_ratio"])

        if pd.notna(row.get("volatility_15")):
            volatility = float(row["volatility_15"])
        else:
            merged_df["log_return"] = np.log(merged_df["5m_close"] / merged_df["5m_close"].shift(1))
            volatility = float(merged_df["log_return"].rolling(96).std().iloc[-2])

        long_prob, short_prob = self._predict_latest_probs(row)
        atr_value = row.get("5m_atr")
        atr_ratio = None
        if pd.notna(atr_value) and price > 0:
            atr_ratio = float(atr_value) / price

        return bar_ts, price, long_prob, short_prob, money_flow_ratio, volatility, atr_ratio

    def _get_equity(self) -> float:
        account_balance = self.client.get_account_balance()
        balance = account_balance["data"][0]
        total_eq = float(balance.get("totalEq", 0) or 0)
        if total_eq > 0:
            return total_eq
        return float(balance["availEq"])

    def _sync_after_trade(self):
        pos_qty2, entry_price2 = self._get_net_position()
        if pos_qty2 == 0:
            self.hold_bars = 0
        self.core.set_state(pos_qty2, entry_price2, self.hold_bars)
        _, _, self.hold_bars = self.core.get_state()

    def _get_net_position(self):
        long_pos, short_pos = self.client.get_position()

        if long_pos["size"] > 0 and short_pos["size"] > 0:
            log_error("检测到同时多空持仓（与回测不一致），尝试双边平仓清理。")
            self.client.close_long_sz(long_pos["size"], config.LEVERAGE)
            self.client.close_short_sz(short_pos["size"], config.LEVERAGE)
            return 0.0, 0.0

        if long_pos["size"] > 0:
            return float(long_pos["size"]), float(long_pos["entry_price"])
        if short_pos["size"] > 0:
            return -float(short_pos["size"]), float(short_pos["entry_price"])
        return 0.0, 0.0

    def _execute_delta(self, current_pos_qty: float, delta_qty: float) -> bool:
        qty = abs(float(delta_qty))
        if qty <= 0:
            return False

        if current_pos_qty > 0:
            if delta_qty > 0:
                return self.client.open_long_sz(qty, config.LEVERAGE)
            return self.client.close_long_sz(qty, config.LEVERAGE)

        if current_pos_qty < 0:
            if delta_qty < 0:
                return self.client.open_short_sz(qty, config.LEVERAGE)
            return self.client.close_short_sz(qty, config.LEVERAGE)

        if delta_qty > 0:
            return self.client.open_long_sz(qty, config.LEVERAGE)
        return self.client.open_short_sz(qty, config.LEVERAGE)

    def _persist_last_bar_state(self, bar_ts):
        if not bool(config.LIVE_PERSIST_LAST_BAR):
            return
        persist_last_bar_ts(self.state_path, bar_ts)

    def run_once_on_new_bar(self):
        bar_ts, price, long_prob, short_prob, money_flow_ratio, volatility, atr_ratio = self._get_latest_features()

        if self.last_bar_ts is not None and bar_ts == self.last_bar_ts:
            return
        self.last_bar_ts = bar_ts
        self._persist_last_bar_state(bar_ts)

        if bool(config.LIVE_RECONCILE_PENDING_ORDERS):
            self.client.cancel_pending_orders()

        log_info(
            f"新bar={bar_ts} price={price:.4f} long={long_prob:.3f} short={short_prob:.3f} "
            f"mf={money_flow_ratio:.3f} vol={volatility:.6f} atr_ratio={0.0 if atr_ratio is None else atr_ratio:.4%}"
        )

        pos_qty, entry_price = self._get_net_position()

        equity = self._get_equity()

        if pos_qty == 0:
            self.hold_bars = 0
        self.core.set_state(pos_qty, entry_price, self.hold_bars)

        out = self.core.on_bar(
            price=price,
            equity=equity,
            long_prob=long_prob,
            short_prob=short_prob,
            money_flow_ratio=money_flow_ratio,
            volatility=volatility,
            atr_ratio=atr_ratio,
        )

        action = out["action"]
        delta = float(out["delta_qty"])

        if action == "CLOSE":
            success = False
            if pos_qty > 0:
                success = self.client.close_long_sz(abs(pos_qty), config.LEVERAGE)
            elif pos_qty < 0:
                success = self.client.close_short_sz(abs(pos_qty), config.LEVERAGE)
            self._sync_after_trade()
            if success:
                log_info(f"执行平仓: reason={out['reason']}")
            else:
                log_error(f"平仓未成交，已重新同步仓位: reason={out['reason']}")
            return

        elif action == "OPEN":
            success = self._execute_delta(pos_qty, delta)
            self._sync_after_trade()
            if success:
                log_info(f"执行开仓: target_ratio={out['target_ratio']:.3f}, qty={abs(delta):.6f}")
            else:
                log_error(f"开仓未成交，已重新同步仓位: target_ratio={out['target_ratio']:.3f}, qty={abs(delta):.6f}")
            return

        elif action == "REBALANCE":
            success = self._execute_delta(pos_qty, delta)
            self._sync_after_trade()
            if success:
                log_info(f"执行调仓: delta_qty={delta:.6f}, reason={out['reason']}")
            else:
                log_error(f"调仓未成交，已重新同步仓位: delta_qty={delta:.6f}, reason={out['reason']}")
            return

        elif action == "HOLD":
            _, _, self.hold_bars = self.core.get_state()
            log_info("无明显信号或目标为0：保持仓位不变")
            return


def run():
    POLL_SEC = config.POLL_SEC
    client = OKXClient()
    trader = LiveTrader(client)
    trader.ensure_runtime_ready()

    log_info("🟢 Live trading monitor started (daemon loop)")
    while True:
        try:
            trader.run_once_on_new_bar()
        except Exception as e:
            log_error(f"实盘循环异常: {e}")
            log_error(traceback.format_exc())

        time.sleep(int(POLL_SEC))


if __name__ == "__main__":
    run()
