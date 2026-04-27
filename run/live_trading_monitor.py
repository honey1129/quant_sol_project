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
from utils.utils import DISPLAY_TIMEZONE
from utils.runtime_dashboard import write_runtime_dashboard_snapshot
from config import config
from core.okx_api import OKXClient
from core.position_manager import PositionManager


LIVE_STATE_PATH = os.path.join(BASE_DIR, "logs", "live_trading_state.json")
HEARTBEAT_LOG_INTERVAL_SEC = 30.0


def should_emit_interval_log(last_emitted_at, now_ts, interval_sec):
    if last_emitted_at is None:
        return True
    return float(now_ts) - float(last_emitted_at) >= float(interval_sec)


def load_last_bar_ts(state_path):
    state = load_runtime_state(state_path)
    return state.get("last_bar_ts")


def load_runtime_state(state_path):
    if not os.path.exists(state_path):
        return {"last_bar_ts": None, "hold_bars": 0}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {"last_bar_ts": None, "hold_bars": 0}

    last_bar_ts = None
    raw_value = payload.get("last_bar_ts")
    if raw_value:
        try:
            last_bar_ts = ensure_utc_timestamp(raw_value)
        except Exception:
            last_bar_ts = None

    try:
        hold_bars = int(payload.get("hold_bars", 0) or 0)
    except (TypeError, ValueError):
        hold_bars = 0
    if hold_bars < 0:
        hold_bars = 0

    return {"last_bar_ts": last_bar_ts, "hold_bars": hold_bars}


def ensure_utc_timestamp(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def persist_last_bar_ts(state_path, bar_ts):
    persist_runtime_state(state_path, last_bar_ts=bar_ts, hold_bars=0)


def persist_runtime_state(state_path, *, last_bar_ts, hold_bars):
    if last_bar_ts is None:
        return
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    payload = {
        "last_bar_ts": ensure_utc_timestamp(last_bar_ts).isoformat(),
        "hold_bars": int(hold_bars),
    }
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)
    os.replace(tmp_path, state_path)


def normalize_ts(value):
    if value is None:
        return None
    try:
        return ensure_utc_timestamp(value).isoformat()
    except Exception:
        return str(value)


def format_display_ts(value):
    if value is None:
        return "None"
    try:
        return ensure_utc_timestamp(value).tz_convert(DISPLAY_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)


class LiveTrader:
    def __init__(self, client):
        self.client = client
        self.position_manager = PositionManager()

        self.MIN_HOLD_BARS = config.MIN_HOLD_BARS
        self.ADD_THRESHOLD = config.ADD_THRESHOLD
        self.MAX_REBALANCE_RATIO = config.MAX_REBALANCE_RATIO
        self.MIN_ADJUST_AMOUNT = float(config.MIN_ADJUST_AMOUNT)

        # ===== 实盘状态 =====
        self.state_path = LIVE_STATE_PATH
        if bool(config.LIVE_PERSIST_LAST_BAR):
            persisted = load_runtime_state(self.state_path)
            self.last_bar_ts = persisted.get("last_bar_ts")
            self.hold_bars = persisted.get("hold_bars", 0)
        else:
            self.last_bar_ts = None
            self.hold_bars = 0
        self.loop_count = 0
        self.same_bar_skip_count = 0
        self.last_heartbeat_logged_at = None
        self.heartbeat_log_interval_sec = HEARTBEAT_LOG_INTERVAL_SEC
        self.last_dashboard_account = {}
        self.last_signal_snapshot = {}
        self.last_position_snapshot = {}
        self.last_bar_snapshot = {}
        self.last_execution = {}

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
            signal_min_prob_diff=config.SIGNAL_MIN_PROB_DIFF,
            min_signal_target_ratio=config.MIN_SIGNAL_TARGET_RATIO,
            reverse_signal_min_prob_diff=config.REVERSE_SIGNAL_MIN_PROB_DIFF,
            reverse_min_target_ratio=config.REVERSE_MIN_TARGET_RATIO,
            reward_risk=float(self.reward_risk),
        )

        if self.last_bar_ts is not None:
            log_info(f"已恢复最近处理 bar: {format_display_ts(self.last_bar_ts)}")

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
            fallback_rr = float(getattr(config, "KELLY_REWARD_RISK", 1.8) or 1.8)
            log_error(f"reward_risk 获取失败，使用默认 {fallback_rr:.4f}：{e}")
            return fallback_rr

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

        if merged_df.empty:
            raise RuntimeError("特征数据不足，暂时无法生成已收盘 bar 信号")

        # merge_multi_period_features 已经只保留确认收盘bar，并对高周期特征做了滞后一根对齐。
        row = merged_df.iloc[-1]
        bar_ts = ensure_utc_timestamp(merged_df.index[-1])
        try:
            price = float(self.client.get_price())
        except Exception as exc:
            log_error(f"get_price 失败，回退使用 bar 收盘价: {exc}")
            price = float(row["5m_close"])
        money_flow_ratio = float(row["money_flow_ratio"])

        if pd.notna(row.get("volatility_15")):
            volatility = float(row["volatility_15"])
        else:
            merged_df["log_return"] = np.log(merged_df["5m_close"] / merged_df["5m_close"].shift(1))
            volatility = float(merged_df["log_return"].rolling(96).std().iloc[-1])

        long_prob, short_prob = self._predict_latest_probs(row)
        atr_value = row.get("5m_atr")
        atr_ratio = None
        if pd.notna(atr_value) and price > 0:
            atr_ratio = float(atr_value) / price

        return bar_ts, price, long_prob, short_prob, money_flow_ratio, volatility, atr_ratio

    def _get_equity(self) -> float:
        account = self._get_account_snapshot()
        total_eq = float(account.get("total_eq", 0) or 0)
        if total_eq > 0:
            return total_eq
        return float(account.get("avail_eq", 0) or 0)

    def _get_account_snapshot(self):
        account_balance = self.client.get_account_balance()
        balance = account_balance["data"][0]
        snapshot = {
            "total_eq": float(balance.get("totalEq", 0) or 0),
            "avail_eq": float(balance.get("availEq", 0) or 0),
            "currency": "USDT",
        }
        self.last_dashboard_account = snapshot
        return snapshot

    def _build_position_snapshot(self, pos_qty, entry_price, current_price=None, pending_orders=None):
        direction = "flat"
        if pos_qty > 0:
            direction = "long"
        elif pos_qty < 0:
            direction = "short"

        notional = None
        if current_price is not None:
            notional = abs(float(pos_qty)) * float(current_price)

        return {
            "direction": direction,
            "net_qty": float(pos_qty),
            "entry_price": float(entry_price),
            "hold_bars": int(self.hold_bars),
            "notional": notional,
            "pending_orders": pending_orders,
        }

    def _build_mixed_position_snapshot(self, long_pos, short_pos, current_price=None, pending_orders=None):
        long_size = float(long_pos.get("size", 0) or 0)
        short_size = float(short_pos.get("size", 0) or 0)
        long_entry_price = float(long_pos.get("entry_price", 0) or 0)
        short_entry_price = float(short_pos.get("entry_price", 0) or 0)
        notional = None
        if current_price is not None:
            notional = (long_size + short_size) * float(current_price)

        return {
            "direction": "mixed",
            "net_qty": long_size - short_size,
            "entry_price": None,
            "long_entry_price": long_entry_price if long_entry_price > 0 else None,
            "short_entry_price": short_entry_price if short_entry_price > 0 else None,
            "hold_bars": int(self.hold_bars),
            "notional": notional,
            "pending_orders": pending_orders,
            "long_qty": long_size,
            "short_qty": short_size,
        }

    def _write_dashboard_snapshot(
        self,
        *,
        runtime_status,
        latest_closed_bar_ts=None,
        current_price=None,
        signal_snapshot=None,
        account_snapshot=None,
        position_snapshot=None,
        decision=None,
        error_message=None,
    ):
        if signal_snapshot is not None:
            self.last_signal_snapshot = signal_snapshot
        if latest_closed_bar_ts is not None:
            self.last_bar_snapshot = {
                "last_processed_bar_ts": normalize_ts(self.last_bar_ts),
                "latest_closed_bar_ts": normalize_ts(latest_closed_bar_ts),
            }
        if account_snapshot is not None:
            self.last_dashboard_account = account_snapshot
        if position_snapshot is not None:
            self.last_position_snapshot = position_snapshot

        payload = {
            "runtime": {
                "last_status": runtime_status,
                "loop_count": int(self.loop_count),
                "same_bar_skip_count": int(self.same_bar_skip_count),
                "poll_sec": int(config.POLL_SEC),
                "heartbeat_interval_sec": float(self.heartbeat_log_interval_sec),
                "last_error": error_message,
            },
            "market": {
                "exchange": "OKX",
                "symbol": config.SYMBOL,
                "last_price": current_price,
                "leverage": float(config.LEVERAGE),
                "simulated": str(config.USE_SERVER) == "1",
            },
            "bar": self.last_bar_snapshot,
            "signal": signal_snapshot if signal_snapshot is not None else self.last_signal_snapshot,
            "account": account_snapshot if account_snapshot is not None else self.last_dashboard_account,
            "position": position_snapshot if position_snapshot is not None else self.last_position_snapshot,
            "decision": decision or {},
            "last_execution": self.last_execution,
        }

        history_point = None
        account_for_history = payload.get("account") or {}
        if account_for_history:
            history_point = {
                "bar_ts": normalize_ts(latest_closed_bar_ts or self.last_bar_ts),
                "total_eq": account_for_history.get("total_eq"),
                "avail_eq": account_for_history.get("avail_eq"),
                "price": current_price,
                "position_qty": (payload.get("position") or {}).get("net_qty"),
            }

        write_runtime_dashboard_snapshot(payload, history_point=history_point)

    def _sync_after_trade(self):
        pos_qty2, entry_price2 = self._get_net_position()
        if pos_qty2 is None:
            return
        if pos_qty2 == 0:
            self.hold_bars = 0
        self.core.set_state(pos_qty2, entry_price2, self.hold_bars)
        _, _, self.hold_bars = self.core.get_state()

    def _get_position_sides(self):
        return self.client.get_position()

    def _reconcile_dual_side_position(self, *, bar_ts, price, signal_snapshot):
        long_pos, short_pos = self._get_position_sides()
        long_size = float(long_pos.get("size", 0) or 0)
        short_size = float(short_pos.get("size", 0) or 0)
        if long_size <= 0 or short_size <= 0:
            return False

        log_error("检测到同时多空持仓，进入恢复流程，只做清仓对账，不执行新信号。")
        close_long_ok = self.client.close_long_sz(long_size, config.LEVERAGE)
        close_short_ok = self.client.close_short_sz(short_size, config.LEVERAGE)

        self.last_execution = {
            "action": "RECONCILE_POSITIONS",
            "reason": "DualSidePosition",
            "success": bool(close_long_ok and close_short_ok),
            "timestamp": normalize_ts(bar_ts),
        }

        account_snapshot = self._get_account_snapshot()
        latest_long_pos, latest_short_pos = self._get_position_sides()
        self._write_dashboard_snapshot(
            runtime_status="running",
            latest_closed_bar_ts=bar_ts,
            current_price=price,
            signal_snapshot=signal_snapshot,
            account_snapshot=account_snapshot,
            position_snapshot=self._build_mixed_position_snapshot(
                latest_long_pos,
                latest_short_pos,
                current_price=price,
                pending_orders=0,
            ),
            decision={
                "action": "RECONCILE_POSITIONS",
                "reason": "DualSidePosition",
            },
        )
        return True

    def _get_net_position(self):
        long_pos, short_pos = self._get_position_sides()
        if long_pos["size"] > 0 and short_pos["size"] > 0:
            return None, None

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
        persist_runtime_state(self.state_path, last_bar_ts=bar_ts, hold_bars=int(self.hold_bars))

    def _maybe_log_same_bar_heartbeat(self, current_bar_ts):
        now_ts = time.monotonic()
        if not should_emit_interval_log(
            self.last_heartbeat_logged_at,
            now_ts,
            self.heartbeat_log_interval_sec,
        ):
            return

        self.last_heartbeat_logged_at = now_ts
        log_info(
            f"心跳: 运行中，最近已处理bar={format_display_ts(self.last_bar_ts)}, "
            f"当前最新已收盘bar={format_display_ts(current_bar_ts)}, 连续跳过同bar次数={self.same_bar_skip_count}"
        )

    def run_once_on_new_bar(self):
        self.loop_count += 1
        bar_ts, price, long_prob, short_prob, money_flow_ratio, volatility, atr_ratio = self._get_latest_features()
        signal_snapshot = {
            "long_prob": float(long_prob),
            "short_prob": float(short_prob),
            "money_flow_ratio": float(money_flow_ratio),
            "volatility": float(volatility),
            "atr_ratio": None if atr_ratio is None else float(atr_ratio),
        }

        if self._reconcile_dual_side_position(
            bar_ts=bar_ts,
            price=price,
            signal_snapshot=signal_snapshot,
        ):
            return

        if self.last_bar_ts is not None and bar_ts == self.last_bar_ts:
            self.same_bar_skip_count += 1
            self._maybe_log_same_bar_heartbeat(bar_ts)
            self._write_dashboard_snapshot(
                runtime_status="waiting_next_bar",
                latest_closed_bar_ts=bar_ts,
                current_price=price,
                signal_snapshot=signal_snapshot,
                decision={
                    "action": "WAIT_SAME_BAR",
                    "reason": "SameClosedBarSkip",
                },
            )
            return

        self.same_bar_skip_count = 0
        self.last_bar_ts = bar_ts
        self.last_heartbeat_logged_at = time.monotonic()

        if bool(config.LIVE_RECONCILE_PENDING_ORDERS):
            self.client.cancel_pending_orders()

        log_info(
            f"新bar={format_display_ts(bar_ts)} price={price:.4f} long={long_prob:.3f} short={short_prob:.3f} "
            f"mf={money_flow_ratio:.3f} vol={volatility:.6f} atr_ratio={0.0 if atr_ratio is None else atr_ratio:.4%}"
        )

        pos_qty, entry_price = self._get_net_position()
        if pos_qty is None:
            log_error("双边持仓仍未清理完成，本轮跳过信号执行。")
            return
        account_snapshot = self._get_account_snapshot()
        equity = float(account_snapshot.get("total_eq", 0) or account_snapshot.get("avail_eq", 0) or 0)

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
        decision = {
            "action": action,
            "reason": out.get("reason"),
            "target_ratio": float(out.get("target_ratio", 0.0) or 0.0),
            "target_position": float(out.get("target_position", 0.0) or 0.0),
            "delta_qty": delta,
        }

        if action == "CLOSE":
            success = False
            if pos_qty > 0:
                success = self.client.close_long_sz(abs(pos_qty), config.LEVERAGE)
            elif pos_qty < 0:
                success = self.client.close_short_sz(abs(pos_qty), config.LEVERAGE)
            self._sync_after_trade()
            account_snapshot = self._get_account_snapshot()
            pos_qty, entry_price = self._get_net_position()
            self.last_execution = {
                "action": "CLOSE",
                "reason": out["reason"],
                "success": bool(success),
                "timestamp": normalize_ts(bar_ts),
            }
            self._write_dashboard_snapshot(
                runtime_status="running",
                latest_closed_bar_ts=bar_ts,
                current_price=price,
                signal_snapshot=signal_snapshot,
                account_snapshot=account_snapshot,
                position_snapshot=self._build_position_snapshot(pos_qty, entry_price, current_price=price, pending_orders=0),
                decision=decision,
            )
            if success:
                log_info(f"执行平仓: reason={out['reason']}")
            else:
                log_error(f"平仓未成交，已重新同步仓位: reason={out['reason']}")
            self._persist_last_bar_state(bar_ts)
            return

        elif action == "OPEN":
            success = self._execute_delta(pos_qty, delta)
            self._sync_after_trade()
            account_snapshot = self._get_account_snapshot()
            pos_qty, entry_price = self._get_net_position()
            self.last_execution = {
                "action": "OPEN",
                "reason": out["reason"],
                "success": bool(success),
                "timestamp": normalize_ts(bar_ts),
            }
            self._write_dashboard_snapshot(
                runtime_status="running",
                latest_closed_bar_ts=bar_ts,
                current_price=price,
                signal_snapshot=signal_snapshot,
                account_snapshot=account_snapshot,
                position_snapshot=self._build_position_snapshot(pos_qty, entry_price, current_price=price, pending_orders=0),
                decision=decision,
            )
            if success:
                log_info(f"执行开仓: target_ratio={out['target_ratio']:.3f}, qty={abs(delta):.6f}")
            else:
                log_error(f"开仓未成交，已重新同步仓位: target_ratio={out['target_ratio']:.3f}, qty={abs(delta):.6f}")
            self._persist_last_bar_state(bar_ts)
            return

        elif action == "REBALANCE":
            success = self._execute_delta(pos_qty, delta)
            self._sync_after_trade()
            account_snapshot = self._get_account_snapshot()
            pos_qty, entry_price = self._get_net_position()
            self.last_execution = {
                "action": "REBALANCE",
                "reason": out["reason"],
                "success": bool(success),
                "timestamp": normalize_ts(bar_ts),
            }
            self._write_dashboard_snapshot(
                runtime_status="running",
                latest_closed_bar_ts=bar_ts,
                current_price=price,
                signal_snapshot=signal_snapshot,
                account_snapshot=account_snapshot,
                position_snapshot=self._build_position_snapshot(pos_qty, entry_price, current_price=price, pending_orders=0),
                decision=decision,
            )
            if success:
                log_info(f"执行调仓: delta_qty={delta:.6f}, reason={out['reason']}")
            else:
                log_error(f"调仓未成交，已重新同步仓位: delta_qty={delta:.6f}, reason={out['reason']}")
            self._persist_last_bar_state(bar_ts)
            return

        elif action == "HOLD":
            self.hold_bars = int(out.get("next_hold_bars", self.hold_bars))
            self.core.set_state(pos_qty, entry_price, self.hold_bars)
            self._write_dashboard_snapshot(
                runtime_status="running",
                latest_closed_bar_ts=bar_ts,
                current_price=price,
                signal_snapshot=signal_snapshot,
                account_snapshot=account_snapshot,
                position_snapshot=self._build_position_snapshot(pos_qty, entry_price, current_price=price, pending_orders=0),
                decision=decision,
            )
            log_info("无明显信号或目标为0：保持仓位不变")
            self._persist_last_bar_state(bar_ts)
            return


def run():
    POLL_SEC = config.POLL_SEC
    client = OKXClient()
    trader = LiveTrader(client)
    trader.ensure_runtime_ready()

    startup_account = {}
    startup_position = {}
    startup_price = None
    try:
        startup_account = trader._get_account_snapshot()
        startup_pos_qty, startup_entry_price = trader._get_net_position()
        try:
            startup_price = float(client.get_price())
        except Exception:
            startup_price = None
        if startup_pos_qty is None:
            long_pos, short_pos = trader._get_position_sides()
            startup_position = trader._build_mixed_position_snapshot(
                long_pos,
                short_pos,
                current_price=startup_price,
                pending_orders=0,
            )
        else:
            startup_position = trader._build_position_snapshot(
                startup_pos_qty,
                startup_entry_price,
                current_price=startup_price,
                pending_orders=0,
            )
    except Exception as exc:
        log_error(f"启动快照初始化失败，将继续进入主循环: {exc}")

    trader._write_dashboard_snapshot(
        runtime_status="starting",
        latest_closed_bar_ts=trader.last_bar_ts,
        current_price=startup_price,
        account_snapshot=startup_account,
        position_snapshot=startup_position,
        decision={
            "action": "START",
            "reason": "MonitorBoot",
        },
    )

    log_info(f"🟢 Live trading monitor started (daemon loop, poll_sec={POLL_SEC})")
    while True:
        try:
            trader.run_once_on_new_bar()
        except Exception as e:
            trader._write_dashboard_snapshot(
                runtime_status="error",
                latest_closed_bar_ts=trader.last_bar_ts,
                decision={
                    "action": "ERROR",
                    "reason": "LoopException",
                },
                error_message=str(e),
            )
            log_error(f"实盘循环异常: {e}")
            log_error(traceback.format_exc())

        time.sleep(int(POLL_SEC))


if __name__ == "__main__":
    run()
