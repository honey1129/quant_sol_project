# live_trading_monitor.py
import os
import time
import json
import joblib
import traceback
import numpy as np
import pandas as pd
from collections import Counter
from core import ml_feature_engineering, signal_engine, trend_filter
from core.reward_risk import get_configured_reward_risk
from core.strategy_core import StrategyCore
from core.dynamic_risk import DynamicRiskController
from core.trend_filter import derive_trend_context
from core.regime_filter import derive_market_regime
from utils.utils import log_info, log_error, notify_important, BASE_DIR, LOGS_DIR
from utils.utils import DISPLAY_TIMEZONE
from utils.runtime_dashboard import write_runtime_dashboard_snapshot
from utils.trade_audit import build_trade_record, append_trade_record, write_daily_report
from config import config
from core.okx_api import OKXClient
from core.position_manager import PositionManager


LIVE_STATE_PATH = os.path.join(BASE_DIR, "logs", "live_trading_state.json")
HEARTBEAT_LOG_INTERVAL_SEC = 30.0
TELEGRAM_RUNTIME_SUMMARY_INTERVAL_SEC = 3600.0
TELEGRAM_HOLD_ALERT_MIN_BARS = 3
TELEGRAM_HOLD_ALERT_COOLDOWN_SEC = 1800.0
TELEGRAM_HOLD_ALERT_REASONS = (
    "SmallTarget",
    "CostGate",
    "RegimeFilter",
    "TrendFilter",
    "LossGuardDirection",
    "LossGuardRegime",
    "LongEntryGuard",
)


def should_emit_interval_log(last_emitted_at, now_ts, interval_sec):
    if last_emitted_at is None:
        return True
    return float(now_ts) - float(last_emitted_at) >= float(interval_sec)


def load_last_bar_ts(state_path):
    state = load_runtime_state(state_path)
    return state.get("last_bar_ts")


def load_runtime_state(state_path):
    if not os.path.exists(state_path):
        return {
            "last_bar_ts": None,
            "hold_bars": 0,
            "cooldown_bars_remaining": 0,
            "reverse_signal_bars": 0,
            "loss_guard_exit_bars": 0,
            "position_qty": None,
            "entry_price": None,
            "take_profit": None,
            "stop_loss": None,
            "active_algo_id": "",
        }
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {
            "last_bar_ts": None,
            "hold_bars": 0,
            "cooldown_bars_remaining": 0,
            "reverse_signal_bars": 0,
            "loss_guard_exit_bars": 0,
            "position_qty": None,
            "entry_price": None,
            "take_profit": None,
            "stop_loss": None,
            "active_algo_id": "",
        }

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

    try:
        cooldown_bars_remaining = int(payload.get("cooldown_bars_remaining", 0) or 0)
    except (TypeError, ValueError):
        cooldown_bars_remaining = 0
    if cooldown_bars_remaining < 0:
        cooldown_bars_remaining = 0

    try:
        reverse_signal_bars = int(payload.get("reverse_signal_bars", 0) or 0)
    except (TypeError, ValueError):
        reverse_signal_bars = 0
    if reverse_signal_bars < 0:
        reverse_signal_bars = 0

    try:
        loss_guard_exit_bars = int(payload.get("loss_guard_exit_bars", 0) or 0)
    except (TypeError, ValueError):
        loss_guard_exit_bars = 0
    if loss_guard_exit_bars < 0:
        loss_guard_exit_bars = 0

    def optional_float(key):
        value = payload.get(key)
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "last_bar_ts": last_bar_ts,
        "hold_bars": hold_bars,
        "cooldown_bars_remaining": cooldown_bars_remaining,
        "reverse_signal_bars": reverse_signal_bars,
        "loss_guard_exit_bars": loss_guard_exit_bars,
        "position_qty": optional_float("position_qty"),
        "entry_price": optional_float("entry_price"),
        "take_profit": optional_float("take_profit"),
        "stop_loss": optional_float("stop_loss"),
        "active_algo_id": str(payload.get("active_algo_id", "") or ""),
    }


def ensure_utc_timestamp(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def persist_last_bar_ts(state_path, bar_ts):
    persist_runtime_state(
        state_path,
        last_bar_ts=bar_ts,
        hold_bars=0,
        cooldown_bars_remaining=0,
        reverse_signal_bars=0,
        loss_guard_exit_bars=0,
        position_qty=0.0,
        entry_price=0.0,
    )

def persist_runtime_state(
    state_path,
    *,
    last_bar_ts,
    hold_bars,
    cooldown_bars_remaining=0,
    reverse_signal_bars=0,
    loss_guard_exit_bars=0,
    position_qty=None,
    entry_price=None,
    take_profit=None,
    stop_loss=None,
    active_algo_id="",
):
    if last_bar_ts is None:
        return
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    payload = {
        "last_bar_ts": ensure_utc_timestamp(last_bar_ts).isoformat(),
        "hold_bars": int(hold_bars),
        "cooldown_bars_remaining": max(0, int(cooldown_bars_remaining)),
        "reverse_signal_bars": max(0, int(reverse_signal_bars)),
        "loss_guard_exit_bars": max(0, int(loss_guard_exit_bars)),
        "position_qty": None if position_qty is None else float(position_qty),
        "entry_price": None if entry_price is None else float(entry_price),
        "take_profit": None if take_profit is None else float(take_profit),
        "stop_loss": None if stop_loss is None else float(stop_loss),
        "active_algo_id": str(active_algo_id or ""),
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


def fmt_optional(value, digits=2):
    try:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def net_position_from_sides(long_pos, short_pos, tolerance=1e-9):
    long_size = float((long_pos or {}).get("size", 0.0) or 0.0)
    short_size = float((short_pos or {}).get("size", 0.0) or 0.0)
    if long_size > tolerance and short_size > tolerance:
        return None, None
    if long_size > tolerance:
        return long_size, float((long_pos or {}).get("entry_price", 0.0) or 0.0)
    if short_size > tolerance:
        return -short_size, float((short_pos or {}).get("entry_price", 0.0) or 0.0)
    return 0.0, 0.0


class LiveTrader:
    def __init__(self, client):
        self.client = client
        self.position_manager = PositionManager()
        self.dynamic_risk_controller = DynamicRiskController()

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
            self.cooldown_bars_remaining = persisted.get("cooldown_bars_remaining", 0)
            self.reverse_signal_bars = persisted.get("reverse_signal_bars", 0)
            self.loss_guard_exit_bars = persisted.get("loss_guard_exit_bars", 0)
        else:
            self.last_bar_ts = None
            self.hold_bars = 0
            self.cooldown_bars_remaining = 0
            self.reverse_signal_bars = 0
            self.loss_guard_exit_bars = 0
            persisted = {
                "position_qty": None,
                "entry_price": None,
                "take_profit": None,
                "stop_loss": None,
                "active_algo_id": "",
            }

        self._restored_take_profit = persisted.get("take_profit")
        self._restored_stop_loss = persisted.get("stop_loss")
        self._restored_algo_id = str(persisted.get("active_algo_id", "") or "")
        self._startup_position_verified = False
        self._startup_position_qty = None
        self._startup_entry_price = 0.0

        # Runtime counters are only reusable when the persisted position fingerprint
        # still matches the exchange position.
        try:
            long_pos, short_pos = client.get_position()
            tolerance = max(1e-9, float(config.LOT_SIZE) / 2.0)
            actual_qty, actual_entry = net_position_from_sides(
                long_pos,
                short_pos,
                tolerance=tolerance,
            )

            persisted_qty = persisted.get("position_qty")
            position_mismatch = (
                persisted_qty is not None
                and actual_qty is not None
                and abs(float(persisted_qty) - float(actual_qty)) > tolerance
            )
            if actual_qty == 0.0:
                self.hold_bars = 0
                self.reverse_signal_bars = 0
                self.loss_guard_exit_bars = 0
                if persisted_qty is not None and abs(float(persisted_qty)) > tolerance:
                    self.cooldown_bars_remaining = max(
                        int(self.cooldown_bars_remaining),
                        int(config.TRADE_COOLDOWN_BARS),
                    )
                    log_info(
                        "重启一致性修正: 持久化状态有仓但 OKX 已空仓，"
                        f"进入保守冷却 cooldown={self.cooldown_bars_remaining}。"
                    )
            elif actual_qty is None or persisted_qty is None or position_mismatch:
                self.hold_bars = 0
                self.reverse_signal_bars = 0
                self.loss_guard_exit_bars = 0
                log_info(
                    "重启一致性修正: 持久化仓位与 OKX 实时仓位不一致，"
                    "已重置持仓计数并保留冷却状态。"
                )

            self._startup_position_verified = True
            self._startup_position_qty = actual_qty
            self._startup_entry_price = actual_entry
        except Exception as exc:
            log_error(f"启动仓位一致性检查失败，首次交易前必须重新对账: {exc}")
        self.loop_count = 0
        self.same_bar_skip_count = 0
        self.last_heartbeat_logged_at = None
        self.heartbeat_log_interval_sec = HEARTBEAT_LOG_INTERVAL_SEC
        self.last_dashboard_account = {}
        self.last_signal_snapshot = {}
        self.last_position_snapshot = {}
        self.last_bar_snapshot = {}
        self.last_execution = {}
        self.consecutive_loop_errors = 0
        self.last_error_notified_count = 0
        self.last_runtime_summary_notified_at = None
        self.last_runtime_summary_equity = None
        self.hold_reason_counts = Counter()
        self.recent_hold_decisions = []
        self.consecutive_abnormal_hold_reason = None
        self.consecutive_abnormal_hold_count = 0
        self.last_hold_alert_notified_at = None
        self.last_hold_alert_key = None

        # ===== 账户级安全门禁 =====
        self._trading_halted = False   # Kill Switch 或日亏损熔断触发后置 True
        self._halt_reason: str = ""
        self._daily_start_equity: float = None  # 当日起始权益（每日重置）
        self._daily_start_date: str = None       # 当日日期（UTC，用于跨天重置）
        # 交易后同步 OKX 仓位失败时置 True，阻止下一轮新开仓/加仓
        self._position_uncertain: bool = False

        # ===== 交易所端 TP/SL 算法单追踪 =====
        # 每次开仓后记录 algoId，平仓前撤销，防止进程崩溃后残留单触发
        self._active_algo_id: str = self._restored_algo_id
        self._tpsl_coverage_verified = not bool(config.EXCHANGE_TPSL_ENABLED)
        self._last_tpsl_reconcile_at = None

        # ===== 模型/特征=====
        feature_path = os.path.join(BASE_DIR, config.FEATURE_LIST_PATH) if "BASE_DIR" in globals() else config.FEATURE_LIST_PATH
        self.feature_cols = joblib.load(feature_path)
        metadata_path = os.path.join(BASE_DIR, config.TRAINING_METADATA_PATH) if "BASE_DIR" in globals() else config.TRAINING_METADATA_PATH
        self.model_metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as file:
                    self.model_metadata = json.load(file)
            except Exception as exc:
                log_error(f"模型训练元数据读取失败，按旧方向模型处理: {exc}")

        model_paths = {n: os.path.join(BASE_DIR, p) for n, p in config.MODEL_PATHS.items()} if "BASE_DIR" in globals() else config.MODEL_PATHS
        self.models = signal_engine.load_models(model_paths)
        self.model_weights = config.MODEL_WEIGHTS
        self.direction_model_weights = getattr(config, "MODEL_DIRECTION_MODEL_WEIGHTS", {})


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
            reverse_exit_consecutive_bars=config.REVERSE_EXIT_CONSECUTIVE_BARS,
            reverse_exit_min_prob_diff=config.REVERSE_EXIT_MIN_PROB_DIFF,
            reward_risk=float(self.reward_risk),
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
            long_entry_overheat_guard_enabled=bool(config.LONG_ENTRY_OVERHEAT_GUARD_ENABLED),
            long_entry_overheat_money_flow_max=float(config.LONG_ENTRY_OVERHEAT_MONEY_FLOW_MAX),
            dynamic_risk_controller=self.dynamic_risk_controller,
        )

        if self._restored_take_profit and self._restored_take_profit > 0:
            self.core.current_take_profit = float(self._restored_take_profit)
        if self._restored_stop_loss and self._restored_stop_loss > 0:
            self.core.current_stop_loss = float(self._restored_stop_loss)
        if self._startup_position_verified and self._startup_position_qty is not None:
            self.core.set_state(
                self._startup_position_qty,
                self._startup_entry_price,
                self.hold_bars,
                cooldown_bars_remaining=self.cooldown_bars_remaining,
                reverse_signal_bars=self.reverse_signal_bars,
                loss_guard_exit_bars=self.loss_guard_exit_bars,
            )

        # 启用简单规则模式
        if bool(config.USE_SIMPLE_RULE_MODE):
            self.core._simple_rule_mode = True
            self.core._simple_rule_position_size = float(config.SIMPLE_RULE_POSITION_SIZE)
            log_info(f"✅ 简单规则模式已启用 - 仓位: {config.SIMPLE_RULE_POSITION_SIZE:.0%}, 绕过ML模型")
        log_info(
            "LossGuard配置: "
            f"enabled={int(bool(config.LOSS_CONDITION_GUARD_ENABLED))} "
            f"block_new_regimes={','.join(config.LOSS_GUARD_BLOCK_NEW_REGIMES) or '-'} "
            f"block_directions={','.join(config.LOSS_GUARD_BLOCK_DIRECTIONS) or '-'} "
            f"exit_regimes={','.join(config.LOSS_GUARD_EXIT_REGIMES) or '-'} "
            f"exit_min_hold_bars={int(config.LOSS_GUARD_EXIT_MIN_HOLD_BARS)} "
            f"exit_only_when_unprofitable={int(bool(config.LOSS_GUARD_EXIT_ONLY_WHEN_UNPROFITABLE))} "
            f"exit_min_unrealized_loss={float(config.LOSS_GUARD_EXIT_MIN_UNREALIZED_LOSS):.4%} "
            f"exit_confirm_bars={int(config.LOSS_GUARD_EXIT_CONFIRM_BARS)}"
        )
        log_info(
            "LongEntryGuard配置: "
            f"enabled={int(bool(config.LONG_ENTRY_GUARD_ENABLED))} "
            f"min_trend_gap={float(config.LONG_ENTRY_MIN_TREND_GAP):.4%} "
            f"high_vol_min_trend_gap={float(config.LONG_ENTRY_HIGH_VOL_MIN_TREND_GAP):.4%} "
            f"block_high_vol_regime={int(bool(config.LONG_ENTRY_BLOCK_HIGH_VOL))} "
            f"overheat_guard={int(bool(config.LONG_ENTRY_OVERHEAT_GUARD_ENABLED))} "
            f"overheat_money_flow_max={float(config.LONG_ENTRY_OVERHEAT_MONEY_FLOW_MAX):.3f}"
        )

        if self.last_bar_ts is not None:
            log_info(f"已恢复最近处理 bar: {format_display_ts(self.last_bar_ts)}")

    def ensure_runtime_ready(self):
        self.client.ensure_trading_ready()
        pos_qty, entry_price = self._get_net_position()
        self._startup_position_verified = True
        self._startup_position_qty = pos_qty
        self._startup_entry_price = 0.0 if entry_price is None else float(entry_price)
        if pos_qty is not None:
            self.core.set_state(
                pos_qty,
                entry_price,
                self.hold_bars,
                cooldown_bars_remaining=self.cooldown_bars_remaining,
                reverse_signal_bars=self.reverse_signal_bars,
                loss_guard_exit_bars=self.loss_guard_exit_bars,
            )
        self._reconcile_exchange_tpsl_on_startup()

    def _load_reward_risk(self):
        reward_risk = get_configured_reward_risk()
        log_info(f"实盘使用固定 reward_risk={reward_risk:.4f}（与回测一致）")
        return reward_risk

    def _predict_latest_probs(self, row: pd.Series):
        """
        预测 long_prob 和 short_prob

        如果开启简单规则模式(USE_SIMPLE_RULE_MODE=True):
            - 绕过ML模型
            - 根据 trend_bias 返回固定概率
            - trend_long -> long_prob=0.9, short_prob=0.1
            - trend_short -> long_prob=0.1, short_prob=0.9
            - neutral -> long_prob=0.5, short_prob=0.5
        """
        if bool(config.USE_SIMPLE_RULE_MODE):
            # 简单规则模式 - 从特征推断 trend
            trend_context = trend_filter.derive_trend_context(
                row,
                interval=config.TREND_FILTER_INTERVAL,
                fast_col=config.TREND_FILTER_FAST_COL,
                slow_col=config.TREND_FILTER_SLOW_COL,
                min_gap=config.TREND_FILTER_MIN_GAP,
            )
            trend_bias = trend_context.get('trend_bias', 'neutral')

            if trend_bias == 'long':
                return 0.90, 0.10
            elif trend_bias == 'short':
                return 0.10, 0.90
            else:
                return 0.50, 0.50

        trend_context = derive_trend_context(
            row,
            interval=config.TREND_FILTER_INTERVAL,
            fast_col=config.TREND_FILTER_FAST_COL,
            slow_col=config.TREND_FILTER_SLOW_COL,
            min_gap=config.TREND_FILTER_MIN_GAP,
        )

        # 正常ML模式
        X = row[self.feature_cols].values.reshape(1, -1).astype(float)
        X = pd.DataFrame(X, columns=self.feature_cols)

        avg = signal_engine.weighted_predict_proba(
            self.models,
            X,
            self.model_weights,
            trend_bias=trend_context.get("trend_bias"),
            model_metadata=self.model_metadata,
            direction_model_weights=self.direction_model_weights,
        )
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

        trend_context = derive_trend_context(
            row,
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

        return bar_ts, price, long_prob, short_prob, money_flow_ratio, volatility, atr_ratio, trend_context, regime_context

    def _get_equity(self) -> float:
        account = self._get_account_snapshot()
        return self._get_sizing_equity(account)

    def _get_sizing_equity(self, account_snapshot) -> float:
        total_eq = float(account_snapshot.get("total_eq", 0) or 0)
        avail_eq = float(account_snapshot.get("avail_eq", 0) or 0)
        if not bool(config.LIVE_USE_AVAILABLE_MARGIN_FOR_SIZING):
            return total_eq if total_eq > 0 else avail_eq

        usable_margin = max(0.0, avail_eq - float(config.LIVE_MIN_FREE_MARGIN_USDT))
        usable_margin *= max(0.0, min(float(config.LIVE_MARGIN_USAGE_RATIO), 1.0))
        margin_backed_equity = usable_margin * float(config.LEVERAGE)

        if total_eq > 0 and margin_backed_equity > 0:
            return min(total_eq, margin_backed_equity)
        if margin_backed_equity > 0:
            return margin_backed_equity
        return total_eq if total_eq > 0 else avail_eq

    def _get_account_snapshot(self):
        account_balance = self.client.get_account_balance()
        balance = account_balance["data"][0]
        snapshot = {
            "total_eq": float(balance.get("totalEq", 0) or 0),
            "avail_eq": float(balance.get("availEq", 0) or 0),
            "currency": "USDT",
        }
        snapshot["sizing_eq"] = self._get_sizing_equity(snapshot)
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
                "cooldown_bars_remaining": int(getattr(self, "cooldown_bars_remaining", 0)),
                "reverse_signal_bars": int(getattr(self, "reverse_signal_bars", 0)),
                "loss_guard_exit_bars": int(getattr(self, "loss_guard_exit_bars", 0)),
                "last_error": error_message,
            },
            "market": {
                "exchange": "OKX",
                "symbol": config.SYMBOL,
                "last_price": current_price,
                "leverage": float(config.LEVERAGE),
                "dynamic_risk_enabled": bool(config.DYNAMIC_RISK_ENABLED),
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
            # OKX 仓位查询失败（双边持仓/网络超时），内存状态不可信
            # 标记为不确定，阻止下一轮新开仓/加仓，避免用过期内存值执行错误方向交易
            self._position_uncertain = True
            log_error("交易后仓位同步失败，position_uncertain=True，下一轮将拒绝新开仓/加仓")
            return False
        # 查询成功，清除不确定标志
        self._position_uncertain = False
        if pos_qty2 == 0:
            self.hold_bars = 0
        self.core.set_state(
            pos_qty2,
            entry_price2,
            self.hold_bars,
            cooldown_bars_remaining=self.cooldown_bars_remaining,
            reverse_signal_bars=self.reverse_signal_bars,
            loss_guard_exit_bars=self.loss_guard_exit_bars,
        )
        _, _, self.hold_bars = self.core.get_state()
        self.reverse_signal_bars = self.core.get_reverse_signal_bars()
        self.loss_guard_exit_bars = self.core.get_loss_guard_exit_bars()
        self._startup_position_verified = True
        self._startup_position_qty = pos_qty2
        self._startup_entry_price = entry_price2
        return True

    def _get_position_sides(self):
        return self.client.get_position()

    def _reconcile_dual_side_position(self, *, bar_ts, price, signal_snapshot):
        long_pos, short_pos = self._get_position_sides()
        long_size = float(long_pos.get("size", 0) or 0)
        short_size = float(short_pos.get("size", 0) or 0)
        if long_size <= 0 or short_size <= 0:
            return False

        log_error("检测到同时多空持仓，进入恢复流程，只做清仓对账，不执行新信号。")
        account_before = self._get_account_snapshot()
        close_long_ok = self.client.close_long_sz(long_size, config.LEVERAGE)
        close_short_ok = self.client.close_short_sz(short_size, config.LEVERAGE)
        account_after = self._get_account_snapshot()
        latest_long_pos, latest_short_pos = self._get_position_sides()

        if isinstance(close_long_ok, dict):
            self._record_trade_execution(
                order_result=close_long_ok,
                action="CLOSE",
                reason="DualSidePosition",
                delta_qty=-abs(long_size),
                reference_price=price,
                bar_ts=bar_ts,
                signal_snapshot=signal_snapshot,
                decision={
                    "action": "RECONCILE_POSITIONS",
                    "reason": "DualSidePosition",
                },
                account_before=account_before,
                pos_qty_before=abs(long_size),
                entry_price_before=float(long_pos.get("entry_price", 0) or 0),
                account_after=account_after,
                pos_qty_after=0.0,
                entry_price_after=0.0,
            )
        if isinstance(close_short_ok, dict):
            self._record_trade_execution(
                order_result=close_short_ok,
                action="CLOSE",
                reason="DualSidePosition",
                delta_qty=abs(short_size),
                reference_price=price,
                bar_ts=bar_ts,
                signal_snapshot=signal_snapshot,
                decision={
                    "action": "RECONCILE_POSITIONS",
                    "reason": "DualSidePosition",
                },
                account_before=account_before,
                pos_qty_before=-abs(short_size),
                entry_price_before=float(short_pos.get("entry_price", 0) or 0),
                account_after=account_after,
                pos_qty_after=0.0,
                entry_price_after=0.0,
            )

        self.last_execution = {
            "action": "RECONCILE_POSITIONS",
            "reason": "DualSidePosition",
            "success": bool(close_long_ok and close_short_ok),
            "timestamp": normalize_ts(bar_ts),
        }

        account_snapshot = account_after
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
        return net_position_from_sides(long_pos, short_pos)

    def _resolve_order_leverage(self, decision=None):
        risk = (decision or {}).get("risk") or {}
        leverage = risk.get("effective_leverage", config.LEVERAGE)
        try:
            leverage = int(round(float(leverage)))
        except (TypeError, ValueError):
            leverage = int(config.LEVERAGE)
        return max(1, min(int(config.LEVERAGE), leverage))

    def _execute_delta(self, current_pos_qty: float, delta_qty: float, decision=None) -> bool:
        qty = abs(float(delta_qty))
        if qty <= 0:
            return False

        try:
            actual_pos_qty, _ = self._get_net_position()
        except Exception as exc:
            log_error(f"执行前仓位对账失败，本轮拒绝下单: {exc}")
            return False
        tolerance = max(1e-9, float(config.LOT_SIZE) / 2.0)
        if actual_pos_qty is None or abs(float(actual_pos_qty) - float(current_pos_qty)) > tolerance:
            log_error(
                "执行前仓位已变化，本轮拒绝下单: "
                f"decision_pos={float(current_pos_qty):.6f}, actual_pos={actual_pos_qty}"
            )
            return False
        blocked_directions = set(str(item).lower() for item in config.LOSS_GUARD_BLOCK_DIRECTIONS)
        if delta_qty < 0 and current_pos_qty >= 0 and "short" in blocked_directions:
            log_error(
                "LossGuard运行时保险: 已拒绝short新开仓 "
                f"qty={qty:.6f}, current_pos={float(current_pos_qty):.6f}, "
                f"reason={(decision or {}).get('reason') or '-'}"
            )
            return False
        leverage = self._resolve_order_leverage(decision)

        if current_pos_qty > 0:
            if delta_qty > 0:
                return self.client.open_long_sz(qty, leverage)
            return self.client.close_long_sz(qty, leverage)

        if current_pos_qty < 0:
            if delta_qty < 0:
                return self.client.open_short_sz(qty, leverage)
            return self.client.close_short_sz(qty, leverage)

        if delta_qty > 0:
            return self.client.open_long_sz(qty, leverage)
        return self.client.open_short_sz(qty, leverage)

    def _persist_last_bar_state(self, bar_ts):
        if not bool(config.LIVE_PERSIST_LAST_BAR):
            return
        position_qty, entry_price, _ = self.core.get_state()
        take_profit, stop_loss = self.core.get_risk_thresholds()
        persist_runtime_state(
            self.state_path,
            last_bar_ts=bar_ts,
            hold_bars=int(self.hold_bars),
            cooldown_bars_remaining=int(self.cooldown_bars_remaining),
            reverse_signal_bars=int(self.reverse_signal_bars),
            loss_guard_exit_bars=int(getattr(self, "loss_guard_exit_bars", 0)),
            position_qty=float(position_qty),
            entry_price=float(entry_price),
            take_profit=float(take_profit),
            stop_loss=float(stop_loss),
            active_algo_id=str(getattr(self, "_active_algo_id", "") or ""),
        )

    def _record_trade_execution(
        self,
        *,
        order_result,
        action,
        reason,
        delta_qty,
        reference_price,
        bar_ts,
        signal_snapshot,
        decision,
        account_before,
        pos_qty_before,
        entry_price_before,
        account_after,
        pos_qty_after,
        entry_price_after,
    ):
        if not isinstance(order_result, dict):
            return None

        record = build_trade_record(
            order_result,
            bar_ts=bar_ts,
            action=action,
            reason=reason,
            delta_qty=delta_qty,
            reference_price=reference_price,
            pos_qty_before=pos_qty_before,
            entry_price_before=entry_price_before,
            pos_qty_after=pos_qty_after,
            entry_price_after=entry_price_after,
            account_before=account_before,
            account_after=account_after,
            signal_snapshot=signal_snapshot,
            decision=decision,
        )
        append_trade_record(record)
        try:
            summary, json_path, md_path = write_daily_report(record["trade_date"])
            log_info(
                f"成交已记录: date={record['trade_date']}, action={record['action']}, "
                f"net_pnl={record['net_realized_pnl']:.2f}, fee={record['fee_abs']:.2f}, "
                f"report={md_path}"
            )
            self.last_execution = {
                "action": record["action"],
                "reason": record["reason"],
                "success": True,
                "timestamp": record["executed_at"],
                "fill_price": record["fill_price"],
                "fee_abs": record["fee_abs"],
                "net_realized_pnl": record["net_realized_pnl"],
                "report_path": md_path,
            }
        except Exception as exc:
            log_error(f"日报生成失败: {exc}")
            self.last_execution = {
                "action": record["action"],
                "reason": record["reason"],
                "success": True,
                "timestamp": record["executed_at"],
                "fill_price": record["fill_price"],
                "fee_abs": record["fee_abs"],
                "net_realized_pnl": record["net_realized_pnl"],
            }
        return record

    def _notify_trade_execution(self, record):
        if not record:
            return

        action_labels = {
            "OPEN": "开仓",
            "CLOSE": "平仓",
            "REBALANCE": "调仓",
        }
        action = str(record.get("action") or "").upper()
        label = action_labels.get(action, action or "成交")
        lines = [
            f"[交易{label}] {record.get('symbol') or config.SYMBOL}",
            f"方向: {record.get('pos_side') or '-'} / {record.get('side') or '-'}",
            f"原因: {record.get('reason') or '-'}",
            f"成交: price={fmt_optional(record.get('fill_price'), 4)}, qty={fmt_optional(record.get('fill_size'), 6)}",
            f"名义金额: {fmt_optional(record.get('notional'), 2)} USDT",
            f"净实现PnL: {fmt_optional(record.get('net_realized_pnl'), 2)} USDT",
            f"手续费: {fmt_optional(record.get('fee_abs'), 2)} {record.get('fee_currency') or 'USDT'}",
            f"权益变化: {fmt_optional(record.get('equity_delta'), 2)} USDT",
        ]
        notify_important("\n".join(lines))

    @staticmethod
    def _fmt_optional_pct(value):
        if value is None:
            return "-"
        try:
            return f"{float(value):.4%}"
        except (TypeError, ValueError):
            return "-"

    @staticmethod
    def _fmt_optional_float(value, digits=4):
        if value is None:
            return "-"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "-"

    @staticmethod
    def _fmt_optional_bool(value):
        if value is None:
            return "-"
        return "1" if bool(value) else "0"

    @staticmethod
    def _risk_reasons_text(risk):
        reasons = (risk or {}).get("reasons") or []
        if isinstance(reasons, str):
            return reasons or "-"
        return ",".join(str(item) for item in reasons) or "-"

    @staticmethod
    def _fmt_optional_pct_brief(value, digits=1):
        if value is None:
            return "-"
        try:
            return f"{float(value) * 100:.{digits}f}%"
        except (TypeError, ValueError):
            return "-"

    @staticmethod
    def _humanize_position_direction(direction):
        labels = {
            "flat": "空仓",
            "long": "多头",
            "short": "空头",
        }
        key = str(direction or "").lower()
        return labels.get(key, direction or "-")

    @staticmethod
    def _equity_label():
        if str(getattr(config, "USE_SERVER", "1")) == "1":
            return "模拟盘权益（虚拟资金）"
        return "实盘权益"

    @staticmethod
    def _humanize_action(action):
        labels = {
            "HOLD": "暂不交易",
            "OPEN": "开仓",
            "CLOSE": "平仓",
            "REBALANCE": "调仓",
            "ADD": "加仓",
            "REDUCE": "减仓",
        }
        key = str(action or "").upper()
        return labels.get(key, action or "-")

    @staticmethod
    def _humanize_regime(regime):
        labels = {
            "trend_long": "上涨趋势",
            "trend_short": "下跌趋势",
            "range_high_vol": "高波动震荡",
            "range_low_vol": "低波动震荡",
            "range": "震荡",
            "unknown": "未知",
        }
        key = str(regime or "").lower()
        return labels.get(key, regime or "-")

    @staticmethod
    def _humanize_trend(trend):
        labels = {
            "long": "偏多",
            "short": "偏空",
            "neutral": "中性",
            "trend_long": "偏多",
            "trend_short": "偏空",
        }
        key = str(trend or "").lower()
        return labels.get(key, trend or "-")

    @staticmethod
    def _humanize_reason(reason):
        raw = str(reason or "-")
        key = raw.split("(", 1)[0]
        labels = {
            "Cooldown": "冷却中，避免刚交易完立刻反复进出",
            "WeakSignal": "信号不够强，暂不交易",
            "CostGate": "预期收益不够覆盖手续费，不划算",
            "RegimeFilter": "当前行情环境不适合这个方向",
            "TrendFilter": "与大趋势方向冲突，放弃交易",
            "SmallTarget": "需要调整的量太小，不值得操作",
            "NoSignal": "没有明确的交易信号",
            "PositionLimit": "已达到最大仓位限制",
            "SameDirNoRebalance": "方向一致且仓位接近目标，继续持有",
            "Neutral": "多空信号持平，观望不动",
            "SameClosedBarSkip": "这根K线已处理过，等下一根",
            "LossGuardDirection": "亏损诊断保护：已禁止该方向新开仓",
            "LossGuardRegime": "亏损诊断保护：当前市场状态暂停新开仓",
            "LossGuardExit": "亏损诊断保护：进入高风险亏损状态，提前退出",
            "LongEntryGuard": "多头入场保护：趋势质量不够，暂不开多",
        }
        text = labels.get(key, raw)
        if key == "Cooldown" and "(" in raw and ")" in raw:
            remaining = raw.split("(", 1)[1].split(")", 1)[0]
            text = f"冷却中，还需等 {remaining} 根K线才能交易"
        if key == "MinHold" and "(" in raw and ")" in raw:
            hold_info = raw.split("(", 1)[1].split(")", 1)[0]
            text = f"最短持仓期未到（已持{hold_info}根K线），继续持有"
        if key == "LossGuardDirection" and "(" in raw and ")" in raw:
            direction = raw.split("(", 1)[1].split(")", 1)[0]
            direction_text = "做空" if direction == "short" else direction
            text = f"亏损诊断保护：已禁止{direction_text}新开仓"
        if key == "LossGuardRegime" and "(" in raw and ")" in raw:
            regime = raw.split("(", 1)[1].split(")", 1)[0]
            text = f"亏损诊断保护：{LiveTrader._humanize_regime(regime)}暂停新开仓"
        if key == "LossGuardExit" and "(" in raw and ")" in raw:
            regime = raw.split("(", 1)[1].split(")", 1)[0]
            text = f"亏损诊断保护：进入{LiveTrader._humanize_regime(regime)}，提前退出"
        if key == "LongEntryGuard" and "(" in raw and ")" in raw:
            detail = raw.split("(", 1)[1].split(")", 1)[0]
            if detail.startswith("weak_trend_gap="):
                text = f"多头入场保护：趋势强度不足（{detail.split('=', 1)[1]}），暂不开多"
            elif detail.startswith("overheat_money_flow="):
                text = f"多头入场保护：资金流过热（{detail.split('=', 1)[1]}），暂不开多"
            elif detail in {"range_high_vol", "high_vol"}:
                text = f"多头入场保护：{LiveTrader._humanize_regime(detail)}里不开新多"
            elif detail == "missing_trend_gap":
                text = "多头入场保护：缺少趋势强度数据，暂不开多"
            elif detail == "trend_not_long":
                text = "多头入场保护：大趋势不是偏多，暂不开多"
            else:
                text = f"多头入场保护：{detail}，暂不开多"
        return text

    def _format_hold_reason_counts(self):
        parts = []
        for reason, count in self.hold_reason_counts.most_common(5):
            parts.append(f"{self._humanize_reason(reason)} {count}次")
        return "，".join(parts) or "暂无"

    def _format_position_summary(self, position_snapshot):
        position_snapshot = position_snapshot or {}
        direction = self._humanize_position_direction(position_snapshot.get("direction"))
        qty = fmt_optional(position_snapshot.get("net_qty"), 6)
        entry = fmt_optional(position_snapshot.get("entry_price"), 4)
        notional = fmt_optional(position_snapshot.get("notional"), 2)
        if str(position_snapshot.get("direction") or "").lower() == "flat":
            return f"{direction}（数量 {qty}，名义金额 {notional} USDT）"
        return f"{direction}（数量 {qty}，开仓价 {entry}，名义金额 {notional} USDT）"

    def _format_equity_change(self, equity):
        prev = getattr(self, "last_runtime_summary_equity", None)
        try:
            current = float(equity)
        except (TypeError, ValueError):
            return "本次新增统计"
        if prev is None:
            return "本次新增统计"
        try:
            delta = current - float(prev)
        except (TypeError, ValueError):
            return "本次新增统计"
        sign = "+" if delta >= 0 else "-"
        pct = ""
        if prev:
            pct = f"（{sign}{abs(delta) / abs(float(prev)) * 100:.2f}%）"
        return f"{sign}{abs(delta):.2f} USDT{pct}"

    def _format_risk_summary_text(self, risk):
        risk = risk or {}
        if not risk:
            return "暂无动态风控数据"

        enabled = risk.get("enabled")
        enabled_text = "-" if enabled is None else ("已开启" if bool(enabled) else "已关闭")
        leverage = risk.get("effective_leverage")
        leverage_text = "-" if leverage is None else f"{leverage}倍"
        trend_aligned = risk.get("trend_aligned")
        trend_text = "-" if trend_aligned is None else ("✓ 顺势" if bool(trend_aligned) else "✗ 逆势")

        multiplier = risk.get("risk_multiplier")
        if multiplier is not None:
            try:
                m = float(multiplier)
                if m > 1.0:
                    multiplier_text = f"{m:.2f}（放大仓位）"
                elif m < 1.0:
                    multiplier_text = f"{m:.2f}（缩小仓位）"
                else:
                    multiplier_text = f"{m:.2f}（标准）"
            except (TypeError, ValueError):
                multiplier_text = "-"
        else:
            multiplier_text = "-"

        vol_ratio = risk.get("volatility_ratio")
        if vol_ratio is not None:
            try:
                v = float(vol_ratio)
                if v < 0.3:
                    vol_text = f"{v:.3f}（低波动，适合交易）"
                elif v < 0.7:
                    vol_text = f"{v:.3f}（中等波动）"
                else:
                    vol_text = f"{v:.3f}（高波动，注意风险）"
            except (TypeError, ValueError):
                vol_text = "-"
        else:
            vol_text = "-"

        return (
            f"风控{enabled_text}，仓位系数 {multiplier_text}，"
            f"杠杆 {leverage_text}，波动水平 {vol_text}，"
            f"方向 {trend_text}，判断依据 {self._humanize_risk_reasons(risk)}"
        )

    @staticmethod
    def _humanize_risk_reasons(risk):
        reasons = (risk or {}).get("reasons") or []
        if isinstance(reasons, str):
            reasons = [r.strip() for r in reasons.split(",") if r.strip()]
        if not reasons:
            return "-"
        labels = {
            "low_volatility": "波动低（利于交易）",
            "high_volatility": "波动高（缩减仓位）",
            "strong_signal": "信号强",
            "weak_signal": "信号弱",
            "trend_aligned": "顺势交易",
            "counter_trend": "逆势（缩减仓位）",
            "max_leverage_cap": "触及杠杆上限",
            "drawdown_protection": "回撤保护中",
        }
        parts = [labels.get(r, r) for r in reasons]
        return "，".join(parts)

    def _format_risk_fields(self, risk, *, compact=False):
        risk = risk or {}
        multiplier_label = "multiplier" if compact else "risk_multiplier"
        enabled_label = "enabled" if compact else "risk_enabled"
        leverage_label = "leverage" if compact else "risk_leverage"
        vol_label = "vol_ratio" if compact else "risk_vol_ratio"
        trend_label = "trend_aligned" if compact else "risk_trend_aligned"
        reasons_label = "reasons" if compact else "risk_reasons"
        leverage = risk.get("effective_leverage")
        leverage_text = "-" if leverage is None else str(leverage)
        return (
            f"{multiplier_label}={self._fmt_optional_float(risk.get('risk_multiplier'), 3)} "
            f"{enabled_label}={self._fmt_optional_bool(risk.get('enabled'))} "
            f"{leverage_label}={leverage_text} "
            f"{vol_label}={self._fmt_optional_float(risk.get('volatility_ratio'), 3)} "
            f"{trend_label}={self._fmt_optional_bool(risk.get('trend_aligned'))} "
            f"{reasons_label}={self._risk_reasons_text(risk)}"
        )

    def _format_hold_decision_log(self, decision, out, *, price, equity, pos_qty, entry_price):
        risk = out.get("risk") or {}
        return (
            "HOLD诊断: "
            f"reason={out.get('reason') or '-'} "
            f"target_ratio={float(out.get('target_ratio') or 0.0):.4f} "
            f"raw_target_ratio={float(out.get('raw_target_ratio') or 0.0):.4f} "
            f"target_position={float(out.get('target_position') or 0.0):.6f} "
            f"delta_qty={float(out.get('delta_qty') or 0.0):.6f} "
            f"edge={self._fmt_optional_pct(out.get('expected_net_edge'))} "
            f"tp={self._fmt_optional_pct(out.get('take_profit'))} "
            f"sl={self._fmt_optional_pct(out.get('stop_loss'))} "
            f"gap={self._fmt_optional_float(out.get('signal_prob_gap'), 4)} "
            f"dominant={self._fmt_optional_float(out.get('dominant_prob'), 4)} "
            f"pos={float(pos_qty or 0.0):.6f} "
            f"entry={float(entry_price or 0.0):.4f} "
            f"price={float(price):.4f} "
            f"equity={float(equity):.2f} "
            f"trend={decision.get('trend_bias') or 'neutral'} "
            f"regime={decision.get('market_regime') or '-'} "
            f"cooldown_next={int(out.get('next_cooldown_bars', 0) or 0)} "
            f"reverse_bars_next={int(out.get('next_reverse_signal_bars', 0) or 0)} "
            f"loss_guard_exit_bars_next={int(out.get('next_loss_guard_exit_bars', 0) or 0)} "
            f"{self._format_risk_fields(risk)}"
        )

    def _notify_trade_failure(self, action, reason, detail):
        notify_important(
            "[交易未成交]\n"
            f"动作: {action}\n"
            f"原因: {reason}\n"
            f"详情: {detail}\n"
            f"最近bar: {format_display_ts(self.last_bar_ts)}"
        )

    def _notify_consecutive_loop_error(self, error):
        threshold = max(1, int(getattr(config, "TELEGRAM_LOOP_ERROR_NOTIFY_THRESHOLD", 3)))
        if self.consecutive_loop_errors < threshold:
            return
        if self.consecutive_loop_errors == self.last_error_notified_count:
            return
        if self.consecutive_loop_errors > threshold and self.consecutive_loop_errors % 5 != 0:
            return

        self.last_error_notified_count = self.consecutive_loop_errors
        notify_important(
            "[实盘连续异常]\n"
            f"连续次数: {self.consecutive_loop_errors}\n"
            f"最近bar: {format_display_ts(self.last_bar_ts)}\n"
            f"错误: {error}"
        )

    def _hold_reason_key(self, reason):
        reason = str(reason or "-")
        if reason.startswith("CostGate"):
            return "CostGate"
        if reason.startswith("RegimeFilter"):
            return "RegimeFilter"
        if reason.startswith("TrendFilter"):
            return "TrendFilter"
        if reason.startswith("LossGuardDirection"):
            return "LossGuardDirection"
        if reason.startswith("LossGuardRegime"):
            return "LossGuardRegime"
        if reason.startswith("LossGuardExit"):
            return "LossGuardExit"
        if reason.startswith("LongEntryGuard"):
            return "LongEntryGuard"
        if reason.startswith("Cooldown"):
            return "Cooldown"
        return reason

    def _is_abnormal_hold_reason(self, reason):
        key = self._hold_reason_key(reason)
        return key in TELEGRAM_HOLD_ALERT_REASONS

    def _format_runtime_summary_notification(self, *, bar_ts, price, equity, position_snapshot, signal_snapshot, decision):
        position_snapshot = position_snapshot or {}
        signal_snapshot = signal_snapshot or {}
        decision = decision or {}
        risk = decision.get("risk") or {}

        # 浮盈浮亏计算
        unrealized_pnl_text = ""
        if str(position_snapshot.get("direction") or "").lower() != "flat":
            entry = position_snapshot.get("entry_price")
            qty = position_snapshot.get("net_qty")
            try:
                if entry and qty and price:
                    direction = str(position_snapshot.get("direction") or "").lower()
                    if direction == "long":
                        pnl = (float(price) - float(entry)) * float(qty)
                    else:
                        pnl = (float(entry) - float(price)) * float(qty)
                    pnl_pct = (float(price) - float(entry)) / float(entry) * 100
                    if direction == "short":
                        pnl_pct = -pnl_pct
                    sign = "+" if pnl >= 0 else ""
                    unrealized_pnl_text = f"，浮盈 {sign}{pnl:.2f} USDT（{sign}{pnl_pct:.2f}%）"
            except (TypeError, ValueError):
                pass

        return (
            "[实盘运行摘要]\n"
            f"⏰ 时间: {format_display_ts(bar_ts)}\n"
            f"\n"
            f"📈 行情: {config.SYMBOL} 当前价 ${fmt_optional(price, 4)}\n"
            f"💰 账户: {self._equity_label()} {fmt_optional(equity, 2)} USDT，"
            f"变化 {self._format_equity_change(equity)}\n"
            f"\n"
            f"📦 持仓: {self._format_position_summary(position_snapshot)}{unrealized_pnl_text}\n"
            f"\n"
            f"🤖 AI判断: 看涨概率 {self._fmt_optional_pct_brief(signal_snapshot.get('long_prob'))}，"
            f"看跌概率 {self._fmt_optional_pct_brief(signal_snapshot.get('short_prob'))}\n"
            f"🌊 市场环境: {self._humanize_regime(signal_snapshot.get('regime'))}，"
            f"趋势 {self._humanize_trend(signal_snapshot.get('trend_bias'))}\n"
            f"\n"
            f"🎬 本轮决策: {self._humanize_action(decision.get('action'))}\n"
            f"   └ 原因: {self._humanize_reason(decision.get('reason'))}\n"
            f"\n"
            f"📐 仓位管理:\n"
            f"   · 风控调整后目标仓位: {self._fmt_optional_pct_brief(decision.get('target_ratio'), 2)}\n"
            f"   · 模型建议原始仓位: {self._fmt_optional_pct_brief(decision.get('raw_target_ratio'), 2)}\n"
            f"   · 预期每笔净赚(扣除手续费): {self._fmt_optional_pct_brief(decision.get('expected_net_edge'), 2)}\n"
            f"   · 止盈线/止损线: {self._fmt_optional_pct_brief(decision.get('take_profit'), 2)} / "
            f"{self._fmt_optional_pct_brief(decision.get('stop_loss'), 2)}\n"
            f"\n"
            f"⏸️ 最近不交易的原因统计: {self._format_hold_reason_counts()}\n"
            f"\n"
            f"🛡️ 动态风控: {self._format_risk_summary_text(risk)}"
        )

    def _maybe_notify_runtime_summary(self, *, bar_ts, price, equity, position_snapshot, signal_snapshot, decision):
        now_ts = time.monotonic()
        if (
            self.last_runtime_summary_notified_at is not None
            and now_ts - float(self.last_runtime_summary_notified_at) < TELEGRAM_RUNTIME_SUMMARY_INTERVAL_SEC
        ):
            return
        self.last_runtime_summary_notified_at = now_ts
        notify_important(self._format_runtime_summary_notification(
            bar_ts=bar_ts,
            price=price,
            equity=equity,
            position_snapshot=position_snapshot,
            signal_snapshot=signal_snapshot,
            decision=decision,
        ))
        try:
            self.last_runtime_summary_equity = float(equity)
        except (TypeError, ValueError):
            pass

    def _maybe_notify_abnormal_hold(self, *, bar_ts, price, equity, decision, signal_snapshot):
        reason = decision.get("reason")
        key = self._hold_reason_key(reason)
        self.hold_reason_counts[key] += 1
        self.recent_hold_decisions.append({
            "bar_ts": normalize_ts(bar_ts),
            "reason": reason,
            "target_ratio": decision.get("target_ratio"),
            "raw_target_ratio": decision.get("raw_target_ratio"),
            "expected_net_edge": decision.get("expected_net_edge"),
            "market_regime": decision.get("market_regime"),
            "trend_bias": decision.get("trend_bias"),
        })
        self.recent_hold_decisions = self.recent_hold_decisions[-48:]

        if not self._is_abnormal_hold_reason(reason):
            self.consecutive_abnormal_hold_reason = None
            self.consecutive_abnormal_hold_count = 0
            return

        if key == self.consecutive_abnormal_hold_reason:
            self.consecutive_abnormal_hold_count += 1
        else:
            self.consecutive_abnormal_hold_reason = key
            self.consecutive_abnormal_hold_count = 1

        if self.consecutive_abnormal_hold_count < TELEGRAM_HOLD_ALERT_MIN_BARS:
            return

        now_ts = time.monotonic()
        alert_key = f"{key}:{self.consecutive_abnormal_hold_count // TELEGRAM_HOLD_ALERT_MIN_BARS}"
        if (
            self.last_hold_alert_notified_at is not None
            and now_ts - float(self.last_hold_alert_notified_at) < TELEGRAM_HOLD_ALERT_COOLDOWN_SEC
            and self.last_hold_alert_key == alert_key
        ):
            return

        self.last_hold_alert_notified_at = now_ts
        self.last_hold_alert_key = alert_key
        notify_important(
            "[异常HOLD聚合]\n"
            f"原因: {self._humanize_reason(reason)}\n"
            f"原始原因: {reason}\n"
            f"连续bar数: {self.consecutive_abnormal_hold_count}\n"
            f"最近bar: {format_display_ts(bar_ts)} price={fmt_optional(price, 4)} equity={fmt_optional(equity, 2)}\n"
            f"信号: long={fmt_optional(signal_snapshot.get('long_prob'), 3)} short={fmt_optional(signal_snapshot.get('short_prob'), 3)} "
            f"gap={fmt_optional(decision.get('signal_prob_gap'), 4)} dominant={fmt_optional(decision.get('dominant_prob'), 4)}\n"
            f"target={fmt_optional(decision.get('target_ratio'), 4)} raw={fmt_optional(decision.get('raw_target_ratio'), 4)} "
            f"edge={self._fmt_optional_pct(decision.get('expected_net_edge'))} "
            f"tp/sl={self._fmt_optional_pct(decision.get('take_profit'))}/{self._fmt_optional_pct(decision.get('stop_loss'))}\n"
            f"regime={decision.get('market_regime') or '-'} trend={decision.get('trend_bias') or '-'}"
        )

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

    def _place_exchange_tpsl(self, pos_side: str, pos_qty: float, entry_price: float, decision: dict):
        """开仓后立即在交易所下 TP/SL OCO 算法单。

        - 算法单 reduceOnly=True，只减仓不反向开仓。
        - algoId 保存到 self._active_algo_id，平仓前撤销。
        - 若下单失败，上层会触发紧急平仓并暂停新开仓。
        """
        if not getattr(config, "EXCHANGE_TPSL_ENABLED", True):
            self._tpsl_coverage_verified = True
            return True
        tp = float(decision.get("take_profit") or 0)
        sl = float(decision.get("stop_loss") or 0)
        if tp <= 0 or sl <= 0 or abs(pos_qty) < 1e-9 or entry_price <= 0:
            log_error(
                f"_place_exchange_tpsl: 参数不足，跳过交易所端止损单"
                f" tp={tp} sl={sl} qty={pos_qty} entry={entry_price}"
            )
            self._tpsl_coverage_verified = False
            return False
        algo_id = self.client.place_tpsl_algo_order(
            pos_side=pos_side,
            sz=abs(pos_qty),
            entry_price=entry_price,
            take_profit_ratio=tp,
            stop_loss_ratio=sl,
        )
        if algo_id:
            self._active_algo_id = algo_id
            self._tpsl_coverage_verified = True
            return True
        else:
            log_error("⚠ 交易所端 TP/SL 下单失败，将触发上层紧急平仓")
            self._tpsl_coverage_verified = False
            return False

    def _cancel_exchange_tpsl(self):
        """平仓前撤销活跃的交易所端 TP/SL 算法单，防止残留单触发后反向开仓。"""
        if not self._active_algo_id:
            return True
        canceled = self.client.cancel_algo_order(self._active_algo_id)
        if canceled:
            self._active_algo_id = ""
            self._tpsl_coverage_verified = False
        return bool(canceled)

    def _reconcile_exchange_tpsl_on_startup(self):
        if not bool(config.EXCHANGE_TPSL_ENABLED):
            self._tpsl_coverage_verified = True
            return True
        if not self._startup_position_verified or self._startup_position_qty is None:
            self._tpsl_coverage_verified = False
            return False

        try:
            pending = self.client.list_pending_tpsl_algo_orders()
        except Exception as exc:
            self._tpsl_coverage_verified = False
            log_error(f"启动时无法核验交易所 TP/SL 覆盖: {exc}")
            return False

        qty = float(self._startup_position_qty)
        tolerance = max(1e-9, float(config.LOT_SIZE) / 2.0)
        if abs(qty) <= tolerance:
            all_canceled = True
            for order in pending:
                algo_id = str(order.get("algoId", "") or "")
                if algo_id and not self.client.cancel_algo_order(algo_id):
                    all_canceled = False
            if all_canceled:
                self._active_algo_id = ""
                self._tpsl_coverage_verified = True
            return all_canceled

        pos_side = "long" if qty > 0 else "short"
        matching = []
        for order in pending:
            try:
                order_size = float(order.get("sz", 0) or 0)
            except (TypeError, ValueError):
                order_size = 0.0
            if str(order.get("posSide", "")) == pos_side and abs(order_size - abs(qty)) <= tolerance:
                matching.append(order)

        if len(pending) == 1 and len(matching) == 1:
            self._active_algo_id = str(matching[0].get("algoId", "") or "")
            self._tpsl_coverage_verified = bool(self._active_algo_id)
            if self._tpsl_coverage_verified:
                log_info(f"启动时已接管交易所 TP/SL: algoId={self._active_algo_id}")
            return self._tpsl_coverage_verified

        for order in pending:
            algo_id = str(order.get("algoId", "") or "")
            if algo_id and not self.client.cancel_algo_order(algo_id):
                self._tpsl_coverage_verified = False
                return False
        self._active_algo_id = ""
        take_profit, stop_loss = self.core.get_risk_thresholds()
        return self._place_exchange_tpsl(
            pos_side=pos_side,
            pos_qty=qty,
            entry_price=float(self._startup_entry_price),
            decision={"take_profit": take_profit, "stop_loss": stop_loss},
        )

    def _replace_exchange_tpsl(self, pos_qty, entry_price, decision):
        if not bool(config.EXCHANGE_TPSL_ENABLED):
            return True
        if not self._cancel_exchange_tpsl():
            log_error("旧 TP/SL 未确认撤销，拒绝创建可能重复的保护单。")
            return False
        if abs(float(pos_qty)) < 1e-9:
            self._tpsl_coverage_verified = True
            return True
        return self._place_exchange_tpsl(
            pos_side="long" if float(pos_qty) > 0 else "short",
            pos_qty=float(pos_qty),
            entry_price=float(entry_price),
            decision=decision,
        )

    def _close_unprotected_position(self, pos_qty, reason="TPSLCoverageFailure"):
        self._trading_halted = True
        self._halt_reason = reason
        leverage = int(config.LEVERAGE)
        if float(pos_qty) > 0:
            closed = self.client.close_long_sz(abs(float(pos_qty)), leverage)
        else:
            closed = self.client.close_short_sz(abs(float(pos_qty)), leverage)
        if closed and self._sync_after_trade():
            self.cooldown_bars_remaining = max(
                int(self.cooldown_bars_remaining),
                int(config.STOP_LOSS_COOLDOWN_BARS),
            )
            self.reverse_signal_bars = 0
            self.loss_guard_exit_bars = 0
            self._tpsl_coverage_verified = True
            notify_important(
                "[保护单失败紧急平仓]\n"
                f"原因: {reason}\n"
                f"原仓位: {float(pos_qty):.6f}\n"
                "新开仓已暂停，需检查 OKX 算法单接口后重启。"
            )
            return True
        notify_important(
            "[严重风险] TP/SL 保护单失败且紧急平仓未确认，"
            f"仓位={float(pos_qty):.6f}，请立即人工检查 OKX。"
        )
        return False

    def run_realtime_risk_check(self):
        """Check local TP/SL against a live ticker independently of bar generation."""
        pos_qty, entry_price = self._get_net_position()
        if pos_qty is None:
            log_error("实时风控检测到双边仓位，交由仓位恢复流程处理。")
            return False
        if abs(float(pos_qty)) < 1e-9 or float(entry_price) <= 0:
            return False

        now = time.monotonic()
        if (
            bool(config.EXCHANGE_TPSL_ENABLED)
            and not self._tpsl_coverage_verified
            and (
                self._last_tpsl_reconcile_at is None
                or now - float(self._last_tpsl_reconcile_at) >= 30.0
            )
        ):
            self._last_tpsl_reconcile_at = now
            self._startup_position_verified = True
            self._startup_position_qty = pos_qty
            self._startup_entry_price = entry_price
            self._reconcile_exchange_tpsl_on_startup()

        price = float(self.client.get_price())
        take_profit, stop_loss = self.core.get_risk_thresholds()
        pnl_pct = (
            (price - float(entry_price)) / float(entry_price)
            if pos_qty > 0
            else (float(entry_price) - price) / float(entry_price)
        )
        reason = None
        if pnl_pct >= float(take_profit):
            reason = "TakeProfitRealtime"
        elif pnl_pct <= -float(stop_loss):
            reason = "StopLossRealtime"
        if reason is None:
            return False

        log_error(
            f"实时风控触发 {reason}: price={price:.6f}, entry={float(entry_price):.6f}, "
            f"pnl={pnl_pct:.4%}"
        )
        self._cancel_exchange_tpsl()
        leverage = int(config.LEVERAGE)
        if pos_qty > 0:
            success = self.client.close_long_sz(abs(float(pos_qty)), leverage)
        else:
            success = self.client.close_short_sz(abs(float(pos_qty)), leverage)
        if not success:
            log_error(f"实时风控平仓未确认成交: reason={reason}")
            return False

        self.cooldown_bars_remaining = int(
            config.STOP_LOSS_COOLDOWN_BARS
            if reason.startswith("StopLoss")
            else config.TAKE_PROFIT_COOLDOWN_BARS
        )
        self.reverse_signal_bars = 0
        self.loss_guard_exit_bars = 0
        if not self._sync_after_trade():
            raise RuntimeError("实时风控平仓后仓位状态无法确认")
        self._tpsl_coverage_verified = True
        self.last_execution = {
            "action": "CLOSE",
            "reason": reason,
            "success": True,
            "timestamp": normalize_ts(pd.Timestamp.now(tz="UTC")),
        }
        self._persist_last_bar_state(self.last_bar_ts)
        notify_important(
            "[实时风控平仓]\n"
            f"原因: {reason}\n"
            f"价格: {price:.6f}\n"
            f"PnL: {pnl_pct:.4%}"
        )
        return True

    def _check_safety_gates(self, equity: float) -> bool:
        """检查 Kill Switch 和日亏损熔断。返回 True 表示应当拒绝新开仓。

        规则：
        - Kill Switch：只要 KILL_SWITCH_FILE 文件存在，立即停止所有新开仓。
          通过 ``touch kill_switch.flag`` 触发，``rm kill_switch.flag`` 恢复。
        - 日亏损熔断：当日权益亏损超过 MAX_DAILY_LOSS_PCT 时停止新开仓。
          每天 UTC 0:00 自动重置，平仓单不受影响。
        """
        from datetime import datetime, timezone as _tz
        today_utc = datetime.now(_tz.utc).strftime("%Y-%m-%d")

        # 跨天重置日起始权益
        if self._daily_start_date != today_utc:
            self._daily_start_equity = float(equity)
            self._daily_start_date = today_utc
            if self._trading_halted and "DailyLoss" in self._halt_reason:
                # 新的一天，自动解除日亏损熔断（Kill Switch 需要手动删文件）
                self._trading_halted = False
                self._halt_reason = ""
                log_info("🟢 日亏损熔断已在新交易日自动解除")

        # Kill Switch 文件检查
        kill_switch_path = os.path.join(BASE_DIR, config.KILL_SWITCH_FILE)
        if os.path.exists(kill_switch_path):
            if not self._trading_halted or "KillSwitch" not in self._halt_reason:
                self._trading_halted = True
                self._halt_reason = "KillSwitch"
                log_error(
                    f"🚨 Kill Switch 已激活（检测到 {kill_switch_path}）——拒绝所有新开仓。"
                    f" 删除该文件并重启进程可恢复。"
                )
                notify_important(
                    "🚨 Kill Switch 已激活\n"
                    "交易已暂停，删除 kill_switch.flag 并重启可恢复"
                )
            return True

        # 日亏损熔断
        max_loss_pct = float(getattr(config, "MAX_DAILY_LOSS_PCT", 0.05))
        if max_loss_pct > 0 and self._daily_start_equity and self._daily_start_equity > 0:
            loss_pct = (self._daily_start_equity - float(equity)) / self._daily_start_equity
            if loss_pct >= max_loss_pct:
                if not self._trading_halted or "DailyLoss" not in self._halt_reason:
                    self._trading_halted = True
                    self._halt_reason = f"DailyLoss({loss_pct:.2%})"
                    log_error(
                        f"🚨 日亏损熔断触发: 当日亏损 {loss_pct:.2%} ≥ 阈值 {max_loss_pct:.2%}"
                        f" (起始权益={self._daily_start_equity:.2f}, 当前={equity:.2f})"
                        f" ——今日拒绝新开仓，持仓的平仓/止损正常执行。"
                    )
                    notify_important(
                        f"🚨 日亏损熔断: {loss_pct:.2%}\n"
                        f"当日亏损超过 {max_loss_pct:.2%}，今日不再开新仓"
                    )
                return True

        if self._trading_halted:
            return True
        return False

    def run_once_on_new_bar(self):
        self.loop_count += 1
        bar_ts, price, long_prob, short_prob, money_flow_ratio, volatility, atr_ratio, trend_context, regime_context = self._get_latest_features()
        signal_snapshot = {
            "long_prob": float(long_prob),
            "short_prob": float(short_prob),
            "money_flow_ratio": float(money_flow_ratio),
            "volatility": float(volatility),
            "atr_ratio": None if atr_ratio is None else float(atr_ratio),
            **trend_context,
            **regime_context,
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
            f"mf={money_flow_ratio:.3f} vol={volatility:.6f} atr_ratio={0.0 if atr_ratio is None else atr_ratio:.4%} "
            f"trend={trend_context.get('trend_bias', 'neutral')} trend_gap={trend_context.get('trend_gap')} "
            f"regime={regime_context.get('regime')}"
        )

        pos_qty, entry_price = self._get_net_position()
        if pos_qty is None:
            log_error("双边持仓仍未清理完成，本轮跳过信号执行。")
            return
        # 本轮成功从 OKX 读取到仓位，清除上一轮可能遗留的不确定标志
        self._position_uncertain = False
        account_snapshot = self._get_account_snapshot()
        equity = self._get_sizing_equity(account_snapshot)

        # 安全门禁：Kill Switch / 日亏损熔断（只阻止新开仓，不影响平仓/止损）
        safety_halted = self._check_safety_gates(equity)

        if pos_qty == 0:
            self.hold_bars = 0
        self.core.set_state(
            pos_qty,
            entry_price,
            self.hold_bars,
            cooldown_bars_remaining=self.cooldown_bars_remaining,
            reverse_signal_bars=self.reverse_signal_bars,
            loss_guard_exit_bars=self.loss_guard_exit_bars,
        )

        out = self.core.on_bar(
            price=price,
            equity=equity,
            long_prob=long_prob,
            short_prob=short_prob,
            money_flow_ratio=money_flow_ratio,
            volatility=volatility,
            atr_ratio=atr_ratio,
            trend_bias=trend_context.get("trend_bias"),
            trend_gap=trend_context.get("trend_gap"),
            is_high_vol=bool(regime_context.get("is_high_vol")),
            market_regime=regime_context.get("regime"),
        )

        action = out["action"]
        delta = float(out["delta_qty"])
        account_before_trade = dict(account_snapshot)
        pos_qty_before_trade = float(pos_qty)
        entry_price_before_trade = float(entry_price)
        decision = {
            "action": action,
            "reason": out.get("reason"),
            "target_ratio": float(out.get("target_ratio", 0.0) or 0.0),
            "target_position": float(out.get("target_position", 0.0) or 0.0),
            "delta_qty": delta,
            "risk": out.get("risk"),
            "raw_target_ratio": float(out.get("raw_target_ratio", 0.0) or 0.0),
            "expected_net_edge": out.get("expected_net_edge"),
            "take_profit": out.get("take_profit"),
            "stop_loss": out.get("stop_loss"),
            "signal_prob_gap": out.get("signal_prob_gap"),
            "dominant_prob": out.get("dominant_prob"),
            "next_cooldown_bars": int(out.get("next_cooldown_bars", self.cooldown_bars_remaining)),
            "next_reverse_signal_bars": int(out.get("next_reverse_signal_bars", self.reverse_signal_bars)),
            "next_loss_guard_exit_bars": int(out.get("next_loss_guard_exit_bars", self.loss_guard_exit_bars)),
            "trend_bias": trend_context.get("trend_bias"),
            "trend_gap": trend_context.get("trend_gap"),
            "market_regime": regime_context.get("regime"),
            "regime_reason": regime_context.get("regime_reason"),
            "is_high_vol": bool(regime_context.get("is_high_vol")),
        }

        if action == "CLOSE":
            # 平仓前先撤销交易所端 TP/SL 算法单，防止残留单触发后反向开仓
            self._cancel_exchange_tpsl()
            success = False
            leverage = self._resolve_order_leverage(decision)
            if pos_qty > 0:
                success = self.client.close_long_sz(abs(pos_qty), leverage)
            elif pos_qty < 0:
                success = self.client.close_short_sz(abs(pos_qty), leverage)
            if success:
                self.cooldown_bars_remaining = int(out.get("next_cooldown_bars", self.cooldown_bars_remaining))
                self.reverse_signal_bars = int(out.get("next_reverse_signal_bars", 0))
                self.loss_guard_exit_bars = int(out.get("next_loss_guard_exit_bars", 0))
            if not self._sync_after_trade():
                raise RuntimeError("平仓后仓位状态无法确认")
            account_snapshot = self._get_account_snapshot()
            pos_qty, entry_price = self._get_net_position()
            trade_record = None
            if success:
                trade_record = self._record_trade_execution(
                    order_result=success,
                    action="CLOSE",
                    reason=out["reason"],
                    delta_qty=delta,
                    reference_price=price,
                    bar_ts=bar_ts,
                    signal_snapshot=signal_snapshot,
                    decision=decision,
                    account_before=account_before_trade,
                    pos_qty_before=pos_qty_before_trade,
                    entry_price_before=entry_price_before_trade,
                    account_after=account_snapshot,
                    pos_qty_after=pos_qty,
                    entry_price_after=entry_price,
                )
            if trade_record is None:
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
                self._notify_trade_execution(trade_record)
            else:
                log_error(f"平仓未成交，已重新同步仓位: reason={out['reason']}")
                self._notify_trade_failure("CLOSE", out["reason"], "平仓委托未确认成交，已重新同步仓位")
            self._persist_last_bar_state(bar_ts)
            return

        elif action == "OPEN":
            # 安全门禁：Kill Switch / 日亏损熔断触发时拒绝开仓，平仓/止损正常执行
            if safety_halted:
                log_info(
                    f"HOLD (安全门禁 {self._halt_reason}): 拒绝新开仓"
                    f" target_ratio={out.get('target_ratio', 0):.4f}"
                    f" dominant={out.get('dominant_prob', 0):.4f}"
                )
                self._persist_last_bar_state(bar_ts)
                return
            # 仓位不确定门禁：上一轮 _sync_after_trade 失败，内存状态不可信，拒绝新开仓
            if self._position_uncertain:
                log_error(
                    "HOLD (position_uncertain): 上一轮仓位同步失败，拒绝新开仓直到获得新鲜 OKX 仓位数据"
                )
                self._persist_last_bar_state(bar_ts)
                return
            success = self._execute_delta(pos_qty, delta, decision)
            if success:
                self.cooldown_bars_remaining = int(out.get("next_cooldown_bars", self.cooldown_bars_remaining))
                self.reverse_signal_bars = int(out.get("next_reverse_signal_bars", 0))
                self.loss_guard_exit_bars = int(out.get("next_loss_guard_exit_bars", 0))
            if not self._sync_after_trade():
                raise RuntimeError("开仓后仓位状态无法确认")
            account_snapshot = self._get_account_snapshot()
            pos_qty, entry_price = self._get_net_position()
            trade_record = None
            if success:
                # 开仓成功后立即在交易所下 TP/SL 算法单（P0修复）
                # entry_price 优先使用 OKX 返回的实际成交均价
                actual_entry = float(entry_price) if entry_price and float(entry_price) > 0 else float(price)
                pos_side_open = "long" if float(pos_qty) > 0 else "short"
                protected = self._place_exchange_tpsl(
                    pos_side=pos_side_open,
                    pos_qty=float(pos_qty) if pos_qty else abs(float(delta)),
                    entry_price=actual_entry,
                    decision=decision,
                )
                if not protected:
                    self._close_unprotected_position(pos_qty)
                    account_snapshot = self._get_account_snapshot()
                    pos_qty, entry_price = self._get_net_position()
                trade_record = self._record_trade_execution(
                    order_result=success,
                    action="OPEN",
                    reason=out["reason"],
                    delta_qty=delta,
                    reference_price=price,
                    bar_ts=bar_ts,
                    signal_snapshot=signal_snapshot,
                    decision=decision,
                    account_before=account_before_trade,
                    pos_qty_before=pos_qty_before_trade,
                    entry_price_before=entry_price_before_trade,
                    account_after=account_snapshot,
                    pos_qty_after=pos_qty,
                    entry_price_after=entry_price,
                )
            if trade_record is None:
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
                self._notify_trade_execution(trade_record)
            else:
                log_error(f"开仓未成交，已重新同步仓位: target_ratio={out['target_ratio']:.3f}, qty={abs(delta):.6f}")
                self._notify_trade_failure(
                    "OPEN",
                    out["reason"],
                    f"target_ratio={out['target_ratio']:.3f}, qty={abs(delta):.6f}",
                )
            self._persist_last_bar_state(bar_ts)
            return

        elif action == "REBALANCE":
            # 仓位不确定时拒绝加仓方向的 REBALANCE（减仓仍允许）
            if self._position_uncertain and delta > 0 and pos_qty >= 0:
                log_error("HOLD (position_uncertain): 上一轮仓位同步失败，拒绝 REBALANCE 加仓")
                self._persist_last_bar_state(bar_ts)
                return
            if self._position_uncertain and delta < 0 and pos_qty <= 0:
                log_error("HOLD (position_uncertain): 上一轮仓位同步失败，拒绝 REBALANCE 加仓")
                self._persist_last_bar_state(bar_ts)
                return
            success = self._execute_delta(pos_qty, delta, decision)
            if success:
                self.cooldown_bars_remaining = int(out.get("next_cooldown_bars", self.cooldown_bars_remaining))
                self.reverse_signal_bars = int(out.get("next_reverse_signal_bars", self.reverse_signal_bars))
                self.loss_guard_exit_bars = int(out.get("next_loss_guard_exit_bars", self.loss_guard_exit_bars))
            if not self._sync_after_trade():
                raise RuntimeError("调仓后仓位状态无法确认")
            account_snapshot = self._get_account_snapshot()
            pos_qty, entry_price = self._get_net_position()
            trade_record = None
            if success:
                protected = self._replace_exchange_tpsl(pos_qty, entry_price, decision)
                if not protected:
                    self._close_unprotected_position(pos_qty)
                    account_snapshot = self._get_account_snapshot()
                    pos_qty, entry_price = self._get_net_position()
                trade_record = self._record_trade_execution(
                    order_result=success,
                    action="REBALANCE",
                    reason=out["reason"],
                    delta_qty=delta,
                    reference_price=price,
                    bar_ts=bar_ts,
                    signal_snapshot=signal_snapshot,
                    decision=decision,
                    account_before=account_before_trade,
                    pos_qty_before=pos_qty_before_trade,
                    entry_price_before=entry_price_before_trade,
                    account_after=account_snapshot,
                    pos_qty_after=pos_qty,
                    entry_price_after=entry_price,
                )
            if trade_record is None:
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
                self._notify_trade_execution(trade_record)
            else:
                log_error(f"调仓未成交，已重新同步仓位: delta_qty={delta:.6f}, reason={out['reason']}")
                self._notify_trade_failure(
                    "REBALANCE",
                    out["reason"],
                    f"delta_qty={delta:.6f}",
                )
            self._persist_last_bar_state(bar_ts)
            return

        elif action == "HOLD":
            self.hold_bars = int(out.get("next_hold_bars", self.hold_bars))
            self.cooldown_bars_remaining = int(out.get("next_cooldown_bars", self.cooldown_bars_remaining))
            self.reverse_signal_bars = int(out.get("next_reverse_signal_bars", self.reverse_signal_bars))
            self.loss_guard_exit_bars = int(out.get("next_loss_guard_exit_bars", self.loss_guard_exit_bars))
            self.core.set_state(
                pos_qty,
                entry_price,
                self.hold_bars,
                cooldown_bars_remaining=self.cooldown_bars_remaining,
                reverse_signal_bars=self.reverse_signal_bars,
                loss_guard_exit_bars=self.loss_guard_exit_bars,
            )
            position_snapshot = self._build_position_snapshot(pos_qty, entry_price, current_price=price, pending_orders=0)
            self._write_dashboard_snapshot(
                runtime_status="running",
                latest_closed_bar_ts=bar_ts,
                current_price=price,
                signal_snapshot=signal_snapshot,
                account_snapshot=account_snapshot,
                position_snapshot=position_snapshot,
                decision=decision,
            )
            self._maybe_notify_abnormal_hold(
                bar_ts=bar_ts,
                price=price,
                equity=equity,
                decision=decision,
                signal_snapshot=signal_snapshot,
            )
            self._maybe_notify_runtime_summary(
                bar_ts=bar_ts,
                price=price,
                equity=equity,
                position_snapshot=position_snapshot,
                signal_snapshot=signal_snapshot,
                decision=decision,
            )
            log_info(self._format_hold_decision_log(
                decision,
                out,
                price=price,
                equity=equity,
                pos_qty=pos_qty,
                entry_price=entry_price,
            ))
            self._persist_last_bar_state(bar_ts)
            return


def run():
    import atexit
    POLL_SEC = max(1, int(config.POLL_SEC))
    BAR_POLL_SEC = max(POLL_SEC, int(getattr(config, "BAR_POLL_SEC", 10)))

    # ===== 多实例防护：PID 锁文件 =====
    pid_file = os.path.join(LOGS_DIR, "live_trading_monitor.pid")
    if os.path.exists(pid_file):
        try:
            old_pid = int(open(pid_file).read().strip())
            # 检查旧进程是否仍在运行
            try:
                os.kill(old_pid, 0)   # 信号0：不发送信号，只检查进程存在
                raise RuntimeError(
                    f"检测到另一个 live_trading_monitor 进程正在运行 (PID={old_pid})。"
                    f" 若确认旧进程已停止，请手动删除 {pid_file} 后重试。"
                )
            except ProcessLookupError:
                log_info(f"旧 PID 文件残留（进程 {old_pid} 已不存在），覆盖重建。")
        except (ValueError, OSError):
            pass  # PID 文件损坏，直接覆盖
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(pid_file) and os.remove(pid_file))

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

    log_info(
        "🟢 Live trading monitor started "
        f"(risk_poll_sec={POLL_SEC}, bar_poll_sec={BAR_POLL_SEC})"
    )
    next_bar_poll_at = 0.0
    while True:
        try:
            risk_action_executed = trader.run_realtime_risk_check()
            now = time.monotonic()
            if not risk_action_executed and now >= next_bar_poll_at:
                trader.run_once_on_new_bar()
                next_bar_poll_at = now + BAR_POLL_SEC
            trader.consecutive_loop_errors = 0
            trader.last_error_notified_count = 0
        except Exception as e:
            trader.consecutive_loop_errors += 1
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
            trader._notify_consecutive_loop_error(e)

        time.sleep(int(POLL_SEC))


if __name__ == "__main__":
    run()
