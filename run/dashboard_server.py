import csv
import glob
import json
import math
import os
import re
import shlex
import shutil
import subprocess
from collections import deque
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from config import config
from dotenv import set_key
from utils.runtime_dashboard import (
    load_runtime_dashboard_history,
    load_runtime_dashboard_status,
)
from utils.utils import BASE_DIR, LOG_FILE, LOGS_DIR


DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8787"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_HISTORY_LIMIT = int(os.getenv("DASHBOARD_HISTORY_LIMIT", "240"))
DASHBOARD_EVENT_LIMIT = int(os.getenv("DASHBOARD_EVENT_LIMIT", "24"))
DASHBOARD_TRADE_LIMIT = int(os.getenv("DASHBOARD_TRADE_LIMIT", "18"))
DASHBOARD_LOG_TAIL_LINES = int(os.getenv("DASHBOARD_LOG_TAIL_LINES", "18000"))
ENV_FILE_PATH = os.getenv("ENV_FILE", os.path.join(BASE_DIR, ".env"))

FRONTEND_ROOT = os.path.join(BASE_DIR, "dashboard-ui")
FRONTEND_DIST_ROOT = os.path.join(FRONTEND_ROOT, "dist")

EVENT_PATTERN = re.compile(
    r"交易环境校验完成|paper_ready_ok|Live trading monitor started|"
    r"已恢复最近处理 bar|新bar=|心跳:|执行开仓|执行平仓|执行调仓|"
    r"无明显信号或目标为0|实盘循环异常|未成交|同时多空持仓"
)
EXECUTION_PATTERN = re.compile(r"执行开仓|执行平仓|执行调仓|未成交")
LOG_LINE_PATTERN = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - "
    r"(?P<level>[A-Z]+) - (?P<message>.*)$"
)
NEW_BAR_PATTERN = re.compile(
    r"新bar=(?P<bar_ts>[\d\-:\s]+)\s+price=(?P<price>-?\d+(?:\.\d+)?)\s+"
    r"long=(?P<long>-?\d+(?:\.\d+)?)\s+short=(?P<short>-?\d+(?:\.\d+)?)\s+"
    r"mf=(?P<mf>-?\d+(?:\.\d+)?)\s+vol=(?P<vol>-?\d+(?:\.\d+)?)\s+"
    r"atr_ratio=(?P<atr>-?\d+(?:\.\d+)?)%"
)
RESTORED_BAR_PATTERN = re.compile(r"已恢复最近处理 bar:\s*(?P<bar_ts>[\d\-:\s]+)")
HEARTBEAT_PATTERN = re.compile(
    r"最近已处理bar=(?P<processed>[^,]+),\s*当前最新已收盘bar=(?P<latest>[^,]+),\s*连续跳过同bar次数=(?P<skip>\d+)"
)
VALUE_WITH_UNIT_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")
PERCENT_IN_PARENS_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?)%\)")
REASON_PATTERN = re.compile(r"reason=([A-Za-z0-9_]+)")
DELTA_QTY_PATTERN = re.compile(r"delta_qty=(-?\d+(?:\.\d+)?)")
QTY_PATTERN = re.compile(r"qty=(-?\d+(?:\.\d+)?)")
BACKTEST_GLOB = os.path.join(LOGS_DIR, "backtest_multi_period_*.csv")


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0)


def utc_now_iso():
    return utc_now().isoformat()


def canonicalize_timeframe(value):
    text = str(value or "").strip()
    match = re.match(r"^(\d+)\s*([mhdMHD])$", text)
    if not match:
        raise ValueError("timeframe must look like 5m / 15m / 1H / 4H")

    count = match.group(1)
    unit = match.group(2).lower()
    if unit == "m":
        return f"{count}m"
    if unit == "h":
        return f"{count}H"
    return f"{count}D"


def safe_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def parse_log_line(line):
    match = LOG_LINE_PATTERN.match((line or "").strip())
    if not match:
        return None
    return match.groupdict()


def log_ts_to_iso(value):
    if not value:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc).isoformat()


def normalize_bar_ts(value):
    if not value:
        return None
    text = str(value).strip()
    if text in {"None", "null", ""}:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return None


def strip_log_prefix(line):
    parsed = parse_log_line(line)
    if not parsed:
        return line
    return parsed["message"]


def read_log_tail_lines(log_path, max_lines=DASHBOARD_LOG_TAIL_LINES):
    if not os.path.exists(log_path):
        return []

    lines = deque(maxlen=max_lines)
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                lines.append(line.rstrip())
    except Exception:
        return []

    return list(lines)


def extract_recent_strategy_events(log_lines, limit=DASHBOARD_EVENT_LIMIT):
    events = [line for line in log_lines if EVENT_PATTERN.search(line)]
    return events[-limit:]


def infer_signal_direction(long_prob, short_prob):
    long_prob = safe_float(long_prob)
    short_prob = safe_float(short_prob)
    if long_prob is None or short_prob is None:
        return "Neutral"
    if long_prob - short_prob > 0.06:
        return "Long"
    if short_prob - long_prob > 0.06:
        return "Short"
    return "Neutral"


def derive_risk_level(exposure_pct, leverage, runtime_status):
    exposure_pct = safe_float(exposure_pct) or 0.0
    leverage = safe_float(leverage) or 0.0
    if runtime_status == "error" or exposure_pct >= 70 or leverage >= 5:
        return "High"
    if exposure_pct >= 35 or leverage >= 3:
        return "Medium"
    return "Low"


def maybe_parse_latest_new_bar(log_lines):
    for line in reversed(log_lines):
        parsed = parse_log_line(line)
        if not parsed:
            continue
        match = NEW_BAR_PATTERN.search(parsed["message"])
        if not match:
            continue
        groups = match.groupdict()
        return {
            "bar_ts": normalize_bar_ts(groups.get("bar_ts")),
            "last_price": safe_float(groups.get("price")),
            "long_prob": safe_float(groups.get("long")),
            "short_prob": safe_float(groups.get("short")),
            "money_flow_ratio": safe_float(groups.get("mf")),
            "volatility": safe_float(groups.get("vol")),
            "atr_ratio": None if safe_float(groups.get("atr")) is None else safe_float(groups.get("atr")) / 100.0,
        }
    return {}


def maybe_parse_bar_progress(log_lines):
    payload = {}
    for line in reversed(log_lines):
        parsed = parse_log_line(line)
        if not parsed:
            continue
        message = parsed["message"]
        restore_match = RESTORED_BAR_PATTERN.search(message)
        if restore_match and not payload.get("last_processed_bar_ts"):
            payload["last_processed_bar_ts"] = normalize_bar_ts(restore_match.group("bar_ts"))
        heartbeat_match = HEARTBEAT_PATTERN.search(message)
        if heartbeat_match:
            payload["last_processed_bar_ts"] = payload.get("last_processed_bar_ts") or normalize_bar_ts(
                heartbeat_match.group("processed")
            )
            payload["latest_closed_bar_ts"] = payload.get("latest_closed_bar_ts") or normalize_bar_ts(
                heartbeat_match.group("latest")
            )
            payload["same_bar_skip_count"] = safe_int(heartbeat_match.group("skip"), 0)
        if payload.get("last_processed_bar_ts") and payload.get("latest_closed_bar_ts"):
            return payload
    return payload


def enrich_status_with_fallbacks(status, log_lines, recent_events):
    status = dict(status or {})
    runtime = dict(status.get("runtime") or {})
    market = dict(status.get("market") or {})
    bar = dict(status.get("bar") or {})
    signal = dict(status.get("signal") or {})
    account = dict(status.get("account") or {})
    position = dict(status.get("position") or {})
    decision = dict(status.get("decision") or {})
    last_execution = dict(status.get("last_execution") or {})

    latest_error = None
    fallback_error = None
    last_event_iso = None
    for line in reversed(log_lines):
        parsed = parse_log_line(line)
        if not parsed:
            continue
        if last_event_iso is None:
            last_event_iso = log_ts_to_iso(parsed["ts"])
        if parsed["level"] == "ERROR":
            message = parsed["message"]
            if fallback_error is None:
                fallback_error = message
            if not message.startswith("Traceback"):
                latest_error = message
                break
    latest_error = latest_error or fallback_error

    new_bar = maybe_parse_latest_new_bar(log_lines)
    progress = maybe_parse_bar_progress(recent_events or log_lines)

    if new_bar:
        market["last_price"] = market.get("last_price") if safe_float(market.get("last_price")) is not None else new_bar["last_price"]
        signal["long_prob"] = signal.get("long_prob") if safe_float(signal.get("long_prob")) is not None else new_bar["long_prob"]
        signal["short_prob"] = signal.get("short_prob") if safe_float(signal.get("short_prob")) is not None else new_bar["short_prob"]
        signal["money_flow_ratio"] = (
            signal.get("money_flow_ratio")
            if safe_float(signal.get("money_flow_ratio")) is not None
            else new_bar["money_flow_ratio"]
        )
        signal["volatility"] = signal.get("volatility") if safe_float(signal.get("volatility")) is not None else new_bar["volatility"]
        signal["atr_ratio"] = signal.get("atr_ratio") if safe_float(signal.get("atr_ratio")) is not None else new_bar["atr_ratio"]
        bar["latest_closed_bar_ts"] = bar.get("latest_closed_bar_ts") or new_bar["bar_ts"]

    if progress:
        bar["last_processed_bar_ts"] = bar.get("last_processed_bar_ts") or progress.get("last_processed_bar_ts")
        bar["latest_closed_bar_ts"] = bar.get("latest_closed_bar_ts") or progress.get("latest_closed_bar_ts")
        if runtime.get("same_bar_skip_count") is None and progress.get("same_bar_skip_count") is not None:
            runtime["same_bar_skip_count"] = progress.get("same_bar_skip_count")

    market.setdefault("exchange", "OKX")
    market.setdefault("symbol", getattr(config, "SYMBOL", "SOL-USDT-SWAP"))
    market.setdefault("leverage", safe_float(getattr(config, "LEVERAGE", 0)))
    market.setdefault("simulated", str(getattr(config, "USE_SERVER", "1")) == "1")

    runtime.setdefault("poll_sec", safe_int(getattr(config, "POLL_SEC", 10), 10))
    runtime.setdefault("heartbeat_interval_sec", 30.0)
    runtime["last_error"] = runtime.get("last_error") or latest_error
    if not runtime.get("last_status"):
        runtime["last_status"] = "error" if latest_error else ("waiting_next_bar" if bar.get("latest_closed_bar_ts") else "starting")

    status["updated_at"] = status.get("updated_at") or last_event_iso or utc_now_iso()
    status["runtime"] = runtime
    status["market"] = market
    status["bar"] = bar
    status["signal"] = signal
    status["account"] = account
    status["position"] = position
    status["decision"] = decision
    status["last_execution"] = last_execution
    return status


def format_env_number(value, *, digits=6, as_int=False):
    if as_int:
        return str(int(round(float(value))))
    numeric = float(value)
    text = f"{numeric:.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def serialize_windows(windows_dict):
    items = []
    for key, value in (windows_dict or {}).items():
        if value is None:
            continue
        items.append(f"{key}:{int(value)}")
    return ",".join(items)


def ensure_env_file_exists(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "a", encoding="utf-8"):
            pass


def persist_env_key(path, key, value):
    ensure_env_file_exists(path)
    set_key(path, key, str(value), quote_mode="never")


def update_config_attr(key, value):
    setattr(config, key, value)
    if isinstance(value, list):
        os.environ[key] = ",".join(str(item) for item in value)
        return
    if isinstance(value, dict):
        os.environ[key] = serialize_windows(value)
        return
    os.environ[key] = str(value)


def build_intervals_with_primary(primary_timeframe):
    existing = [str(item) for item in (getattr(config, "INTERVALS", []) or []) if item]
    canonical_primary = canonicalize_timeframe(primary_timeframe)
    ordered = [canonical_primary]
    ordered.extend(item for item in existing if item != canonical_primary)
    return ordered


def build_windows_with_primary(primary_timeframe):
    canonical_primary = canonicalize_timeframe(primary_timeframe)
    current_windows = dict(getattr(config, "WINDOWS", {}) or {})
    current_intervals = list(getattr(config, "INTERVALS", []) or [])
    old_primary = current_intervals[0] if current_intervals else None
    fallback_window = None
    if old_primary:
        fallback_window = current_windows.get(old_primary)
    if fallback_window is None:
        fallback_window = current_windows.get(canonical_primary)
    if fallback_window is None:
        fallback_window = 2000
    current_windows.setdefault(canonical_primary, int(fallback_window))
    return current_windows


def validate_strategy_params_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("strategy params payload must be a JSON object")

    timeframe = canonicalize_timeframe(payload.get("timeframe"))
    ma_period = safe_int(payload.get("maPeriod"), 0)
    rsi_period = safe_int(payload.get("rsiPeriod"), 0)
    atr_multiplier = safe_float(payload.get("atrMultiplier"))
    stop_loss_pct = safe_float(payload.get("stopLossPct"))
    take_profit_pct = safe_float(payload.get("takeProfitPct"))
    position_size = safe_float(payload.get("positionSizePct"))
    max_leverage = safe_float(payload.get("maxLeverage"))

    errors = []
    if ma_period <= 0:
        errors.append("maPeriod must be > 0")
    if rsi_period <= 0:
        errors.append("rsiPeriod must be > 0")
    if atr_multiplier is None or atr_multiplier <= 0:
        errors.append("atrMultiplier must be > 0")
    if stop_loss_pct is None or stop_loss_pct <= 0 or stop_loss_pct >= 100:
        errors.append("stopLossPct must be between 0 and 100")
    if take_profit_pct is None or take_profit_pct <= 0 or take_profit_pct >= 100:
        errors.append("takeProfitPct must be between 0 and 100")
    if position_size is None or position_size <= 0:
        errors.append("positionSizePct must be > 0")
    if max_leverage is None or max_leverage <= 0:
        errors.append("maxLeverage must be > 0")
    if errors:
        raise ValueError("; ".join(errors))

    intervals = build_intervals_with_primary(timeframe)
    windows = build_windows_with_primary(timeframe)

    return {
        "timeframe": timeframe,
        "ma_period": int(ma_period),
        "rsi_period": int(rsi_period),
        "atr_stop_loss_multiplier": float(atr_multiplier),
        "stop_loss": float(stop_loss_pct) / 100.0,
        "take_profit": float(take_profit_pct) / 100.0,
        "position_size": float(position_size),
        "leverage": int(round(float(max_leverage))),
        "intervals": intervals,
        "windows": windows,
    }


def save_strategy_params(payload, env_path=ENV_FILE_PATH):
    normalized = validate_strategy_params_payload(payload)

    persist_env_key(env_path, "INTERVALS", ",".join(normalized["intervals"]))
    persist_env_key(env_path, "WINDOWS", serialize_windows(normalized["windows"]))
    persist_env_key(env_path, "MA_PERIOD", format_env_number(normalized["ma_period"], as_int=True))
    persist_env_key(env_path, "RSI_PERIOD", format_env_number(normalized["rsi_period"], as_int=True))
    persist_env_key(env_path, "ATR_STOP_LOSS_MULTIPLIER", format_env_number(normalized["atr_stop_loss_multiplier"]))
    persist_env_key(env_path, "STOP_LOSS", format_env_number(normalized["stop_loss"]))
    persist_env_key(env_path, "TAKE_PROFIT", format_env_number(normalized["take_profit"]))
    persist_env_key(env_path, "POSITION_SIZE", format_env_number(normalized["position_size"]))
    persist_env_key(env_path, "LEVERAGE", format_env_number(normalized["leverage"], as_int=True))

    update_config_attr("INTERVALS", normalized["intervals"])
    update_config_attr("WINDOWS", normalized["windows"])
    update_config_attr("MA_PERIOD", normalized["ma_period"])
    update_config_attr("RSI_PERIOD", normalized["rsi_period"])
    update_config_attr("ATR_STOP_LOSS_MULTIPLIER", normalized["atr_stop_loss_multiplier"])
    update_config_attr("STOP_LOSS", normalized["stop_loss"])
    update_config_attr("TAKE_PROFIT", normalized["take_profit"])
    update_config_attr("POSITION_SIZE", normalized["position_size"])
    update_config_attr("LEVERAGE", normalized["leverage"])

    return {
        "ok": True,
        "saved_at": utc_now_iso(),
        "env_path": env_path,
        "restart_required": True,
        "message": (
            "已写入 .env 并刷新 Dashboard 配置视图。"
            "量化交易主进程如为独立 PM2/daemon，需要重启后才会真正应用新参数。"
        ),
        "saved_params": build_strategy_params(),
        "bundle": build_dashboard_bundle(),
    }


def get_restart_command_text():
    return str(os.getenv("DASHBOARD_STRATEGY_RESTART_CMD", "") or "").strip()


def get_restart_pm2_app_name():
    return str(os.getenv("DASHBOARD_STRATEGY_PM2_APP", "quant_okx_paper") or "").strip() or "quant_okx_paper"


def get_restart_timeout_sec():
    return max(1, safe_int(os.getenv("DASHBOARD_STRATEGY_RESTART_TIMEOUT_SEC", "25"), 25))


def resolve_restart_strategy_command():
    custom_command = get_restart_command_text()
    if custom_command:
        try:
            parsed = shlex.split(custom_command)
        except ValueError as exc:
            raise RuntimeError(f"invalid DASHBOARD_STRATEGY_RESTART_CMD: {exc}") from exc
        if not parsed:
            raise RuntimeError("DASHBOARD_STRATEGY_RESTART_CMD is empty after parsing")
        return parsed, "custom"

    pm2_path = shutil.which("pm2")
    if pm2_path:
        return [pm2_path, "restart", get_restart_pm2_app_name(), "--update-env"], "pm2"

    raise RuntimeError(
        "pm2 not found. Install pm2 or set DASHBOARD_STRATEGY_RESTART_CMD to a restartable command."
    )


def summarize_subprocess_output(completed):
    parts = []
    stdout_text = str(getattr(completed, "stdout", "") or "").strip()
    stderr_text = str(getattr(completed, "stderr", "") or "").strip()
    if stdout_text:
        parts.append(stdout_text)
    if stderr_text:
        parts.append(stderr_text)
    if not parts:
        return ""
    summary = " | ".join(parts)
    if len(summary) > 800:
        summary = f"{summary[:797]}..."
    return summary


def restart_strategy_process():
    command, command_mode = resolve_restart_strategy_command()
    timeout_sec = get_restart_timeout_sec()

    try:
        completed = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"restart command timed out after {timeout_sec}s") from exc

    output_summary = summarize_subprocess_output(completed)
    if completed.returncode != 0:
        if command_mode == "pm2":
            raise RuntimeError(
                f"pm2 restart failed for {get_restart_pm2_app_name()}: {output_summary or 'no output'}"
            )
        raise RuntimeError(f"restart command failed: {output_summary or 'no output'}")

    if command_mode == "pm2":
        message = (
            f"已向 PM2 发送重启指令: {get_restart_pm2_app_name()}。"
            "策略进程会在数秒内重新拉起。"
        )
    else:
        message = "已执行自定义策略重启命令。"

    return {
        "ok": True,
        "restarted_at": utc_now_iso(),
        "command_mode": command_mode,
        "command": command,
        "output": output_summary or None,
        "message": message,
        "bundle": build_dashboard_bundle(),
    }


def build_strategy_meta(status):
    market = status.get("market") or {}
    symbol = market.get("symbol") or getattr(config, "SYMBOL", "SOL-USDT-SWAP")
    intervals = list(getattr(config, "INTERVALS", []) or [])
    mode = "Paper" if str(getattr(config, "USE_SERVER", "1")) == "1" else "Live"
    interval_label = " / ".join(intervals) if intervals else "Live"
    return {
        "product_name": "Quant Alpha Dashboard",
        "strategy_name": f"{symbol} {interval_label} {mode} Strategy",
        "exchange": market.get("exchange") or "OKX",
        "symbol": symbol,
        "mode": mode.lower(),
        "simulated": bool(market.get("simulated", str(getattr(config, "USE_SERVER", "1")) == "1")),
        "intervals": intervals,
    }


def build_strategy_params():
    intervals = list(getattr(config, "INTERVALS", []) or [])
    primary_interval = intervals[0] if intervals else "5m"

    atr_multiplier = safe_float(getattr(config, "ATR_STOP_LOSS_MULTIPLIER", None))
    if atr_multiplier is None:
        atr_multiplier = safe_float(getattr(config, "ATR_TAKE_PROFIT_MULTIPLIER", None))

    return {
        "timeframe": primary_interval,
        "intervals": intervals,
        "ma_period": safe_int(getattr(config, "MA_PERIOD", 34), 34),
        "rsi_period": safe_int(getattr(config, "RSI_PERIOD", 14), 14),
        "atr_multiplier": atr_multiplier,
        "stop_loss_pct": None if safe_float(getattr(config, "STOP_LOSS", None)) is None else float(config.STOP_LOSS) * 100.0,
        "take_profit_pct": None if safe_float(getattr(config, "TAKE_PROFIT", None)) is None else float(config.TAKE_PROFIT) * 100.0,
        "position_size_pct": safe_float(getattr(config, "POSITION_SIZE", None)),
        "max_leverage": safe_float(getattr(config, "LEVERAGE", None)),
        "adaptive_tp_sl_enabled": bool(getattr(config, "ADAPTIVE_TP_SL_ENABLED", False)),
        "threshold_long": safe_float(getattr(config, "THRESHOLD_LONG", None)),
        "threshold_short": safe_float(getattr(config, "THRESHOLD_SHORT", None)),
        "atr_take_profit_multiplier": safe_float(getattr(config, "ATR_TAKE_PROFIT_MULTIPLIER", None)),
        "atr_stop_loss_multiplier": safe_float(getattr(config, "ATR_STOP_LOSS_MULTIPLIER", None)),
        "volatility_take_profit_multiplier": safe_float(getattr(config, "VOLATILITY_TAKE_PROFIT_MULTIPLIER", None)),
        "volatility_stop_loss_multiplier": safe_float(getattr(config, "VOLATILITY_STOP_LOSS_MULTIPLIER", None)),
    }


def build_signal_summary(status):
    runtime = status.get("runtime") or {}
    signal = status.get("signal") or {}
    last_execution = status.get("last_execution") or {}
    bar = status.get("bar") or {}

    long_prob = safe_float(signal.get("long_prob"))
    short_prob = safe_float(signal.get("short_prob"))
    sources = ["ML Model"]
    if safe_float(signal.get("money_flow_ratio")) is not None:
        sources.append("RSI")
    if safe_float(signal.get("volatility")) is not None:
        sources.append("ATR")
    if safe_float(signal.get("atr_ratio")) is not None:
        sources.append("MACD")

    next_run_at = None
    poll_sec = safe_int(runtime.get("poll_sec"), 10)
    if poll_sec > 0:
        next_run_at = (utc_now() + timedelta(seconds=poll_sec)).isoformat()

    score = None
    if long_prob is not None and short_prob is not None:
        score = round(max(long_prob, short_prob) * 100)

    return {
        "direction": infer_signal_direction(long_prob, short_prob),
        "sources": list(dict.fromkeys(sources)),
        "score": score,
        "last_triggered_at": last_execution.get("timestamp") or bar.get("latest_closed_bar_ts") or status.get("updated_at"),
        "next_run_at": next_run_at,
    }


def compute_daily_pnl(history):
    normalized = []
    for point in history or []:
        ts = point.get("bar_ts") or point.get("timestamp")
        total_eq = safe_float(point.get("total_eq"))
        if not ts or total_eq is None:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        normalized.append((dt, total_eq))

    if not normalized:
        return None

    latest_dt, latest_eq = normalized[-1]
    anchor_eq = normalized[0][1]
    threshold = latest_dt - timedelta(days=1)
    for dt, eq in normalized:
        if dt >= threshold:
            anchor_eq = eq
            break
    return latest_eq - anchor_eq


def parse_latest_backtest_summary(log_lines):
    latest_idx = None
    for idx, line in enumerate(log_lines):
        if "回测完成" in line:
            latest_idx = idx
    if latest_idx is None:
        return {}

    summary = {}
    relevant = log_lines[latest_idx: latest_idx + 12]
    for line in relevant:
        message = strip_log_prefix(line)
        timestamp = None
        parsed = parse_log_line(line)
        if parsed:
            timestamp = log_ts_to_iso(parsed["ts"])
            summary["timestamp"] = summary.get("timestamp") or timestamp

        if "期末净值" in message:
            summary["final_equity"] = safe_float(VALUE_WITH_UNIT_PATTERN.search(message).group(1)) if VALUE_WITH_UNIT_PATTERN.search(message) else None
        elif "累计收益" in message:
            value_match = VALUE_WITH_UNIT_PATTERN.search(message)
            pct_match = PERCENT_IN_PARENS_PATTERN.search(message)
            summary["pnl"] = safe_float(value_match.group(1)) if value_match else None
            summary["return_pct"] = safe_float(pct_match.group(1)) if pct_match else None
        elif "最大回撤" in message:
            percent_match = VALUE_WITH_UNIT_PATTERN.search(message)
            summary["max_drawdown_pct"] = safe_float(percent_match.group(1)) if percent_match else None
        elif "交易次数" in message:
            trade_match = VALUE_WITH_UNIT_PATTERN.search(message)
            summary["trade_count"] = safe_int(trade_match.group(1), 0) if trade_match else None
        elif "手续费合计" in message:
            fee_match = VALUE_WITH_UNIT_PATTERN.search(message)
            summary["fees_paid"] = safe_float(fee_match.group(1)) if fee_match else None
        elif "滑点成本合计" in message:
            slip_match = VALUE_WITH_UNIT_PATTERN.search(message)
            summary["slippage_cost"] = safe_float(slip_match.group(1)) if slip_match else None
    return summary


def parse_latest_backtest_csv_metrics():
    files = glob.glob(BACKTEST_GLOB)
    if not files:
        return {}

    latest_path = max(files, key=os.path.getmtime)
    balances = []
    timestamps = []
    active_trade_start_balance = None
    wins = 0
    losses = 0
    close_actions = {"止盈", "止损", "平仓", "反向平仓"}
    prev_balance = None

    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                balance = safe_float(row.get("balance"))
                timestamp = row.get("timestamp")
                action = str(row.get("action") or "")

                if balance is not None:
                    balances.append(balance)
                if timestamp:
                    timestamps.append(timestamp)

                if action.startswith("开") and active_trade_start_balance is None:
                    active_trade_start_balance = prev_balance if prev_balance is not None else balance
                elif action in close_actions:
                    start_balance = active_trade_start_balance if active_trade_start_balance is not None else prev_balance
                    if start_balance is not None and balance is not None:
                        trade_pnl = balance - start_balance
                        if trade_pnl > 0:
                            wins += 1
                        elif trade_pnl < 0:
                            losses += 1
                    active_trade_start_balance = None

                if balance is not None:
                    prev_balance = balance
    except Exception:
        return {}

    returns = []
    for idx in range(1, len(balances)):
        prev = balances[idx - 1]
        curr = balances[idx]
        if prev and prev > 0:
            returns.append((curr - prev) / prev)

    sharpe_ratio = None
    if len(returns) >= 2:
        mean_ret = sum(returns) / len(returns)
        variance = sum((item - mean_ret) ** 2 for item in returns) / (len(returns) - 1)
        std_ret = math.sqrt(max(variance, 0.0))
        if std_ret > 0:
            sharpe_ratio = mean_ret / std_ret * math.sqrt(len(returns))

    total_closed = wins + losses
    return {
        "source_path": latest_path,
        "period_start": timestamps[0] if timestamps else None,
        "period_end": timestamps[-1] if timestamps else None,
        "win_rate_pct": (wins / total_closed * 100.0) if total_closed > 0 else None,
        "sharpe_ratio": sharpe_ratio,
        "trade_count_closed": total_closed,
    }


def build_risk_snapshot(status, history, recent_events):
    runtime = status.get("runtime") or {}
    market = status.get("market") or {}
    account = status.get("account") or {}
    position = status.get("position") or {}

    total_eq = safe_float(account.get("total_eq")) or safe_float((status.get("performance") or {}).get("current_total_eq"))
    notional = safe_float(position.get("notional"))
    leverage = safe_float(market.get("leverage")) or safe_float(getattr(config, "LEVERAGE", 0)) or 0.0
    margin_usage_pct = 0.0
    if total_eq and total_eq > 0 and notional is not None:
        margin_usage_pct = max(0.0, notional / total_eq * 100.0)

    daily_pnl = compute_daily_pnl(history)
    daily_loss_used_pct = 0.0
    if daily_pnl is not None and daily_pnl < 0 and total_eq and total_eq > 0:
        daily_loss_used_pct = abs(daily_pnl) / total_eq * 100.0

    latest_error = runtime.get("last_error")
    if not latest_error:
        for line in reversed(recent_events):
            parsed = parse_log_line(line)
            if parsed and parsed["level"] == "ERROR":
                latest_error = parsed["message"]
                break

    runtime_status = str(runtime.get("last_status") or "").lower()
    api_status = "Connected"
    ws_status = "Connected"
    if runtime_status == "error":
        api_status = "Disconnected" if latest_error else "Degraded"
        ws_status = "Disconnected" if latest_error else "Lagging"
    elif runtime_status in {"starting", "paused"}:
        api_status = "Degraded"
        ws_status = "Lagging"

    risk_triggered = bool(latest_error) or runtime_status == "error"
    return {
        "current_leverage": leverage,
        "max_loss_per_trade_pct": None if safe_float(getattr(config, "STOP_LOSS", None)) is None else float(config.STOP_LOSS) * 100.0,
        "margin_usage_pct": margin_usage_pct,
        "daily_loss_limit_pct": None,
        "daily_loss_used_pct": daily_loss_used_pct,
        "risk_triggered": risk_triggered,
        "risk_level": derive_risk_level(margin_usage_pct, leverage, runtime_status),
        "api_status": api_status,
        "ws_status": ws_status,
        "latest_error": latest_error,
    }


def infer_trade_status(message):
    if "未成交" in message:
        return "Canceled"
    if "止盈" in message:
        return "Take Profit"
    if "止损" in message:
        return "Stopped"
    return "Filled"


def infer_trade_side(message, fallback_direction):
    delta_match = DELTA_QTY_PATTERN.search(message)
    if delta_match:
        delta_qty = safe_float(delta_match.group(1))
        if delta_qty is not None:
            return "Long" if delta_qty > 0 else "Short"
    qty_match = QTY_PATTERN.search(message)
    if qty_match:
        qty = safe_float(qty_match.group(1))
        if qty is not None and fallback_direction != "Neutral":
            return fallback_direction
    if "平仓" in message and fallback_direction in {"Long", "Short"}:
        return fallback_direction
    return fallback_direction if fallback_direction in {"Long", "Short"} else "Long"


def parse_recent_trade_rows(log_lines, status, limit=DASHBOARD_TRADE_LIMIT):
    market = status.get("market") or {}
    position = status.get("position") or {}
    signal = status.get("signal") or {}
    symbol = market.get("symbol") or getattr(config, "SYMBOL", "SOL-USDT-SWAP")
    fallback_direction = infer_signal_direction(signal.get("long_prob"), signal.get("short_prob"))
    current_price = safe_float(market.get("last_price"))
    entry_price = safe_float(position.get("entry_price"))

    rows = []
    for line in reversed(log_lines):
        if not EXECUTION_PATTERN.search(line):
            continue
        parsed = parse_log_line(line)
        message = parsed["message"] if parsed else line
        ts = log_ts_to_iso(parsed["ts"]) if parsed else utc_now_iso()
        reason_match = REASON_PATTERN.search(message)
        rows.append(
            {
                "time": ts,
                "symbol": symbol,
                "side": infer_trade_side(message, fallback_direction),
                "entry": entry_price if "执行开仓" in message else None,
                "exit": current_price if "执行平仓" in message else None,
                "pnl": None,
                "fee": None,
                "slippage": None,
                "reason": reason_match.group(1) if reason_match else message,
                "status": infer_trade_status(message),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def build_metrics_snapshot(status, history, risk_snapshot, backtest_summary, backtest_csv_metrics):
    account = status.get("account") or {}
    performance = status.get("performance") or {}
    position = status.get("position") or {}

    current_equity = safe_float(account.get("total_eq")) or safe_float(performance.get("current_total_eq"))
    daily_pnl = compute_daily_pnl(history)
    total_return_pct = safe_float(performance.get("return_pct"))
    if total_return_pct is None:
        total_return_pct = safe_float(backtest_summary.get("return_pct"))

    max_drawdown_pct = None
    curve_drawdowns = []
    peak = None
    for point in history:
        total_eq = safe_float(point.get("total_eq"))
        if total_eq is None:
            continue
        peak = total_eq if peak is None else max(peak, total_eq)
        if peak and peak > 0:
            curve_drawdowns.append((total_eq - peak) / peak * 100.0)
    if curve_drawdowns:
        max_drawdown_pct = min(curve_drawdowns)
    if max_drawdown_pct is None:
        max_drawdown_pct = safe_float(backtest_summary.get("max_drawdown_pct"))

    open_positions = 0
    net_qty = safe_float(position.get("net_qty"))
    if net_qty is not None and abs(net_qty) > 0:
        open_positions = 1

    return {
        "equity": current_equity,
        "daily_pnl": daily_pnl,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": safe_float(backtest_csv_metrics.get("sharpe_ratio")),
        "win_rate_pct": safe_float(backtest_csv_metrics.get("win_rate_pct")),
        "open_positions": open_positions,
        "risk_level": risk_snapshot.get("risk_level"),
        "backtest_trade_count": backtest_summary.get("trade_count"),
        "fees_paid": backtest_summary.get("fees_paid"),
        "slippage_cost": backtest_summary.get("slippage_cost"),
    }


def build_dashboard_bundle():
    raw_status = load_runtime_dashboard_status()
    history = load_runtime_dashboard_history()[-DASHBOARD_HISTORY_LIMIT:]
    frontend_index = os.path.join(FRONTEND_DIST_ROOT, "index.html")
    log_lines = read_log_tail_lines(LOG_FILE)
    recent_events = extract_recent_strategy_events(log_lines, limit=DASHBOARD_EVENT_LIMIT)
    status = enrich_status_with_fallbacks(raw_status, log_lines, recent_events)
    backtest_summary = parse_latest_backtest_summary(log_lines)
    backtest_csv_metrics = parse_latest_backtest_csv_metrics()
    risk_snapshot = build_risk_snapshot(status, history, recent_events)

    return {
        "generated_at": utc_now_iso(),
        "frontend_built": os.path.isfile(frontend_index),
        "status": status,
        "history": history,
        "recent_events": recent_events,
        "strategy_meta": build_strategy_meta(status),
        "strategy_params": build_strategy_params(),
        "signal_summary": build_signal_summary(status),
        "risk_snapshot": risk_snapshot,
        "metrics": build_metrics_snapshot(status, history, risk_snapshot, backtest_summary, backtest_csv_metrics),
        "recent_trades": parse_recent_trade_rows(log_lines, status, limit=DASHBOARD_TRADE_LIMIT),
        "research_metrics": {
            **backtest_summary,
            **backtest_csv_metrics,
        },
    }


def _write_json_response(handler, payload, status=HTTPStatus.OK):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_request(handler):
    content_length = safe_int(handler.headers.get("Content-Length"), 0)
    if content_length <= 0:
        return {}
    raw_body = handler.rfile.read(content_length)
    if not raw_body:
        return {}
    return json.loads(raw_body.decode("utf-8"))


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        static_root = FRONTEND_DIST_ROOT if os.path.isfile(os.path.join(FRONTEND_DIST_ROOT, "index.html")) else BASE_DIR
        super().__init__(*args, directory=static_root, **kwargs)

    def log_message(self, format, *args):
        return

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/dashboard":
            return _write_json_response(self, build_dashboard_bundle())

        if path == "/api/health":
            return _write_json_response(self, {"ok": True, "generated_at": utc_now_iso()})

        frontend_ready = os.path.isfile(os.path.join(FRONTEND_DIST_ROOT, "index.html"))
        if not frontend_ready:
            body = (
                "<html><body style='font-family: sans-serif; padding: 24px;'>"
                "<h1>Dashboard UI not built yet</h1>"
                "<p>Run <code>cd dashboard-ui && npm install && npm run build</code> first, "
                "then restart <code>python -m run.dashboard_server</code>.</p>"
                "<p>API endpoints are already available: <code>/api/dashboard</code> and <code>/api/health</code>.</p>"
                "</body></html>"
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path != "/" and not os.path.exists(os.path.join(FRONTEND_DIST_ROOT, path.lstrip("/"))):
            self.path = "/index.html"

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/strategy-params":
            try:
                payload = _read_json_request(self)
                result = save_strategy_params(payload)
            except ValueError as exc:
                return _write_json_response(
                    self,
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )
            except Exception as exc:
                return _write_json_response(
                    self,
                    {"ok": False, "error": f"failed to save strategy params: {exc}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

            return _write_json_response(self, result)

        if path == "/api/restart-strategy":
            try:
                result = restart_strategy_process()
            except RuntimeError as exc:
                return _write_json_response(
                    self,
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )
            except Exception as exc:
                return _write_json_response(
                    self,
                    {"ok": False, "error": f"failed to restart strategy: {exc}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

            return _write_json_response(self, result)

        return _write_json_response(
            self,
            {"ok": False, "error": f"unsupported path: {path}"},
            status=HTTPStatus.NOT_FOUND,
        )


def main():
    server = ThreadingHTTPServer((DASHBOARD_HOST, DASHBOARD_PORT), DashboardRequestHandler)
    print(f"dashboard_server listening on http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
