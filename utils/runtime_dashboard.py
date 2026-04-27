import copy
import json
import math
import os
from datetime import datetime, timezone

from utils.utils import LOGS_DIR, log_error


RUNTIME_DASHBOARD_STATUS_PATH = os.path.join(LOGS_DIR, "runtime_dashboard_status.json")
RUNTIME_DASHBOARD_HISTORY_PATH = os.path.join(LOGS_DIR, "runtime_dashboard_history.json")
RUNTIME_DASHBOARD_BASELINE_PATH = os.path.join(LOGS_DIR, "runtime_dashboard_baseline.json")
RUNTIME_DASHBOARD_MAX_HISTORY_POINTS = 1440


def _utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _backup_corrupt_file(path, original_error):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    corrupt_path = f"{path}.corrupt-{ts}"
    try:
        os.replace(path, corrupt_path)
        log_error(f"⚠ runtime_dashboard JSON 损坏，已备份到 {corrupt_path}: {original_error}")
    except Exception as backup_exc:
        log_error(
            f"⚠ runtime_dashboard JSON 损坏且备份失败: path={path}, "
            f"err={original_error}, backup_err={backup_exc}"
        )


def _read_json(path, default, *, backup_on_corrupt=False):
    if not os.path.exists(path):
        return copy.deepcopy(default)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        if backup_on_corrupt:
            _backup_corrupt_file(path, exc)
        else:
            log_error(f"⚠ runtime_dashboard JSON 损坏: path={path}, err={exc}")
        return copy.deepcopy(default)


def _write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_runtime_dashboard_status():
    return _read_json(RUNTIME_DASHBOARD_STATUS_PATH, {})


def load_runtime_dashboard_history():
    history = _read_json(RUNTIME_DASHBOARD_HISTORY_PATH, [], backup_on_corrupt=True)
    return history if isinstance(history, list) else []


def _load_or_initialize_baseline(total_eq):
    file_corrupt = False
    baseline = {}

    if os.path.exists(RUNTIME_DASHBOARD_BASELINE_PATH):
        try:
            with open(RUNTIME_DASHBOARD_BASELINE_PATH, "r", encoding="utf-8") as f:
                baseline = json.load(f)
        except Exception as exc:
            file_corrupt = True
            _backup_corrupt_file(RUNTIME_DASHBOARD_BASELINE_PATH, exc)
            baseline = {}

    baseline_total_eq = _safe_float(baseline.get("baseline_total_eq"))
    if baseline_total_eq is not None and baseline_total_eq > 0:
        return baseline_total_eq, baseline

    # 损坏时拒绝静默重置：保留基线为空，等待人工修复
    if file_corrupt:
        return None, {}

    total_eq = _safe_float(total_eq)
    if total_eq is None or total_eq <= 0:
        return None, baseline

    baseline = {
        "baseline_total_eq": total_eq,
        "initialized_at": _utc_now_iso(),
    }
    _write_json_atomic(RUNTIME_DASHBOARD_BASELINE_PATH, baseline)
    return total_eq, baseline


def _upsert_history_point(history, point, max_points=RUNTIME_DASHBOARD_MAX_HISTORY_POINTS):
    point_key = str(point.get("bar_ts") or point.get("timestamp") or "")
    if history:
        latest_key = str(history[-1].get("bar_ts") or history[-1].get("timestamp") or "")
        if latest_key == point_key:
            history[-1] = point
        else:
            history.append(point)
    else:
        history.append(point)

    if len(history) > max_points:
        history = history[-max_points:]
    return history


def _compute_performance(history, baseline_total_eq):
    total_eq_values = [
        _safe_float(item.get("total_eq"))
        for item in history
    ]
    total_eq_values = [value for value in total_eq_values if value is not None and value > 0]

    current_total_eq = total_eq_values[-1] if total_eq_values else None
    peak_total_eq = max(total_eq_values) if total_eq_values else None
    min_total_eq = min(total_eq_values) if total_eq_values else None

    net_pnl = None
    return_pct = None
    drawdown_pct = None

    if current_total_eq is not None and baseline_total_eq is not None and baseline_total_eq > 0:
        net_pnl = current_total_eq - baseline_total_eq
        return_pct = net_pnl / baseline_total_eq * 100.0

    if current_total_eq is not None and peak_total_eq is not None and peak_total_eq > 0:
        drawdown_pct = (current_total_eq - peak_total_eq) / peak_total_eq * 100.0

    return {
        "baseline_total_eq": baseline_total_eq,
        "current_total_eq": current_total_eq,
        "peak_total_eq": peak_total_eq,
        "min_total_eq": min_total_eq,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
        "drawdown_pct": drawdown_pct,
        "history_points": len(history),
    }


def write_runtime_dashboard_snapshot(snapshot, *, history_point=None):
    payload = copy.deepcopy(snapshot)
    payload["updated_at"] = payload.get("updated_at") or _utc_now_iso()

    account = payload.setdefault("account", {})
    runtime = payload.setdefault("runtime", {})

    total_eq = _safe_float(account.get("total_eq"))
    avail_eq = _safe_float(account.get("avail_eq"))
    if total_eq is not None:
        account["total_eq"] = total_eq
    if avail_eq is not None:
        account["avail_eq"] = avail_eq

    baseline_total_eq, baseline_payload = _load_or_initialize_baseline(total_eq)
    payload["baseline"] = baseline_payload

    history = load_runtime_dashboard_history()
    if history_point is not None:
        point = copy.deepcopy(history_point)
        point["timestamp"] = point.get("timestamp") or payload["updated_at"]
        point_total_eq = _safe_float(point.get("total_eq"))
        point_avail_eq = _safe_float(point.get("avail_eq"))
        if point_total_eq is not None:
            point["total_eq"] = point_total_eq
        if point_avail_eq is not None:
            point["avail_eq"] = point_avail_eq

        if baseline_total_eq is not None and point_total_eq is not None and baseline_total_eq > 0:
            point["net_pnl"] = point_total_eq - baseline_total_eq
            point["return_pct"] = point["net_pnl"] / baseline_total_eq * 100.0

        history = _upsert_history_point(history, point)
        _write_json_atomic(RUNTIME_DASHBOARD_HISTORY_PATH, history)

    performance = _compute_performance(history, baseline_total_eq)
    payload["performance"] = performance
    runtime["last_status"] = runtime.get("last_status") or "unknown"
    runtime["loop_count"] = _safe_int(runtime.get("loop_count"), 0)
    runtime["same_bar_skip_count"] = _safe_int(runtime.get("same_bar_skip_count"), 0)

    _write_json_atomic(RUNTIME_DASHBOARD_STATUS_PATH, payload)
    return payload
