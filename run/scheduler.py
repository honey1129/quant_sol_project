# scheduler.py
import logging
import time
import subprocess
import os
import sys
import json
import signal
from datetime import datetime, timedelta

from utils.utils import BASE_DIR, DISPLAY_TIMEZONE
from utils.safe_runner import safe_run
from config import config

log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, 'scheduler.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PID_FILE = os.path.join(log_dir, "live_trading_monitor.pid")
MODEL_RETRAIN_STATE_FILE = os.path.join(log_dir, "model_retrain_state.json")
DAILY_REPORT_STATE_FILE = os.path.join(log_dir, "daily_report_state.json")

def train_job():
    logging.info("🟢 开始训练任务")
    subprocess.run([sys.executable, "-m", "train.train"])
    logging.info("✅ 训练任务完成")

def backtest_job():
    logging.info("🟢 开始回测任务")
    subprocess.run([sys.executable, "-m", "backtest.backtest"])
    logging.info("✅ 回测任务完成")

def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _parse_dt(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone()
        return parsed.replace(tzinfo=None)
    except Exception:
        return None


def _read_monitor_pid():
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        return pid if pid > 0 else None
    except Exception:
        return None


def stop_live_monitor_for_model_reload():
    pid = _read_monitor_pid()
    if not pid or not _pid_is_running(pid):
        return

    logging.info(f"🟡 模型已更新，准备重启 live monitor 以加载新模型 pid={pid}")
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 20
    while time.time() < deadline:
        if not _pid_is_running(pid):
            break
        time.sleep(1)

    if _pid_is_running(pid):
        logging.warning(f"⚠ live monitor 未在20秒内退出，将继续保留当前进程 pid={pid}")
        return

    try:
        os.remove(PID_FILE)
    except FileNotFoundError:
        pass
    logging.info("✅ live monitor 已停止，下一轮 scheduler 将自动拉起")

def ensure_live_monitor_running():
    # 1) pidfile存在且进程仍在 -> 不做事
    pid = _read_monitor_pid()
    if pid and _pid_is_running(pid):
        return

    # 2) 不在运行 -> 拉起常驻进程（非阻塞）
    logging.info("🟡 实盘监控未运行，尝试启动 run.live_trading_monitor")
    p = subprocess.Popen([sys.executable, "-m", "run.live_trading_monitor"])

    with open(PID_FILE, "w") as f:
        f.write(str(p.pid))

    logging.info(f"✅ 已启动实盘监控进程 pid={p.pid}")


def should_run_model_retrain(now=None):
    if not bool(config.MODEL_RETRAIN_ENABLED):
        return False

    now = now or datetime.now()
    scheduled_at = now.replace(
        hour=int(config.MODEL_RETRAIN_HOUR),
        minute=int(config.MODEL_RETRAIN_MINUTE),
        second=0,
        microsecond=0,
    )
    if now < scheduled_at:
        return False

    state = _load_json(MODEL_RETRAIN_STATE_FILE, {})
    last_attempt_at = _parse_dt(state.get("last_attempt_at"))
    if last_attempt_at and last_attempt_at.date() == now.date():
        return False

    last_success_at = _parse_dt(state.get("last_success_at"))
    interval = timedelta(hours=max(1.0, float(config.MODEL_RETRAIN_INTERVAL_HOURS)))
    if last_success_at and now - last_success_at < interval:
        return False

    return True


def model_retrain_job():
    logging.info("🟢 开始自动模型重训")
    result = subprocess.run([sys.executable, "-m", "run.retrain_models"], cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"自动模型重训失败: exit_code={result.returncode}")

    logging.info("✅ 自动模型重训完成")
    if bool(config.MODEL_RETRAIN_RESTART_LIVE_MONITOR):
        stop_live_monitor_for_model_reload()

def should_run_daily_report(now=None):
    now = now or datetime.now()
    if now.tzinfo is None:
        display_now = now.astimezone()
    else:
        display_now = now
    display_now = display_now.astimezone(DISPLAY_TIMEZONE)
    scheduled_at = display_now.replace(
        hour=int(config.DAILY_REPORT_HOUR),
        minute=int(config.DAILY_REPORT_MINUTE),
        second=0,
        microsecond=0,
    )
    if display_now < scheduled_at:
        return False

    state = _load_json(DAILY_REPORT_STATE_FILE, {})
    if state.get("last_report_date") == display_now.strftime("%Y-%m-%d"):
        return False
    return True


def daily_report_job():
    logging.info("🟢 开始生成每日交易复盘")
    result = subprocess.run([sys.executable, "-m", "run.daily_trade_report"], cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"每日交易复盘生成失败: exit_code={result.returncode}")

    now = datetime.now(DISPLAY_TIMEZONE)
    with open(DAILY_REPORT_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_report_date": now.strftime("%Y-%m-%d")}, f, ensure_ascii=False, sort_keys=True)
    logging.info("✅ 每日交易复盘完成")

def scheduler():
    now = datetime.now()

    if should_run_model_retrain(now):
        safe_run(model_retrain_job, max_retry=1)

    if should_run_daily_report(now):
        safe_run(daily_report_job, max_retry=1)

    # 确保实盘常驻进程存在
    safe_run(ensure_live_monitor_running)

if __name__ == '__main__':
    while True:
        scheduler()
        time.sleep(60)
