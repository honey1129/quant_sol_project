# scheduler.py
import logging
import time
import subprocess
import os
import sys

from utils.utils import BASE_DIR
from utils.safe_runner import safe_run

log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, 'scheduler.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PID_FILE = os.path.join(log_dir, "live_trading_monitor.pid")

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

def ensure_live_monitor_running():
    # 1) pidfile存在且进程仍在 -> 不做事
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            if pid > 0 and _pid_is_running(pid):
                return
        except Exception:
            pass

    # 2) 不在运行 -> 拉起常驻进程（非阻塞）
    logging.info("🟡 实盘监控未运行，尝试启动 run.live_trading_monitor")
    p = subprocess.Popen([sys.executable, "-m", "run.live_trading_monitor"])

    with open(PID_FILE, "w") as f:
        f.write(str(p.pid))

    logging.info(f"✅ 已启动实盘监控进程 pid={p.pid}")

def scheduler():
    now = time.localtime()

    # 每天凌晨2点自动训练与回测
    if now.tm_hour == 2 and now.tm_min == 0:
        safe_run(train_job)
        safe_run(backtest_job)

    # 其他时间：确保实盘常驻进程存在
    else:
        safe_run(ensure_live_monitor_running)

if __name__ == '__main__':
    while True:
        scheduler()
        time.sleep(60)
