import logging
import time
import subprocess
import os
import sys

from utils.utils import BASE_DIR
from utils.safe_runner import safe_run

# 保证日志目录存在 (统一项目绝对路径)
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

# 初始化日志
logging.basicConfig(
    filename=os.path.join(log_dir, 'scheduler.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ✅ 执行 train 模块
def train_job():
    logging.info("🟢 开始训练任务")
    subprocess.run([sys.executable, "-m", "train.train"])
    logging.info("✅ 训练任务完成")

# ✅ 执行 backtest 模块
def backtest_job():
    logging.info("🟢 开始回测任务")
    subprocess.run([sys.executable, "-m", "backtest.backtest"])
    logging.info("✅ 回测任务完成")

# ✅ 执行实盘模块
def live_trade_job():
    logging.info("🟢 开始实盘交易任务")
    subprocess.run([sys.executable, "-m", "run.live_trading_monitor"])
    logging.info("✅ 实盘交易完成")

# 核心调度逻辑
def scheduler():
    now = time.localtime()

    # 每天凌晨2点自动训练与回测
    if now.tm_hour == 2 and now.tm_min == 0:
        safe_run(train_job)
        safe_run(backtest_job)

    # 每 5 分钟执行实盘轮询
    elif now.tm_min % 5 == 0:
        safe_run(live_trade_job)

if __name__ == '__main__':
    while True:
        scheduler()
        time.sleep(60)
