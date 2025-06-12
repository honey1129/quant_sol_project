import logging
import time
import subprocess
import os
from safe_runner import safe_run

# 确保日志目录存在
os.makedirs("logs", exist_ok=True)

# 初始化日志系统
logging.basicConfig(
    filename='logs/scheduler.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_job():
    logging.info("🟢 开始训练任务")
    subprocess.run(['python', 'train.py'])
    logging.info("✅ 训练任务完成")

def backtest_job():
    logging.info("🟢 开始回测任务")
    subprocess.run(['python', 'sandbox.py'])
    logging.info("✅ 回测任务完成")

def live_trade_job():
    logging.info("🟢 开始实盘交易任务")
    subprocess.run(['python', 'live_trading_monitor.py'])
    logging.info("✅ 实盘交易完成")

def scheduler():
    now = time.localtime()
    safe_run(live_trade_job)

    # 每天凌晨2点整点训练 + 回测
    if now.tm_hour == 2 and now.tm_min == 0:
        safe_run(train_job)
        safe_run(backtest_job)



if __name__ == '__main__':
    while True:
        scheduler()
        time.sleep(300)。# 每5分钟执行一次（5 x 60 = 300秒）
