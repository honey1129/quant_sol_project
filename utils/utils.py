import os
import requests
from config import config
import logging

# ✅ 统一定义项目根目录 (无论在哪个子模块调用，都能自动找对)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ 模型与日志目录（全部基于BASE_DIR）
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# ✅ 自动确保目录存在
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ✅ 统一日志文件
LOG_FILE = os.path.join(LOGS_DIR, 'live_trading.log')

# ✅ 日志配置 (完整日志 + 防止okx包内日志干扰)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ✅ 屏蔽冗余日志
logging.getLogger("okx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ✅ 统一日志封装
def log_info(msg):
    print(msg)
    logging.info(msg)
    send_telegram(msg)

def log_error(msg):
    print("❌", msg)
    logging.error(msg)
    send_telegram(f"❌ {msg}")

# ✅ Telegram 通知模块
def send_telegram(message):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": config.TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram通知失败: {e}")
