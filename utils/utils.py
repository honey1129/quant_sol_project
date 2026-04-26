import logging
import os
from datetime import datetime, timedelta, timezone

import requests
from zoneinfo import ZoneInfo

from config import config

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
DISPLAY_TIMEZONE_NAME = "Asia/Shanghai"
try:
    DISPLAY_TIMEZONE = ZoneInfo(DISPLAY_TIMEZONE_NAME)
except Exception:
    DISPLAY_TIMEZONE = timezone(timedelta(hours=8))


class DisplayTimezoneFormatter(logging.Formatter):
    """Force log timestamps to render in Asia/Shanghai for operator clarity."""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=DISPLAY_TIMEZONE)
        if datefmt:
            return dt.strftime(datefmt)
        base = dt.strftime(self.default_time_format)
        return self.default_msec_format % (base, record.msecs)

# ✅ 日志配置 (完整日志 + 防止okx包内日志干扰)
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_has_project_file_handler = False
for _handler in _root_logger.handlers:
    if isinstance(_handler, logging.FileHandler) and getattr(_handler, "baseFilename", "") == LOG_FILE:
        _handler.setFormatter(DisplayTimezoneFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        _has_project_file_handler = True

if not _has_project_file_handler:
    _file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _file_handler.setFormatter(DisplayTimezoneFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    _root_logger.addHandler(_file_handler)

# ✅ 屏蔽冗余日志
logging.getLogger("okx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_telegram_disabled_for_process = False

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
    global _telegram_disabled_for_process

    if _telegram_disabled_for_process or not getattr(config, "TELEGRAM_ENABLED", True):
        return

    bot_token = str(getattr(config, "TELEGRAM_BOT_TOKEN", "") or "").strip()
    chat_id = str(getattr(config, "TELEGRAM_CHAT_ID", "") or "").strip()
    placeholder_values = {
        "",
        "你的TG_BOT_TOKEN",
        "你的TG_CHAT_ID",
        "YOUR_TG_BOT_TOKEN",
        "YOUR_TG_CHAT_ID",
    }
    if bot_token in placeholder_values or chat_id in placeholder_values:
        return

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": config.TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload, timeout=5)
        response.raise_for_status()
    except Exception as e:
        _telegram_disabled_for_process = True
        print(f"Telegram通知失败，当前进程后续已静默: {e}")
