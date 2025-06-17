import os
from dotenv import load_dotenv
from typing import Callable, Dict

# 自动加载.env文件
load_dotenv()

# ✅ 辅助函数：解析 key:value,key:value 格式字符串为字典
def parse_env_dict(env_str: str, value_type: Callable[[str], any] = str) -> Dict[str, any]:
    items = env_str.split(",") if env_str else []
    parsed = {}
    for item in items:
        key, value = item.split(":")
        parsed[key] = value_type(value)
    return parsed

# ✅ 辅助函数：解析用逗号分隔的列表
def parse_env_list(env_str):
    if not env_str:
        return []
    return [item.strip() for item in env_str.split(",")]

# ✅ OKX API
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_SECRET = os.getenv("OKX_SECRET")
OKX_PASSWORD = os.getenv("OKX_PASSWORD")

USE_SERVER = os.getenv("USE_SERVER", '1')

# ✅ 交易参数
SYMBOL = os.getenv("SYMBOL", "SOL-USDT-SWAP")
LEVERAGE = int(os.getenv("LEVERAGE", 3))
POSITION_SIZE = float(os.getenv("POSITION_SIZE", 50))

# ✅ 多周期
INTERVALS = parse_env_list(os.getenv("INTERVALS", "5m,15m,1H"))
WINDOWS = parse_env_dict(os.getenv("WINDOWS", ""), int)

# ✅ 风控参数
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", 0.02))
STOP_LOSS = float(os.getenv("STOP_LOSS", 0.01))

# ✅ 策略阈值
THRESHOLD_LONG = float(os.getenv("THRESHOLD_LONG", 0.55))
THRESHOLD_SHORT = float(os.getenv("THRESHOLD_SHORT", 0.45))

# ✅ 合约配置
LOT_SIZE = float(os.getenv("LOT_SIZE", 0.01))
TICK_SIZE = float(os.getenv("TICK_SIZE", 0.001))

# ✅ 模型配置
MODEL_PATH = os.getenv("MODEL_PATH", "models/model_okx.pkl")
FEATURE_LIST_PATH = os.getenv("FEATURE_LIST_PATH", "models/feature_list.pkl")
MODEL_PATHS = parse_env_dict(os.getenv("MODEL_PATHS", ""), str)
MODEL_WEIGHTS = parse_env_dict(os.getenv("MODEL_WEIGHTS", ""), float)

# ✅ 信号平滑参数
SMOOTH_ALPHA = float(os.getenv("SMOOTH_ALPHA", 0.3))

# ✅ 仓位边界
POSITION_MIN = float(os.getenv("POSITION_MIN", 0.05))
POSITION_MAX = float(os.getenv("POSITION_MAX", 0.3))
MAX_POSITION_RATIO = float(os.getenv("MAX_POSITION_RATIO", 0.3))
MIN_ADJUST_AMOUNT = float(os.getenv("MIN_ADJUST_AMOUNT", 50))
ADJUST_UNIT = float(os.getenv("ADJUST_UNIT", 50))

# ✅ Kelly 盈亏比
KELLY_REWARD_RISK = float(os.getenv("KELLY_REWARD_RISK", 2.5))

# ✅ 动态风险预算
TARGET_VOL = float(os.getenv("TARGET_VOL", 0.015))

# ✅ 回测参数
MAX_POSITION = float(os.getenv("MAX_POSITION", 0.4))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", 1000))
FEE_RATE = float(os.getenv("FEE_RATE", 0.0005))

# ✅ Telegram配置
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
