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


def parse_env_bool(env_str, default=False):
    if env_str is None:
        return default
    return str(env_str).strip().lower() in {"1", "true", "yes", "on"}

EXCHANGE = os.getenv("EXCHANGE", "OKX")

# ✅ OKX API
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_SECRET = os.getenv("OKX_SECRET")
OKX_PASSWORD = os.getenv("OKX_PASSWORD")

# ✅ WEEX API
WEEX_API_KEY = os.getenv("WEEX_API_KEY")
WEEX_SECRET = os.getenv("WEEX_SECRET")
WEEX_PASSWORD = os.getenv("WEEX_PASSWORD")

USE_SERVER = os.getenv("USE_SERVER", '1')

# ✅ 交易参数
SYMBOL = os.getenv("SYMBOL", "SOL-USDT-SWAP")
LEVERAGE = int(os.getenv("LEVERAGE", 3))
POSITION_SIZE = float(os.getenv("POSITION_SIZE", 50))

# ✅ 多周期
INTERVALS = parse_env_list(os.getenv("INTERVALS", "5m,15m,1H"))
WINDOWS = parse_env_dict(os.getenv("WINDOWS", ""), int)
MA_PERIOD = int(os.getenv("MA_PERIOD", 34))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))

# ✅ 风控参数
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", 0.02))
STOP_LOSS = float(os.getenv("STOP_LOSS", 0.01))
ADAPTIVE_TP_SL_ENABLED = parse_env_bool(os.getenv("ADAPTIVE_TP_SL_ENABLED"), True)
ATR_TAKE_PROFIT_MULTIPLIER = float(os.getenv("ATR_TAKE_PROFIT_MULTIPLIER", 5.0))
ATR_STOP_LOSS_MULTIPLIER = float(os.getenv("ATR_STOP_LOSS_MULTIPLIER", 2.2))
VOLATILITY_TAKE_PROFIT_MULTIPLIER = float(os.getenv("VOLATILITY_TAKE_PROFIT_MULTIPLIER", 7.0))
VOLATILITY_STOP_LOSS_MULTIPLIER = float(os.getenv("VOLATILITY_STOP_LOSS_MULTIPLIER", 2.6))
ADAPTIVE_TAKE_PROFIT_MIN = float(os.getenv("ADAPTIVE_TAKE_PROFIT_MIN", 0.009))
ADAPTIVE_TAKE_PROFIT_MAX = float(os.getenv("ADAPTIVE_TAKE_PROFIT_MAX", 0.045))
ADAPTIVE_STOP_LOSS_MIN = float(os.getenv("ADAPTIVE_STOP_LOSS_MIN", 0.0055))
ADAPTIVE_STOP_LOSS_MAX = float(os.getenv("ADAPTIVE_STOP_LOSS_MAX", 0.025))

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


TRAILING_STOP = float(os.getenv("TRAILING_STOP", 0.03))           # 移动止损 3%
MAX_HOLD_BARS = float(os.getenv("MAX_HOLD_BARS", 96))

MIN_HOLD_BARS=float(os.getenv("MIN_HOLD_BARS", 8))
TRAILING_EXIT=float(os.getenv("TRAILING_EXIT", 0.008))

# ✅ 仓位边界
POSITION_MIN = float(os.getenv("POSITION_MIN", 0.05))
POSITION_MAX = float(os.getenv("POSITION_MAX", 0.3))
MAX_POSITION_RATIO = float(os.getenv("MAX_POSITION_RATIO", 0.3))
BASE_POSITION_RATIO = float(os.getenv("BASE_POSITION_RATIO", 0.1))
MIN_ADJUST_AMOUNT = float(os.getenv("MIN_ADJUST_AMOUNT", 50))
ADJUST_UNIT = float(os.getenv("ADJUST_UNIT", 50))
ADD_THRESHOLD = float(os.getenv("ADD_THRESHOLD", 0.15))
MAX_REBALANCE_RATIO = float(os.getenv("MAX_REBALANCE_RATIO", 0.3))
SIGNAL_MIN_PROB_DIFF = float(os.getenv("SIGNAL_MIN_PROB_DIFF", 0.18))
MIN_SIGNAL_TARGET_RATIO = float(os.getenv("MIN_SIGNAL_TARGET_RATIO", 0.08))
REVERSE_SIGNAL_MIN_PROB_DIFF = float(os.getenv("REVERSE_SIGNAL_MIN_PROB_DIFF", 0.26))
REVERSE_MIN_TARGET_RATIO = float(os.getenv("REVERSE_MIN_TARGET_RATIO", 0.12))

# ✅ Kelly 盈亏比
KELLY_REWARD_RISK = float(os.getenv("KELLY_REWARD_RISK", 2.5))

# ✅ 动态风险预算
TARGET_VOL = float(os.getenv("TARGET_VOL", 0.015))

# ✅ 回测参数
MAX_POSITION = float(os.getenv("MAX_POSITION", 0.4))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", 1000))
FEE_RATE = float(os.getenv("FEE_RATE", 0.0005))
BACKTEST_SLIPPAGE_BPS = float(os.getenv("BACKTEST_SLIPPAGE_BPS", 3.0))
BACKTEST_ENABLE_FUNDING = parse_env_bool(os.getenv("BACKTEST_ENABLE_FUNDING"), True)
BACKTEST_FUNDING_HISTORY_LIMIT = int(os.getenv("BACKTEST_FUNDING_HISTORY_LIMIT", 400))
BACKTEST_INTRABAR_TP_SL = parse_env_bool(os.getenv("BACKTEST_INTRABAR_TP_SL"), True)
BACKTEST_WORST_CASE_TP_SL = parse_env_bool(os.getenv("BACKTEST_WORST_CASE_TP_SL"), True)

# ✅ 实盘/模拟盘保护
LIVE_REQUIRE_SIMULATED_TRADING = parse_env_bool(os.getenv("LIVE_REQUIRE_SIMULATED_TRADING"), True)
LIVE_AUTO_SET_POSITION_MODE = parse_env_bool(os.getenv("LIVE_AUTO_SET_POSITION_MODE"), True)
LIVE_AUTO_SET_LEVERAGE = parse_env_bool(os.getenv("LIVE_AUTO_SET_LEVERAGE"), True)
LIVE_RECONCILE_PENDING_ORDERS = parse_env_bool(os.getenv("LIVE_RECONCILE_PENDING_ORDERS"), True)
LIVE_PERSIST_LAST_BAR = parse_env_bool(os.getenv("LIVE_PERSIST_LAST_BAR"), True)

# ✅ Telegram配置
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = parse_env_bool(os.getenv("TELEGRAM_ENABLED"), True)


POLL_SEC=os.getenv("POLL_SEC", 10)
