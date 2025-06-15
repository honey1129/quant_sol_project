# OKX 实盘API配置
# OKX_API_KEY = '1eb8b2ff-ba12-4712-9326-a0e12ec491f0'
# OKX_SECRET = '2A57C2C82FC251372CEC8B0C38E843D7'
# OKX_PASSWORD = 'Liu*7046937'



# === OKX API 账户信息 ===
OKX_API_KEY = '69f40cb6-b305-405d-8442-b7b535b637e9'
OKX_SECRET = '30B4C825F95EC5743CF3158F060C4306'
OKX_PASSWORD = 'Liu*7046937'

# 是否使用模拟盘 (0为实盘，1为模拟盘)
USE_SERVER = "1"

# 合约下单单位与价格精度
LOT_SIZE = 0.01   # 最小下单数量 (单位: 合约张数)
TICK_SIZE = 0.01  # 最小价格变动 (单位: USD)

USE_SERVER_TIME = True

# === 交易品种 ===
SYMBOL = "SOL-USDT-SWAP"

# === 多周期 ===
INTERVALS = ["5m", "15m", "1H"]
WINDOWS = {
    "5m": 5000,
    "15m": 5000,
    "1H": 2000
}

# 技术指标参数
MA_PERIOD = 20
RSI_PERIOD = 14

# === 历史数据窗口长度 ===
LOOKBACK = 5000

# === 风控与策略参数 ===
BASE_THRESHOLD = 0.5        # 模型阈值
TREND_FILTER_BASE = 0.25    # 趋势过滤器强度
VOLATILITY_LIMIT = 0.05     # 波动率过滤上限

MAX_POSITION = 0.4          # 最大仓位占比（仓位管理核心参数）
BASE_TP = 0.04              # 止盈比例
BASE_SL = 0.015             # 止损比例

INITIAL_BALANCE = 1000      # 回测初始资金

LEVERAGE = 3                # 杠杆倍数
FEE_RATE = 0.0005           # 交易手续费（买卖双边各0.05%）

# 回测参数
POSITION_SIZE = 200     # 单次投入金额

# === 风控参数 ===
MODEL_PATH = 'models/model_okx.pkl'


# 风控参数（默认初始）
TAKE_PROFIT = 0.05
STOP_LOSS = 0.02

# 动态仓位参数
POSITION_MIN = 0.1  # 最小仓位占比 (10%)
POSITION_MAX = 0.8  # 最大仓位占比 (80%)
ADJUST_UNIT = 50  # 最小调仓单位 (USDT)

# 机器学习预测信号阈值
THRESHOLD_LONG = 0.6
THRESHOLD_SHORT = 0.4

# Telegram通知配置
TELEGRAM_BOT_TOKEN = '7209058361:AAE8M4ZJK0PjYPg6ahGHIlpsxzQoI33WxAA'
TELEGRAM_CHAT_ID = '8047263861'