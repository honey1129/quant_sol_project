
# Quant SOL/USDT Production System (Restructured V5)

## 📌 项目简介

本项目为 **机构级机器学习驱动的量化实盘系统 (SOL/USDT 专项)**，全流程使用 OKX v5 官方 API，具备：

- ✅ 实盘稳定运行逻辑
- ✅ 风控止盈止损逻辑
- ✅ 机器学习训练与回测模块
- ✅ 完整容错、超时重试机制
- ✅ 完整工程化依赖管理

支持 **OKX 模拟盘 & 实盘一键切换**，可长期 7x24 云端部署。

---

## 📂 项目结构

```yaml
quant_sol_project/
│
├── config/
│   └── config.py              # 全局配置文件
│
├── core/
│   ├── okx_api.py             # OKX API 封装
│   ├── ml_feature_engineering.py  # 特征工程逻辑
│   ├── signal_engine.py       # 多模型信号融合逻辑
│   ├── position_manager.py    # 仓位管理逻辑
│   └── predict.py             # 预测模块
│
├── models/                    # 模型文件存储目录
│   ├── feature_list.pkl
│   └── model_okx.pkl
│
├── train/
│   └── train.py               # 机器学习训练脚本
│
├── run/
│   ├── live_trading_monitor.py  # 实盘轮询执行逻辑
│   └── scheduler.py           # 定时任务调度器
│
├── backtest/
│   └── backtest.py            # 策略回测模块
│
├── tests/
│   └── okx-api-test.py        # OKX API 测试模块
│
├── utils/
│   ├── safe_runner.py         # 容错安全封装
│   └── utils.py               # 工具函数集合
│
├── logs/                      # 实盘日志输出目录
│
├── requirements.txt           # 依赖文件
├── pyproject.toml             # 工程化依赖 (可选)
└── README.md                  # 当前文档
```

---

## ⚙ 环境准备

### 推荐使用 Python 3.9+

建议使用 `pdm` 或 `poetry` 进行依赖管理。

### 使用 PDM 安装（推荐）

```bash
# 安装pdm（如未安装）
pip install pdm

# 安装项目依赖
pdm install
```

### 使用 poetry 安装 (可选)

```bash
# 安装 poetry（如未安装）
pip install poetry

# 安装项目依赖
poetry install
```

---

## 🔑 配置文件说明

请在 `config/config.py` 填写你的 OKX API 参数和交易配置：

```python
OKX_API_KEY = '你的APIKey'
OKX_SECRET = '你的Secret'
OKX_PASSWORD = '你的Passphrase'

USE_SERVER = "0"  # "0" 实盘, "1" 模拟盘

SYMBOL = 'SOL-USDT-SWAP'
LEVERAGE = 3
POSITION_SIZE = 50

TAKE_PROFIT = 0.02
STOP_LOSS = 0.01

THRESHOLD_LONG = 0.55
THRESHOLD_SHORT = 0.45

MODEL_PATH = 'models/model_okx.pkl'

TELEGRAM_BOT_TOKEN = '你的TG Bot Token'
TELEGRAM_CHAT_ID = '你的TG Chat ID'
```

---

## 🚀 部署流程

### 1️⃣ 训练模型

先使用 `train/train.py` 完成机器学习训练：

```bash
python train/train.py
```

训练好的模型将保存在：

```bash
models/model_okx.pkl
```

### 2️⃣ 回测验证（可选）

使用 `backtest/backtest.py` 进行策略回测与压力测试：

```bash
python backtest/backtest.py
```

### 3️⃣ 启动实盘系统

正式实盘轮询执行逻辑：

```bash
python run/live_trading_monitor.py
```

建议结合 Linux crontab、supervisor、systemd、或云端守护进程持续运行。

---

## 🧪 测试方法

- ✅ 模拟盘测试：直接将 `USE_SERVER = "1"` 即可安全跑模拟盘测试真实下单逻辑。
- ✅ 风控测试：可在训练后，手动制造多空信号快速验证止盈止损逻辑。
- ✅ 异常容错测试：可手动断网测试容错逻辑是否如预期超时重试。
- ✅ Telegram通知测试：可直接手动触发 send_telegram() 发送测试消息。

---

## 📈 未来升级方向

- 多模型融合 (LightGBM、XGBoost、RandomForest)
- 多周期特征支持（5m、15m、1h、4h 等）
- 智能仓位动态管理
- 风控增强与资金曲线可视化监控
- 云端自动部署支持

