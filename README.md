# Quant OKX v5 Production

## 📌 项目简介

本项目为**机构级机器学习驱动的量化实盘系统**，全流程使用 OKX v5 官方 API（非 ccxt），具备：

- ✅ 实盘稳定运行逻辑
- ✅ 风控止盈止损逻辑
- ✅ 机器学习训练与回测模块
- ✅ 完整容错、超时重试机制
- ✅ 完整工程化依赖管理（pyproject.toml）

支持 **OKX 模拟盘 & 实盘一键切换**，可长期 7x24 云端部署。

---

## 📂 项目结构

```yaml
quant_okx_v5_production/
│
├── config.py # 全局配置文件
├── okx_api.py # OKX v5 API封装（账户、行情、下单接口）
├── utils.py # 特征工程、数据获取、Telegram通知、K线容错拉取
│
├── train.py # 机器学习训练脚本
├── sandbox.py # 回测模块
├── live_trading_monitor.py # 实盘轮询执行逻辑
│
├── models/ # 模型文件存储目录
├── logs/ # 实盘日志输出目录
├── data/ # 历史数据缓存目录
│
├── pyproject.toml # 工程化依赖文件 (PDM/Poetry 管理)
└── README.md # 当前文档

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

### 🔑 配置文件说明 (config.py)
请先在 config.py 填写你的 OKX API 参数和交易配置：
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
TELEGRAM_CHAT_ID = '你的TG Chat ID'`
```

### 🚀 部署流程
## 1️⃣ 训练模型
先使用 train.py 完成机器学习训练：


```bash
python train.py
```

训练好的模型将保存在：

```bash
models/model_okx.pkl
```

## 2️⃣ 回测验证（可选）
使用 sandbox.py 进行策略回测与压力测试：

```bash
python sandbox.py
```
## 3️⃣启动实盘系统
正式实盘轮询执行逻辑：

```bash
python live_trading_monitor.py
```
建议结合 Linux crontab、supervisor、systemd、或云端守护进程持续运行。

---
### 🧪 测试方法

* ✅ 模拟盘测试

直接将 USE_SERVER = "1" 即可安全跑模拟盘测试真实下单逻辑。

* ✅ 风控测试

可在训练后，手动制造多空信号快速验证止盈止损逻辑。

* ✅ 异常容错测试

可手动断网测试容错逻辑是否如预期超时重试。

* ✅ Telegram通知测试

可直接手动触发 send_telegram() 发送测试消息。