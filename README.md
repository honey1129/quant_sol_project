# Quant Sol Project

面向 OKX `SOL-USDT-SWAP` 永续合约的量化交易研究与模拟盘执行系统，覆盖模型训练、回测、严格样本外验证、实时交易风控和运行监控。

## 当前状态

- 默认 ML 策略仅作为模拟盘运行基线，不代表已通过实盘上线门槛。
- 旧策略未通过严格滚动样本外审计，不再继续使用同一批数据调参。
- Directional V2 正在冻结留出期内收集全新未见数据，最早于 2026-08-22 且满足成交样本门槛后评估。
- 项目当前用于研究和测试盘验证，不建议连接真实资金。

## 核心能力

- 5m / 15m / 1H 多周期特征工程与 ML 信号生成
- 回测、walk-forward 和严格滚动样本外验证
- OKX WebSocket 实时价格与仓位订阅
- 交易所端 OCO 止盈止损、本地实时风控和账户熔断
- 订单幂等、成交确认、重启对账和审计记录
- PM2 常驻运行、Dashboard 监控和 Telegram 通知

## 项目结构

```text
backtest/       回测引擎
config/         环境变量与全局配置
core/           信号、风控、仓位和 OKX API
dashboard-ui/   实时监控面板
monitoring/     运行报告
research/       冻结实验规范与结果
run/            交易、验证、重训和部署入口
tests/          自动化测试
train/          模型训练
utils/          日志、审计和通用工具
```

## 快速开始

需要 Python 3.10+。首次使用时创建虚拟环境并复制配置模板：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写 OKX 模拟盘凭证，并保持以下保护项：

```env
USE_SERVER=1
LIVE_REQUIRE_SIMULATED_TRADING=1
EXCHANGE_TPSL_ENABLED=1
POLL_SEC=1
```

## 常用命令

| 用途 | 命令 |
|---|---|
| 运行测试 | `python -m pytest -q` |
| 训练模型 | `python -m train.train` |
| 运行回测 | `python -m backtest.backtest` |
| 测试盘预检 | `PYTHONPATH=. TELEGRAM_ENABLED=0 python run/check_okx_paper_ready.py` |
| 启动测试盘 | `PYTHONPATH=. TELEGRAM_ENABLED=0 python -m run.live_trading_monitor` |
| 严格 OOS 审计 | `python -m run.strict_oos_validation` |
| 查看 V2 留出状态 | `python -m run.directional_v2_experiment` |
| 生成成交日报 | `PYTHONPATH=. python -m run.daily_trade_report` |

## 核心风控

- 开仓后立即创建 `reduceOnly` 的交易所端 OCO 止盈止损单。
- 实时风控优先使用 WebSocket，数据过期时快速降级到限时 REST 读取。
- Kill Switch 和单日亏损熔断只禁止新开仓，不阻止平仓。
- 下单前重新核对仓位，超时订单主动撤销，重启后重新接管仓位和保护单。
- 日志与成交审计记录决策、成交、延迟和滑点，便于复盘。

## 风险提示

历史回测和模拟盘结果不代表未来收益。切换实盘前必须完成独立样本外验证，并确认策略收益能够覆盖手续费、滑点、资金费和极端行情风险。
