# Quant Project

基于 OKX 永续合约的量化交易项目，覆盖训练、回测、测试盘联调与 VPS 常驻运行。当前代码主流程围绕 OKX，重点已经补齐以下几块：

- 多周期特征工程：`5m / 15m / 1H`
- 多模型集成：LightGBM、XGBoost、RandomForest
- 更真实的回测撮合：盯市净值、手续费、滑点、资金费、bar 内 TP/SL
- 基于 ATR / 波动率的自适应止盈止损
- 测试盘保护：强制模拟盘、自动校验持仓模式、自动设置杠杆、启动前清挂单
- 实盘状态保护：`clOrdId` 幂等下单、`last_bar_ts` 持久化，避免重启后重复处理同一根 K 线

推荐使用顺序：

1. 训练模型
2. 跑回测
3. 调参自适应 TP/SL
4. 本地测试盘联调
5. VPS 常驻运行测试盘

## 项目结构

```text
quant_sol_project/
├── backtest/                  # 回测逻辑
├── config/                    # .env 与全局配置解析
├── core/                      # 交易所接口、策略核心、特征工程、仓位管理
├── run/                       # 启动脚本、测试盘检查、VPS bootstrap、参数扫描
├── tests/                     # 核心单测
├── train/                     # 模型训练
├── utils/                     # 日志、通知、通用工具
├── ecosystem.paper.config.js  # PM2 测试盘配置
├── .env.example               # 环境变量模板
└── README.md
```

## 核心能力

### 1. 训练

- 从 OKX 拉取多周期历史数据
- 生成多周期特征与高级特征
- 按时间顺序切分训练集 / 测试集，避免未来数据泄漏
- 在训练集内部做类别平衡
- 产出模型文件和特征列表

默认模型输出：

- `models/lgb_model.pkl`
- `models/xgb_model.pkl`
- `models/rf_model.pkl`
- `models/feature_list.pkl`

### 2. 回测

当前回测比旧版本更接近真实运行环境，主要包括：

- 使用盯市净值而不是只看已实现余额
- 计入手续费和滑点成本
- 可选加载资金费历史
- 支持单根 5m bar 内的 TP/SL 触发
- 同一根 bar 同时打到 TP 和 SL 时，默认按最保守的最差情况处理
- 自适应 TP/SL 会根据 ATR 与波动率动态调整阈值

回测输出会包含：

- `final_equity`
- `return_pct`
- `max_drawdown_pct`
- `trade_count`
- `take_profit_count`
- `stop_loss_count`
- `fees_paid`
- `slippage_cost`
- `funding_pnl`

同时会在 `logs/` 下生成带汇总信息的回测 CSV。

### 3. 测试盘 / 运行保护

`run.live_trading_monitor` 启动时会先做交易环境校验：

- `LIVE_REQUIRE_SIMULATED_TRADING=1` 时，要求 `USE_SERVER=1`
- 自动检查并切到 `long_short_mode`
- 自动检查并设置多空双边杠杆
- 启动前可自动清理挂单
- 每次只处理上一根已收盘 bar
- 重启后恢复 `logs/live_trading_state.json` 中的最近处理 bar 时间戳

`core.okx_api` 中还补了幂等保护：

- 下单使用唯一 `clOrdId`
- 下单响应异常时，会按 `clOrdId` 反查是否已受理，避免重复下单

## 环境准备

### Python

推荐 `Python 3.10+`，至少保证和当前依赖兼容。

### 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 初始化配置

```bash
cp .env.example .env
```

至少先检查这些关键配置：

| 配置项 | 说明 | 建议 |
| --- | --- | --- |
| `OKX_API_KEY` / `OKX_SECRET` / `OKX_PASSWORD` | OKX API 凭证 | 测试盘先填模拟盘密钥 |
| `USE_SERVER` | `1`=模拟盘，`0`=实盘 | 联调阶段固定用 `1` |
| `SYMBOL` | 交易标的 | 默认 `SOL-USDT-SWAP` |
| `LEVERAGE` | 杠杆倍数 | 先保守，小杠杆测试 |
| `POSITION_SIZE` | 单次目标资金量 | 先用最小可测仓位 |
| `MODEL_PATHS` | 集成模型路径 | 保持和训练输出一致 |
| `MODEL_WEIGHTS` | 各模型权重 | 和 `MODEL_PATHS` 对齐 |
| `ADAPTIVE_TP_SL_ENABLED` | 是否启用自适应 TP/SL | 建议保持 `1` |
| `LIVE_REQUIRE_SIMULATED_TRADING` | 是否强制模拟盘保护 | 建议保持 `1` |
| `LIVE_AUTO_SET_POSITION_MODE` | 自动切双向持仓 | 建议保持 `1` |
| `LIVE_AUTO_SET_LEVERAGE` | 自动设置杠杆 | 建议保持 `1` |
| `LIVE_RECONCILE_PENDING_ORDERS` | 启动/轮询前清挂单 | 建议保持 `1` |
| `LIVE_PERSIST_LAST_BAR` | 持久化最近 bar | 建议保持 `1` |

如果你是第一次做测试盘联调，建议先保留：

```env
USE_SERVER=1
LIVE_REQUIRE_SIMULATED_TRADING=1
TELEGRAM_ENABLED=0
```

## 常用命令

### 1. 训练模型

```bash
python -m train.train
```

### 2. 跑回测

```bash
python -m backtest.backtest
```

### 3. 扫描自适应 TP/SL 参数

```bash
python -m run.tune_adaptive_tp_sl
```

脚本会输出多组候选参数的回测对比，并给出 `recommended_overrides`。

### 4. 测试盘启动前预检

```bash
PYTHONPATH=. TELEGRAM_ENABLED=0 python run/check_okx_paper_ready.py
```

如果环境无误，会输出类似：

```text
paper_ready_ok
symbol=SOL-USDT-SWAP
use_server=1
leverage=3
...
```

### 5. 启动本地测试盘监控

```bash
PYTHONPATH=. TELEGRAM_ENABLED=0 python -m run.live_trading_monitor
```

### 6. 启动守护调度器

```bash
python -m run.scheduler
```

调度器逻辑：

- 每天凌晨 2 点自动训练和回测
- 其他时间确保 `run.live_trading_monitor` 常驻

## 测试盘联调建议

推荐按这个顺序执行：

1. 先确认 `.env` 使用的是模拟盘密钥，且 `USE_SERVER=1`
2. 先跑 `run/check_okx_paper_ready.py`
3. 手动确认没有遗留仓位和挂单
4. 用最小仓位启动 `run.live_trading_monitor`
5. 观察开仓、加减仓、平仓、重启恢复是否都正常
6. 确认 `logs/live_trading.log` 和 `logs/live_trading_state.json` 正常更新

## VPS 部署

### 1. 一键初始化 Python 环境

```bash
bash run/bootstrap_vps.sh
```

这个脚本会：

- 创建 `.venv`
- 安装依赖
- 给出后续测试盘启动命令

### 2. 安装 PM2

```bash
npm install -g pm2
```

### 3. 先做测试盘预检

```bash
PYTHONPATH=. TELEGRAM_ENABLED=0 .venv/bin/python run/check_okx_paper_ready.py
```

### 4. 用 PM2 常驻测试盘

```bash
pm2 start ecosystem.paper.config.js
pm2 save
```

PM2 里默认跑的是：

```bash
.venv/bin/python -m run.live_trading_monitor
```

### 5. 查看日志

```bash
pm2 logs quant_okx_paper
tail -f logs/live_trading.log
tail -f logs/scheduler.log
```

## 日志与状态文件

- `logs/live_trading.log`: 交易监控主日志
- `logs/scheduler.log`: 调度器日志
- `logs/live_trading_state.json`: 最近处理 bar 的持久化状态
- `logs/backtest_*.csv`: 回测交易记录与汇总

## 单元测试

建议至少跑这几组核心测试：

```bash
python -m unittest \
  tests.test_backtest_equity \
  tests.test_backtest_intrabar \
  tests.test_strategy_core \
  tests.test_live_runtime_state
```

这些测试主要覆盖：

- 盯市净值计算
- 同 bar 内 TP/SL 处理
- 自适应止盈止损阈值
- `clOrdId` 辅助逻辑
- `last_bar_ts` 持久化与恢复

## 风险说明

- 当前代码更适合先跑测试盘，不建议直接切到实盘大资金。
- `LIVE_REQUIRE_SIMULATED_TRADING=1` 是默认保护，不建议轻易关闭。
- 切到真实资金前，至少先完成一轮本地测试盘和一轮 VPS 测试盘观察。
- 即使策略能稳定下单，也仍然要关注交易所接口波动、网络延迟、滑点扩大和资金费变化。

## 历史截图

下面两张图仅作项目历史效果示意，具体结果请以你当前本地训练与回测输出为准。

![训练示意](/Users/honey/PycharmProjects/quant_sol_project/img.png)

![回测示意](/Users/honey/PycharmProjects/quant_sol_project/img_1.png)

## 参考链接

- [OKX API Key 管理](https://www.okx.com/account/my-api)
- [OKX API 文档](https://www.okx.com/docs-v5/zh/#overview)
