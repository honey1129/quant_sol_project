# Quant Project

基于 OKX 永续合约的量化交易项目，覆盖训练、回测、测试盘联调与 VPS 常驻运行。当前代码主流程围绕 OKX，重点已经补齐以下几块：

- 多周期特征工程：`5m / 15m / 1H`
- 平稳化模型特征：绝对价格/成交量列保留下游使用，但不会直接喂给模型，避免模型记忆训练期价格带
- 可选 Rubik 特征：OI、taker 主动成交量、多空账户比默认关闭，确认 OOS 有效后再启用
- 多模型集成：LightGBM、XGBoost、RandomForest
- 更真实的回测撮合：盯市净值、手续费、滑点、资金费、bar 内 TP/SL
- 基于 ATR / 波动率的自适应止盈止损
- 高周期趋势过滤与 regime 分层过滤，避免在高波动趋势行情里逆势开仓
- 动态风险缩放、冷却期、最小调仓金额和亏损加仓拦截，减少手续费/滑点磨损
- 测试盘保护：强制模拟盘、自动校验持仓模式、自动设置杠杆、启动前清挂单
- 实盘状态保护：`clOrdId` 幂等下单、`last_bar_ts` / 冷却状态持久化，避免重启后重复处理同一根 K 线或立刻连续交易
- 自动重训带 walk-forward、OOS 回测、regime 偏置门禁和旧模型回滚保护

推荐使用顺序：

1. 训练模型
2. 跑模型概率诊断与 OOS 回测
3. 校准交易阈值 / 自适应 TP/SL
4. 本地测试盘联调
5. VPS 常驻运行测试盘

## 项目结构

```text
quant_sol_project/
├── backtest/                  # 回测逻辑
├── config/                    # .env 与全局配置解析
├── core/                      # 交易所接口、策略核心、特征工程、仓位管理
├── dashboard-ui/              # React + Vite 运行状态面板
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
- 生成多周期特征、高级特征、regime/trend 显式特征和平稳化派生特征
- 训练时通过 `model_feature_columns()` 排除绝对价格水位、原始成交量、累计 OBV、confirm 标志等非平稳列
- `.env.example` 默认使用 `MODEL_LABEL_FUTURE_WINDOW=8` / `MODEL_LABEL_THRESHOLD=0.004`，让标签避开 5m 小噪声
- 默认关闭 label 阶段 tradable 过滤：`MODEL_TRAIN_TRADABLE_LABELS=0`，regime/trend 闸仍在交易决策层生效
- 默认启用三分类质量标签：short / long / no_trade
- 可选接入 Rubik OI/taker/多空比平稳特征：`MODEL_USE_RUBIK_FEATURES=1` 时才拉取
- 按时间顺序切分训练集 / 验证集 / 最终 OOS 回测集，中间留 purge gap，避免未来数据泄漏
- 使用 sample weight 平衡 long / short / no_trade，并在方向内部按 regime 平衡，不再随机下采样破坏时间序列
- 产出模型文件、特征列表和训练元数据

默认模型输出：

- `models/lgb_model.pkl`
- `models/xgb_model.pkl`
- `models/rf_model.pkl`
- `models/feature_list.pkl`
- `models/training_metadata.json`

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
- `closed_trade_count`
- `win_rate_pct`
- `profit_factor`
- `avg_win_loss_ratio`
- `net_pnl_after_costs`
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
- 重启后恢复 `cooldown_bars_remaining`，交易后冷却不会因为进程重启而丢失

`core.okx_api` 中还补了幂等保护：

- 下单使用唯一 `clOrdId`
- 下单响应异常时，会按 `clOrdId` 反查是否已受理，避免重复下单

### 4. Dashboard 可视化

项目现在带了一个轻量监控面板，数据来源不是另起一套数据库，而是直接读取实盘循环写下来的运行快照：

- 前端：React + Vite，目录 `dashboard-ui/`
- 后端：`python -m run.dashboard_server`
- 快照来源：
  - `logs/runtime_dashboard_status.json`
  - `logs/runtime_dashboard_history.json`
  - `logs/runtime_dashboard_baseline.json`

页面包含的主要内容：

- 当前运行状态：是否运行中、最近处理 bar、最近执行、轮询状态
- 账户收益状态：总权益、净收益、收益率、回撤、可用保证金率
- 仓位状态：方向、仓位规模、名义敞口、估算浮盈
- 图表：净收益曲线、账户权益曲线、价格曲线、仓位曲线
- 最近策略事件流：新 bar、心跳、开平仓、调仓、异常

### 5. 真实成交审计 / 每日复盘

实盘监控现在会在成功成交后记录 OKX 返回的订单与成交明细，并自动刷新当天日报：

- 原始成交 JSONL：`logs/live_fills.jsonl`
- 每日报告 JSON：`logs/daily_reports/YYYY-MM-DD.json`
- 每日报告 Markdown：`logs/daily_reports/YYYY-MM-DD.md`
- 最新日报快捷入口：`logs/daily_report_latest.md`

每条成交记录会包含：

- OKX 订单号、`clOrdId`、订单状态、方向、仓位方向
- 实际成交均价、成交数量、名义金额
- 交易所返回手续费；如果交易所订单没有返回手续费，则按 `FEE_RATE` 估算并标记来源
- 相对策略参考价的滑点
- 开平仓 / 调仓原因、信号快照、策略决策快照
- 交易前后权益、仓位、均价
- 对平仓和减仓交易计算毛实现 PnL 与扣费后净实现 PnL

每日复盘会按动作、原因、多空方向分别归因，方便判断当天亏损来自模型方向、TP/SL、反向平仓、rebalance、手续费还是滑点。

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

### 推荐线上测试盘 `.env`

下面这份参数偏保守，目标是降低日志中暴露出的高频 rebalance、手续费磨损和模型偏多时的逆势开仓问题。线上已有 `.env` 时，可以按需覆盖同名字段；`OKX_API_KEY`、`OKX_SECRET`、`OKX_PASSWORD`、`TELEGRAM_BOT_TOKEN` 等密钥不要提交到 Git。

```env
# 是否使用模拟盘 (0为实盘，1为模拟盘)
USE_SERVER=1

# 交易参数
SYMBOL=SOL-USDT-SWAP
LEVERAGE=3
POSITION_SIZE=50

# 多周期
INTERVALS=5m,15m,1H
WINDOWS=5m:15000,15m:6000,1H:2000

# 风控参数
TAKE_PROFIT=0.02
STOP_LOSS=0.01

# 策略阈值
THRESHOLD_LONG=0.56
THRESHOLD_SHORT=0.56

# 合约配置
LOT_SIZE=0.01
TICK_SIZE=0.001

# 模型路径
MODEL_PATH=models/model_okx.pkl

# 信号平滑参数
SMOOTH_ALPHA=0.3

# 模型文件路径
MODEL_PATHS=lgb_v1:models/lgb_model.pkl,xgb_v1:models/xgb_model.pkl,rf_v1:models/rf_model.pkl

# 模型权重
MODEL_WEIGHTS=lgb_v1:0.5,xgb_v1:0.3,rf_v1:0.2

# 训练/验证/OOS 样本切分
TRAINING_METADATA_PATH=models/training_metadata.json
MODEL_LABEL_FUTURE_WINDOW=8
MODEL_LABEL_THRESHOLD=0.004
MODEL_TRAIN_TRADABLE_LABELS=0
MODEL_TRAIN_NO_TRADE_LABELS=1
MODEL_USE_RUBIK_FEATURES=0
MODEL_RUBIK_PERIOD=1H
MODEL_RECENT_SAMPLE_WEIGHT_BOOST=0.15
MODEL_TRADE_SAMPLE_WEIGHT_MULTIPLIER=1.0
MODEL_NO_TRADE_SAMPLE_WEIGHT_MULTIPLIER=1.0
MODEL_SAMPLE_WEIGHT_MIN=0.25
MODEL_SAMPLE_WEIGHT_MAX=20.0
MODEL_TRAIN_RATIO=0.70
MODEL_VALIDATION_RATIO=0.15
MODEL_PURGE_BARS=8
MODEL_FINAL_TRAIN_ON_VALIDATION=1
MODEL_WALK_FORWARD_ENABLED=1
MODEL_WALK_FORWARD_FOLDS=3
MODEL_WALK_FORWARD_MIN_FOLDS=2
MODEL_WALK_FORWARD_MIN_VALIDATION_ROWS=100

# 仓位边界
POSITION_MIN=0.08
POSITION_MAX=0.45
BASE_POSITION_RATIO=0.12
MAX_POSITION_RATIO=0.45
MIN_ADJUST_AMOUNT=150
ADJUST_UNIT=50
TRADE_COOLDOWN_BARS=6
ADD_THRESHOLD=0.25
MAX_REBALANCE_RATIO=0.25
BLOCK_LOSING_POSITION_ADDS=1
SIGNAL_MIN_PROB_DIFF=0.12
MIN_SIGNAL_TARGET_RATIO=0.04
REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO=0.05
REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO=0.05
REVERSE_SIGNAL_MIN_PROB_DIFF=0.18
REVERSE_MIN_TARGET_RATIO=0.08
REVERSE_EXIT_CONSECUTIVE_BARS=2
REVERSE_EXIT_MIN_PROB_DIFF=0.28

# Kelly 盈亏比
KELLY_REWARD_RISK=2.8

# 动态风险预算目标波动
TARGET_VOL=0.02
MIN_HOLD_BARS=4
TRAILING_EXIT=0.02
DYNAMIC_RISK_ENABLED=0
DYNAMIC_LEVERAGE_MIN=1
DYNAMIC_LEVERAGE_MAX=3
DYNAMIC_POSITION_MAX=0.45
DYNAMIC_RISK_HIGH_VOL_MULTIPLIER=0.65
DYNAMIC_RISK_LOW_SIGNAL_MULTIPLIER=0.60
DYNAMIC_RISK_TREND_MISMATCH_MULTIPLIER=0.50
DYNAMIC_RISK_STRONG_SIGNAL_THRESHOLD=0.30
DYNAMIC_RISK_WEAK_SIGNAL_THRESHOLD=0.16

MAX_POSITION=0.4
INITIAL_BALANCE=1000
BACKTEST_MIN_ADJUST_AMOUNT=40

# 手续费/滑点感知
FEE_RATE=0.0005
BACKTEST_INTRABAR_TP_SL=0

# TP/SL
ADAPTIVE_TP_SL_ENABLED=1
ATR_TAKE_PROFIT_MULTIPLIER=5.0
ATR_STOP_LOSS_MULTIPLIER=2.2
VOLATILITY_TAKE_PROFIT_MULTIPLIER=7.0
VOLATILITY_STOP_LOSS_MULTIPLIER=2.6
ADAPTIVE_TAKE_PROFIT_MIN=0.009
ADAPTIVE_TAKE_PROFIT_MAX=0.045
ADAPTIVE_STOP_LOSS_MIN=0.0055
ADAPTIVE_STOP_LOSS_MAX=0.025
REGIME_HIGH_VOL_STOP_LOSS_MIN=0.009

# 自动重训准入
MODEL_RETRAIN_ENABLED=1
MODEL_RETRAIN_HOUR=3
MODEL_RETRAIN_MINUTE=10
MODEL_RETRAIN_INTERVAL_HOURS=24
MODEL_RETRAIN_VALIDATE_BACKTEST=1
MODEL_RETRAIN_MIN_RETURN_PCT=0.0
MODEL_RETRAIN_MAX_DRAWDOWN_PCT=-5.0
MODEL_RETRAIN_MIN_CLOSED_TRADES=10
MODEL_RETRAIN_MIN_WIN_RATE_PCT=45.0
MODEL_RETRAIN_MIN_PROFIT_FACTOR=1.05
MODEL_RETRAIN_MIN_AVG_WIN_LOSS_RATIO=0.8
MODEL_RETRAIN_MIN_NET_PNL_AFTER_COSTS=0.0
MODEL_RETRAIN_MIN_OOS_ROWS=100
MODEL_RETRAIN_REGIME_GATE_ENABLED=1
MODEL_RETRAIN_REGIME_GATE_MIN_ROWS=30
MODEL_RETRAIN_MAX_TREND_SHORT_LONG_DOMINANCE_PCT=80.0
MODEL_RETRAIN_MAX_TREND_LONG_SHORT_DOMINANCE_PCT=80.0
MODEL_RETRAIN_RESTART_LIVE_MONITOR=1
MODEL_RETRAIN_KEEP_BACKUPS=5

# Regime 分层过滤
REGIME_FILTER_ENABLED=true
REGIME_TREND_GAP_THRESHOLD=0.003
REGIME_HIGH_VOL_ATR_THRESHOLD=0.0016
REGIME_HIGH_VOLATILITY_THRESHOLD=0.0012
REGIME_MONEY_FLOW_EXTREME_THRESHOLD=1.8
REGIME_RANGE_ALLOW_TRADES=true
REGIME_HIGH_VOL_ALLOW_TRADES=false
REGIME_RANGE_THRESHOLD_BONUS=0.04
REGIME_HIGH_VOL_THRESHOLD_BONUS=0.06
REGIME_TREND_AGAINST_BLOCK=true
REGIME_RANGE_TARGET_MULTIPLIER=0.60
REGIME_HIGH_VOL_TARGET_MULTIPLIER=0.35

# 实盘/模拟盘保护；默认强制模拟盘，防止复制示例后误连实盘。
LIVE_REQUIRE_SIMULATED_TRADING=1
LIVE_AUTO_SET_POSITION_MODE=1
LIVE_AUTO_SET_LEVERAGE=1
LIVE_RECONCILE_PENDING_ORDERS=1
LIVE_PERSIST_LAST_BAR=1
LIVE_USE_AVAILABLE_MARGIN_FOR_SIZING=1
LIVE_MARGIN_USAGE_RATIO=0.85
LIVE_MIN_FREE_MARGIN_USDT=30
LIVE_REWARD_RISK_MIN=1.2

# Telegram 通知
TELEGRAM_BOT_TOKEN=这里填你的token
TELEGRAM_CHAT_ID=这里填你的chat_id
TELEGRAM_ENABLED=0
TELEGRAM_LOOP_ERROR_NOTIFY_THRESHOLD=3
DAILY_REPORT_HOUR=23
DAILY_REPORT_MINUTE=59

TRAILING_STOP=0.03
MAX_HOLD_BARS=96

POLL_SEC=10
```

参数含义重点：

- `WINDOWS=5m:15000,15m:6000,1H:2000`：拉长训练窗口；模型输入已平稳化，跨价位段样本更可复用。
- `MODEL_LABEL_FUTURE_WINDOW=8` / `MODEL_LABEL_THRESHOLD=0.004`：当前标签口径来自 OOS 扫描，目标是避开 5m 细碎噪声。
- `MODEL_TRAIN_TRADABLE_LABELS=0` / `MODEL_TRAIN_NO_TRADE_LABELS=1`：训练标签保留原始方向和 no_trade 质量标签；regime/trend 过滤留在交易决策层执行。
- `MODEL_USE_RUBIK_FEATURES=0`：Rubik OI/taker/多空比特征默认关闭。开启前应重新训练并用 OOS 回测确认收益质量。
- `THRESHOLD_LONG=0.56` / `THRESHOLD_SHORT=0.56` / `SIGNAL_MIN_PROB_DIFF=0.12`：当前示例阈值配合新版标签和模型概率尺度，复制旧模型时不要盲目套用。
- `REGIME_HIGH_VOL_ALLOW_TRADES=false` / `REGIME_TREND_AGAINST_BLOCK=true`：高波动趋势区间不放行逆势交易，尤其防止 `trend_short` 中强行开多。
- `MIN_ADJUST_AMOUNT=150` / `BACKTEST_MIN_ADJUST_AMOUNT=40`：实盘最小调仓和回测最小调仓分开配置，避免小额回测收益被线上最小交易额吞掉。
- `MAX_POSITION_RATIO=0.45` / `LEVERAGE=3`：示例仍是测试盘参数，真实资金前要重新按账户权益调小。
- `DYNAMIC_RISK_ENABLED=0`：新版示例先关闭动态风险缩放，避免和阈值校准同时改变；需要时再单独 A/B。
- `MODEL_RECENT_SAMPLE_WEIGHT_BOOST=0.15`：训练不随机下采样；样本权重先平衡 long/short/no_trade，再在方向内部按 regime 平衡，并轻微提高近期样本权重。
- `MODEL_TRAIN_RATIO` / `MODEL_VALIDATION_RATIO` / `MODEL_PURGE_BARS`：训练区、验证区、最终 OOS 回测区按时间切开，中间留 purge gap。
- `MODEL_FINAL_TRAIN_ON_VALIDATION=1`：验证指标仍来自严格时间切分；最终保存的模型用 OOS 之前的 train+validation 历史段重训，减少线上模型滞后。
- `MODEL_WALK_FORWARD_ENABLED=1`：重训准入前会在验证区内做滚动 walk-forward 验证。
- `MODEL_RETRAIN_MIN_*` / `MODEL_RETRAIN_REGIME_GATE_*`：候选模型必须满足 OOS 交易数、胜率、PF、平均盈亏比、手续费后收益和 regime 偏置门槛；不达标会保留旧模型并回滚。
- `TELEGRAM_ENABLED=1`：只播报重要事件，包括开平仓/调仓成交、重训成功或失败回滚、连续实盘异常、手动生成的每日复盘汇总；普通心跳和一般日志只写本地文件。

改完线上 `.env` 后，需要重启 PM2 进程才会生效：

```bash
pm2 restart quant_okx_paper
pm2 restart quant_okx_dashboard
```

## 常用命令

### 1. 训练模型

```bash
python -m train.train
```

### 2. 诊断模型概率

```bash
python -m run.diagnose_model_probs --bars 1500
```

这个脚本会输出最近样本的 long/short 概率分布、方向命中率、预测多空占比，以及当前阈值下通过 long/short gate 的 bar 数量。它适合在每次改特征、标签或重训后先看模型是否又学偏了。

### 3. 跑回测

```bash
python -m backtest.backtest
```

### 4. 扫描自适应 TP/SL 参数

```bash
python -m run.tune_adaptive_tp_sl
```

脚本会输出多组候选参数的回测对比，并给出 `recommended_overrides`。

### 5. 校准交易阈值和概率

```bash
python -m run.calibrate_trade_thresholds --split oos --asymmetric \
  --long-thresholds 0.45,0.50,0.55,0.60 \
  --short-thresholds 0.40,0.45,0.50,0.55,0.60 \
  --gaps 0.04,0.08,0.12 \
  --min-target-ratios 0.01,0.02,0.04
```

脚本会把 Brier、ECE、概率分桶、候选阈值回测结果写入 `logs/trade_threshold_calibration_*.json`。如需验证概率校准器，可额外加：

```bash
--probability-calibration sigmoid --probability-calibration-source validation
```

概率校准器必须先用 validation 拟合，再在 OOS 上验证；不要用 OOS 反向拟合。最近一轮验证里 raw 概率优于 isotonic/sigmoid，因此当前默认不启用概率校准，只把它作为诊断工具。

### 6. 测试盘启动前预检

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

### 7. 启动本地测试盘监控

```bash
PYTHONPATH=. TELEGRAM_ENABLED=0 python -m run.live_trading_monitor
```

### 8. 启动守护调度器

```bash
python -m run.scheduler
```

调度器逻辑：

- 到达 `MODEL_RETRAIN_HOUR` / `MODEL_RETRAIN_MINUTE` 后触发自动重训；候选模型需通过 walk-forward、OOS 回测和 regime 偏置门禁才会替换旧模型
- 到达 `DAILY_REPORT_HOUR` / `DAILY_REPORT_MINUTE` 后生成每日交易复盘
- 其他时间确保 `run.live_trading_monitor` 常驻

### 9. 本地启动 Dashboard

先启动 Python 数据接口：

```bash
PYTHONPATH=. .venv/bin/python -m run.dashboard_server
```

如果你想本地开发 React 页面：

```bash
cd dashboard-ui
npm install
npm run dev
```

默认会把 `/api` 代理到 `http://127.0.0.1:8787`。

如果你只想构建静态页面并交给 Python server 托管：

```bash
cd dashboard-ui
npm install
npm run build
cd ..
PYTHONPATH=. .venv/bin/python -m run.dashboard_server
```

访问：

- Dashboard 页面：`http://127.0.0.1:8787`
- 健康检查：`http://127.0.0.1:8787/api/health`
- 原始数据：`http://127.0.0.1:8787/api/dashboard`

## 测试盘联调建议

推荐按这个顺序执行：

1. 先确认 `.env` 使用的是模拟盘密钥，且 `USE_SERVER=1`
2. 先跑 `run/check_okx_paper_ready.py`
3. 手动确认没有遗留仓位和挂单
4. 用最小仓位启动 `run.live_trading_monitor`
5. 观察开仓、加减仓、平仓、重启恢复是否都正常
6. 确认 `logs/live_trading.log` 和 `logs/live_trading_state.json` 正常更新

## VPS 部署

### 1. 推荐：一条命令部署测试盘

```bash
bash run/deploy_paper_vps.sh
```

如果你希望部署前顺手把服务器上的当前分支 fast-forward 到最新代码，可以用：

```bash
bash run/deploy_paper_vps.sh --git-pull
```

这个脚本会：

- 可选执行 `git pull --ff-only`
- 自动安装缺失的系统依赖
- 自动补齐 `python3-venv` / `python3-pip` / `nodejs` / `npm` / `pm2`
- 校验 `.env` 是否存在且为 OKX 测试盘配置
- 检查模型文件是否齐全
- 自动修复损坏的 `.venv`
- 安装依赖
- 构建 `dashboard-ui/dist`
- 执行 `run/check_okx_paper_ready.py`
- 用 PM2 启动或重载：
  - `quant_okx_paper`
  - `quant_okx_dashboard`

说明：

- 脚本默认会尝试通过系统包管理器安装缺失组件
- 当前支持 `apt-get`、`dnf`、`yum`、`apk`
- 需要 `root` 或 `sudo` 权限

常用参数：

- `--git-pull`: 部署前拉最新代码
- `--skip-check`: 跳过测试盘预检
- `--skip-start`: 只安装和校验，不启动 PM2

### 1.1 从本机一条命令同步并远程部署到 VPS

如果你的本机可以直接 `ssh/rsync` 到 VPS，可以直接用：

```bash
bash run/deploy_remote_vps.sh --host 185.214.135.24
```

如果这次还需要把本地 `.env` 和 `models/` 一起传上去：

```bash
bash run/deploy_remote_vps.sh --host 185.214.135.24 --sync-env --sync-models
```

说明：

- 这个脚本会先用 `rsync` 同步代码到 VPS
- 默认不会覆盖 VPS 上已有的 `.env` 和 `models/`
- 同步后会自动在远端执行 `bash run/deploy_paper_vps.sh`
- 如果是密码登录，`ssh/rsync` 会正常提示你输入密码

常用参数：

- `--user root`: 指定远端用户
- `--port 22`: 指定 SSH 端口
- `--remote-dir /root/quant_sol_project`: 指定远端项目目录
- `--sync-env`: 同步本地 `.env`
- `--sync-models`: 同步本地 `models/`
- `--skip-check`: 远端跳过测试盘预检
- `--skip-start`: 远端安装完但不启动 PM2
- `--skip-deploy`: 只同步文件，不执行远端部署脚本

### 1.2 使用 GitHub Actions 自动部署到 VPS

仓库已包含 `.github/workflows/deploy-vps.yml`。当 `main` 分支有 push，或你在 GitHub 页面手动运行 `Deploy VPS` workflow 时，会自动通过 SSH 同步代码到 VPS，并执行：

```bash
bash run/deploy_paper_vps.sh
```

GitHub Actions 默认不会上传 `.env`、`models/`、`logs/`，避免把交易密钥和模型产物放进 GitHub。第一次使用前，需要先在 VPS 上准备好 `.env` 和 `models/`。

在 GitHub 仓库页面进入 `Settings -> Secrets and variables -> Actions -> New repository secret`，添加：

```text
VPS_HOST=你的VPS IP或域名
VPS_SSH_KEY=用于登录VPS的SSH私钥
```

可选：

```text
VPS_USER=root
VPS_PORT=22
VPS_PROJECT_DIR=/root/quant_sol_project
```

建议专门生成一把部署密钥，不要直接复用你的个人主密钥：

```bash
ssh-keygen -t ed25519 -C "github-actions-quant-sol" -f ~/.ssh/quant_sol_actions
ssh-copy-id -i ~/.ssh/quant_sol_actions.pub root@你的VPS_IP
```

然后把 `~/.ssh/quant_sol_actions` 的内容填入 `VPS_SSH_KEY`。

### 2. 旧版基础引导脚本

如果你只想先建 Python 环境，不立即启动服务，也可以继续用：

```bash
bash run/bootstrap_vps.sh
```

### 3. 手动安装 PM2（仅当你不想让脚本自动处理时）

```bash
npm install -g pm2
```

### 4. 手动做测试盘预检

```bash
PYTHONPATH=. TELEGRAM_ENABLED=0 .venv/bin/python run/check_okx_paper_ready.py
```

### 5. 手动用 PM2 常驻测试盘

```bash
pm2 start ecosystem.paper.config.js
pm2 save
```

PM2 里默认会一起跑：

```bash
.venv/bin/python -m run.live_trading_monitor
.venv/bin/python -m run.dashboard_server
```

### 6. 查看日志

```bash
pm2 logs quant_okx_paper
pm2 logs quant_okx_dashboard
tail -f logs/live_trading.log
tail -f logs/scheduler.log
```

### 7. VPS 查看 Dashboard

如果 `deploy_paper_vps.sh` 已经跑完，并且 PM2 里 `quant_okx_dashboard` 是 `online`，那么默认可以直接访问：

```text
http://你的VPSIP:8787
```

你也可以先在服务器本机确认：

```bash
curl http://127.0.0.1:8787/api/health
curl http://127.0.0.1:8787/api/dashboard
```

### 8. 只看最近策略状态摘要

默认查看最近 60 条关键状态：

```bash
bash run/strategy_status_summary.sh
```

实时追踪关键状态：

```bash
bash run/strategy_status_summary.sh --follow
```

### 9. 手动生成每日成交复盘

实盘成交后会自动刷新当天日报；如果你想手动重建某一天的报告，可以运行：

```bash
PYTHONPATH=. .venv/bin/python -m run.daily_trade_report --date 2026-05-05
```

不传 `--date` 时默认生成今天的报告：

```bash
PYTHONPATH=. .venv/bin/python -m run.daily_trade_report
```

## 日志与状态文件

- `logs/live_trading.log`: 交易监控主日志
- `logs/live_fills.jsonl`: 真实成交结构化记录
- `logs/daily_reports/YYYY-MM-DD.md`: 每日成交复盘 Markdown
- `logs/daily_reports/YYYY-MM-DD.json`: 每日成交复盘 JSON
- `logs/daily_report_latest.md`: 最新每日复盘快捷入口
- `logs/scheduler.log`: 调度器日志
- `logs/live_trading_state.json`: 最近处理 bar 的持久化状态
- `logs/runtime_dashboard_status.json`: 当前 dashboard 状态快照
- `logs/runtime_dashboard_history.json`: dashboard 曲线历史
- `logs/runtime_dashboard_baseline.json`: Dashboard 收益基线
- `logs/backtest_*.csv`: 回测交易记录与汇总
- `logs/training_diagnostics_*.json`: regime 分层分类指标、混淆矩阵、信号方向占比与回测诊断
- `logs/trade_threshold_calibration_*.json`: 阈值 sweep、Brier/ECE、概率分桶和候选交易质量报告

## 单元测试

建议至少跑这几组核心测试：

```bash
python -m unittest \
  tests.test_backtest_equity \
  tests.test_backtest_intrabar \
  tests.test_strategy_core \
  tests.test_live_runtime_state \
  tests.test_runtime_dashboard \
  tests.test_trade_audit \
  tests.test_stationary_features \
  tests.test_rubik_features \
  tests.test_calibrate_trade_thresholds
```

这些测试主要覆盖：

- 盯市净值计算
- 同 bar 内 TP/SL 处理
- 自适应止盈止损阈值
- `clOrdId` 辅助逻辑
- `last_bar_ts` 持久化与恢复
- 真实成交记录与每日复盘汇总
- 平稳特征不会泄漏绝对价格水位进模型
- Rubik 特征无前视、默认关闭、启用后只进入平稳派生列
- 阈值/概率校准报告可复现且写入安全

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
