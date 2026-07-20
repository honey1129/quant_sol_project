# Quant Sol Project

基于 OKX SOL-USDT-SWAP 永续合约的量化交易系统，覆盖 ML 模型训练、回测验证、自动重训、测试盘联调与 VPS 常驻运行。

## 核心特性

- **多周期特征工程**：5m / 15m / 1H，特征平稳化处理，避免绝对价格泄漏到模型
- **二分类交易质量模型**：LightGBM + XGBoost + RandomForest 集成，预测"当前方向是否值得入场"
- **方向质量子模型**：long / short 各自独立训练和校准，避免多空信号混淆
- **执行概率变换**：基于基准 trade rate 的 odds-lift 映射，将稀疏正类概率转换为策略执行概率
- **市场状态分类（Regime）**：规则based分类（trend_long/trend_short/range/range_high_vol），动态调整阈值与仓位
- **自适应 TP/SL**：基于 ATR 和波动率动态调整，结合 TP/SL 比例下限保证期望值
- **多层风控守卫**：LossGuard / LongEntryGuard / RegimeFilter / TrendFilter 分层拦截
- **交易所端 TP/SL 保护**：每次开仓后立即在 OKX 下 OCO 算法止盈止损单（mark price 触发），进程崩溃时止损仍有效
- **账户级熔断**：Kill Switch 文件开关 + 单日最大亏损熔断，触发后只平仓不新开仓
- **动态合约规格**：启动时从 OKX `get_instruments` 读取 lotSz/tickSz/ctVal，避免硬编码精度失效
- **单实例保护**：PID 锁文件防止 PM2 重启窗口内双进程同时下单
- **自动每日重训**：walk-forward 验证 + OOS 回测 + regime 偏置门禁，不达标自动回滚旧模型
- **GitHub Actions CI/CD**：push main 分支自动通过 rsync 部署到 VPS
- **PM2 常驻运行**：进程保活、日志管理、状态持久化
- **Telegram 通知**：开平仓成交、重训结果、每日复盘自动推送

---

## 交易模式

### ML 模式（推荐，默认）

使用三模型集成预测每根 K 线的交易质量概率，经执行概率变换后输出 `long_prob` / `short_prob`：

- `long_prob > THRESHOLD_LONG` 且通过所有守卫 → 开多
- `short_prob > THRESHOLD_SHORT` 且通过所有守卫 → 开空
- 否则持平观望

**注意**：ML 模式需要先训练模型，且模型质量直接决定信号频率。若模型正类召回率接近 0%，系统会长时间不开仓，这是正常的保守行为。

### 简单规则模式（仅用于框架验证）

绕过 ML 模型，直接基于 EMA 趋势方向输出固定概率（trend_long→0.9，trend_short→0.1），适合在没有训练数据或模型时先跑通框架：

```env
USE_SIMPLE_RULE_MODE=1
SIMPLE_RULE_POSITION_SIZE=0.15
```

⚠️ **不推荐用于生产**：固定概率无法区分高质量和低质量信号，历史数据显示该模式的止损交易占比极高。

---

## 项目结构

```text
quant_sol_project/
├── .github/workflows/
│   └── deploy-vps.yml         # GitHub Actions 自动部署
├── backtest/
│   ├── backtest.py            # 主回测引擎（盯市净值、手续费、资金费、bar内TP/SL）
│   └── simple_rule_backtest.py
├── config/
│   └── config.py              # 全局配置，200+ 参数均由环境变量控制
├── core/
│   ├── direction_quality.py   # 方向质量子模型与概率校准
│   ├── dynamic_risk.py        # 动态风险控制器
│   ├── ml_feature_engineering.py  # 特征工程（平稳化派生列）
│   ├── okx_api.py             # OKX API 封装
│   ├── position_manager.py    # Kelly 仓位计算
│   ├── predict.py             # 信号推理入口
│   ├── regime_filter.py       # 市场状态分类（规则based）
│   ├── reward_risk.py
│   ├── signal_engine.py       # 多模型加权融合与执行概率变换
│   ├── strategy_core.py       # 策略决策状态机（on_bar）
│   └── trend_filter.py        # 高周期趋势过滤
├── dashboard-ui/              # React + Vite 实时监控面板
├── monitoring/
│   └── hourly_performance_report.py
├── run/
│   ├── calibrate_trade_thresholds.py
│   ├── check_okx_paper_ready.py
│   ├── daily_trade_report.py
│   ├── deploy_paper_vps.sh    # VPS 一键部署脚本
│   ├── deploy_remote_vps.sh   # 本机远程同步+部署
│   ├── live_trading_monitor.py  # 实盘/测试盘主进程
│   ├── market_condition_loss_diagnostics.py
│   ├── retrain_models.py      # 自动重训入口
│   ├── rule_breakout_flow_stability.py
│   └── sweep_trend_filter.py
├── tests/                     # 核心单元测试
├── train/
│   └── train.py               # 模型训练主入口
├── utils/
│   ├── runtime_dashboard.py
│   ├── safe_runner.py
│   ├── trade_audit.py
│   └── utils.py
├── ecosystem.paper.config.js  # PM2 测试盘配置
├── requirements.txt
├── .env.example               # 环境变量模板
└── README.md
```

---

## 策略决策流程

每根 5m K 线收盘后，`strategy_core.py` 的 `on_bar()` 按以下优先级处理：

| 优先级 | 条件 | 动作 |
|--------|------|------|
| 1 | 持仓且 pnl ≥ TP | CLOSE（TakeProfit） |
| 2 | 持仓且 pnl ≤ -SL | CLOSE（StopLoss） |
| 3 | 持仓且 LossGuard 触发 | CLOSE（强制平仓） |
| 4 | 空仓且冷却期 > 0 | HOLD（Cooldown） |
| 5 | 反向强信号 | CLOSE（ReverseClose） |
| 6 | 未达最小持仓时间 | HOLD（MinHold） |
| 7 | 同向信号 + 仓位偏差足够 | REBALANCE（加减仓） |
| 8 | 无信号或信号弱 | HOLD（FlatNoSignal/WeakSignal/SmallTarget） |

**HOLD 原因字段**（`reason`）是诊断信号质量的关键：

- `WeakSignal`：概率未过阈值（`dominant_prob ≤ threshold`）
- `SmallTarget`：Kelly 仓位 < `MIN_SIGNAL_TARGET_RATIO`（信号通过阈值但期望仓位太小）
- `CostGate`：期望净收益为负（概率过了但手续费吃掉了期望值）
- `LongEntryGuard`：多头入场质量保护（趋势强度不足、高波动、资金流过热）
- `LossGuardDirection/Regime`：损失保护封锁指定方向或市场状态

---

## 环境准备

### Python

推荐 Python 3.10+（VPS 已验证 Python 3.12）。

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 初始化配置

```bash
cp .env.example .env
```

**必填项**：

| 配置项 | 说明 |
|--------|------|
| `OKX_API_KEY` / `OKX_SECRET` / `OKX_PASSWORD` | OKX API 凭证（测试盘用模拟盘密钥） |
| `USE_SERVER` | `1`=模拟盘，`0`=实盘。**联调阶段必须用 `1`** |
| `SYMBOL` | 交易标的，默认 `SOL-USDT-SWAP` |
| `LEVERAGE` | 杠杆，建议测试阶段用 `3` |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Telegram 通知（可选） |

**首次运行建议**：

```env
USE_SERVER=1
LIVE_REQUIRE_SIMULATED_TRADING=1
TELEGRAM_ENABLED=0
USE_SIMPLE_RULE_MODE=0
```

---

## 常用命令

### 训练模型

```bash
python -m train.train
```

训练前确认 `.env` 中的关键标签参数（见下方"训练标签配置"节）。

### 回测

```bash
python -m backtest.backtest
```

### 测试盘预检

```bash
PYTHONPATH=. TELEGRAM_ENABLED=0 python run/check_okx_paper_ready.py
```

### 启动本地测试盘

```bash
PYTHONPATH=. TELEGRAM_ENABLED=0 python -m run.live_trading_monitor
```

### 手动重训模型

```bash
PYTHONPATH=. python -m run.retrain_models
```

### 每日成交复盘

```bash
PYTHONPATH=. .venv/bin/python -m run.daily_trade_report
```

### 诊断当前模型信号分布

```bash
python -m run.calibrate_trade_thresholds --split oos --asymmetric
```

---

## 训练标签配置

训练质量直接决定模型能否生成有效信号。当前二分类标签的核心参数：

```env
# 前视窗口与 TP/SL 定义
MODEL_LABEL_LOOKAHEAD_BARS=24       # 最多观察 24 根 5m K线（2小时）
MODEL_LABEL_TAKE_PROFIT=0.016       # 1.6% 即为正样本（TP）
MODEL_LABEL_STOP_LOSS=0.014         # 1.4% 止损

# 超时处理
MODEL_LABEL_TIMEOUT_AS_TRADE=1      # 超时但净收益达标也标为正样本
MODEL_LABEL_TIMEOUT_MIN_NET_RETURN=0.0015  # 超时正样本最低净收益 0.15%

# long:trend_long 方向特殊处理
MODEL_LABEL_LONG_TREND_WEAK_TP_AS_TRADE=0  # 慢速/高回撤 TP 不算正样本
```

**⚠️ 已知问题**：当 TP 定义为 1.6% 时，在低波动震荡行情中正类比例约为 8-10%，模型容易出现 recall ≈ 0%（把所有样本都预测为 no_trade）。如发现 `validation_metrics` 中 trade recall = 0，建议适当降低 `MODEL_LABEL_TAKE_PROFIT`（如改为 `0.010`）以增加正类样本。

---

## 自动重训

每天凌晨 `MODEL_RETRAIN_HOUR:MODEL_RETRAIN_MINUTE`（默认 UTC 03:10）自动触发：

1. 从 OKX 拉取最新历史数据，重新训练候选模型
2. 通过验证集门禁（trade precision、recall 检查）
3. 运行 OOS 回测，检查收益/回撤/盈利因子
4. 通过所有门禁 → 替换线上模型，重启监控进程
5. 任一门禁失败 → 保留旧模型，发送 Telegram 告警

**重训门禁配置**（可通过 `.env` 调整）：

```env
MODEL_RETRAIN_MIN_CLOSED_TRADES=10         # OOS 最少平仓笔数
MODEL_RETRAIN_MIN_PROFIT_FACTOR=1.05       # OOS 最低盈利因子
MODEL_RETRAIN_MIN_VALIDATION_TRADE_PRECISION=0.25  # 验证集交易精度

# 绝对安全底线（只能提高，不能下调）
MODEL_RETRAIN_HARD_MIN_PROFIT_FACTOR=1.0   # 候选模型的盈利因子必须严格大于 1
MODEL_RETRAIN_HARD_MIN_CLOSED_TRADES=1     # 候选模型必须至少有 1 笔平仓交易

# Walk-forward 验证
MODEL_WALK_FORWARD_ENABLED=False           # 当模型 recall ≈ 0 时可关闭以允许部署
```

---

## 安全与风控机制

系统在下单链路上做了多层保护，核心目标是"进程不可用时资金仍受保护"，以及"策略失控时自动停手"。

### 交易所端 TP/SL（进程崩溃保护）

每次开仓成交后，`okx_api.py` 立即通过 `place_algo_order` 在 OKX 下一张 OCO 止盈止损单：

- `reduceOnly=True`，只平仓不反向开仓
- 触发价类型为 **mark price**（`TPSL_TRIGGER_PX_TYPE=mark`），比 last price 抗插针
- 常规平仓 / 反手前先撤销残留算法单；本地实时风控先提交 `reduceOnly` 紧急平仓，成交后再清理 OCO，避免撤单确认阻塞止损
- 若算法单下单失败，会立即市价平掉刚开的仓位（`_close_unprotected_position`），绝不留裸仓

```env
EXCHANGE_TPSL_ENABLED=1              # 开启交易所端 TP/SL
TPSL_TRIGGER_PX_TYPE=mark            # mark / last / index
POLL_SEC=1                           # 本地实时风控目标轮询间隔
RISK_LOOP_WARN_SEC=3                 # 实际检查耗时或相邻间隔超过该值时记录告警
OKX_WEBSOCKET_ENABLED=1              # 实时价格和仓位优先使用 OKX WebSocket
OKX_WEBSOCKET_STALE_SEC=5            # 缓存超过该时长自动降级到 REST
```

运行快照同时记录本地风控检查的最近/最大耗时、相邻间隔和累计慢检查次数。实时价格与仓位由 OKX public/private WebSocket 推送，数据断流或过期时自动降级到 REST。多周期行情拉取与特征计算在独立只读工作线程执行，不阻塞本地实时风控；策略状态更新和交易决策仍由主线程串行执行。持仓期间若实际检查间隔超过阈值，主日志与 Dashboard 观测接口会产生 `risk_loop_latency` 告警；交易所端 OCO 保护不依赖该本地循环。

成交审计会记录触发来源、止盈/止损阈值、检测价格、下单确认耗时、触发到成交耗时，以及检测阶段、订单执行阶段和阈值到成交的分段滑点。每日复盘的最近成交表会直接展示阈值滑点与触发到成交耗时。

### 账户级熔断与 Kill Switch

主循环每根 bar 前检查两个熔断条件（`_check_safety_gates`）：

- **Kill Switch**：检测到 `KILL_SWITCH_FILE` 指定的文件存在时，立即停止所有新开仓
- **单日最大亏损**：当日权益回撤超过 `MAX_DAILY_LOSS_PCT` 时熔断，跨 UTC 日自动重置

熔断触发后只拒绝新开仓，平仓 / 止损照常执行。

```env
MAX_DAILY_LOSS_PCT=0.05                          # 单日最大亏损 5% 触发熔断
KILL_SWITCH_FILE=logs/kill_switch.flag           # 该文件存在即停止开仓
```

紧急停止交易：在 VPS 上 `touch /root/quant_sol_project/logs/kill_switch.flag` 即可，删除文件后恢复。

### 订单一致性保护

- **执行前对账**：`_execute_delta` 下单前重新从 OKX 读取实时仓位，与决策时的仓位不一致则本轮拒绝下单
- **成交超时撤单**：`wait_until_filled` 超时后主动撤销未确认订单，防止悬空单稍后意外成交
- **重启一致性修正**：重启时若 OKX 实时仓位与本地持久化状态不符，重置持仓计数并进入保守冷却
- **clOrdId 幂等**：下单前先按 clOrdId 查询是否已受理，网络超时重试不会重复下单

---

## VPS 部署

### GitHub Actions 自动部署（推荐）

推送到 `main` 分支后自动触发，无需手动操作。

在 GitHub 仓库 `Settings → Secrets → Actions` 中添加：

```
VPS_HOST=<VPS IP>
VPS_SSH_KEY=<SSH 私钥内容>
VPS_USER=root         # 可选，默认 root
VPS_PORT=22           # 可选
VPS_PROJECT_DIR=/root/quant_sol_project  # 可选
```

**重要**：`.env`、`models/`、`logs/` 均被排除在同步之外，这些文件只在 VPS 本地管理，不上传 GitHub。

### 手动一键部署

```bash
# 在 VPS 上执行
bash run/deploy_paper_vps.sh

# 或从本机同步并部署
bash run/deploy_remote_vps.sh --host <VPS_IP>
```

### 更新 VPS 配置参数

直接编辑 VPS 上的 `.env`，然后重启进程（不需要推代码）：

```bash
# SSH 进入 VPS 后
nano /root/quant_sol_project/.env

# 重启并加载新参数
pm2 restart quant_okx_paper --update-env
pm2 restart quant_okx_dashboard --update-env
```

### 查看运行日志

```bash
# 实时日志（推荐）
tail -f /root/quant_sol_project/logs/live_trading.log

# PM2 日志
pm2 logs quant_okx_paper --lines 100

# 每日复盘
cat /root/quant_sol_project/logs/daily_reports/$(date +%Y-%m-%d).md
```

---

## PM2 进程管理

```bash
pm2 list                          # 查看所有进程
pm2 status quant_okx_paper        # 查看单个进程状态
pm2 restart quant_okx_paper --update-env  # 重启并加载最新 .env
pm2 logs quant_okx_paper --lines 50      # 查看最近日志
pm2 save                          # 保存进程列表（重启后自动恢复）
```

---

## Dashboard

项目内置轻量监控面板，读取实盘循环写入的状态快照，无需额外数据库。

**VPS 访问**：
```
http://<VPS_IP>:8787
```

**本地开发**：
```bash
# 后端
PYTHONPATH=. .venv/bin/python -m run.dashboard_server

# 前端（实时开发模式）
cd dashboard-ui && npm install && npm run dev
```

**面板内容**：当前持仓、权益曲线、真实成交价与费用、阈值滑点、触发到成交延迟、K线+仓位叠加图、策略事件流。最近成交优先读取 `logs/live_fills.jsonl`，仅在没有成交审计记录时降级为日志解析。

账户收益与权益曲线使用 USDT 本币权益，避免 `totalEq` 的美元折算汇率波动污染策略盈亏。执行质量告警阈值可通过 `DASHBOARD_EXECUTION_LATENCY_WARN_MS` 和 `DASHBOARD_THRESHOLD_SLIPPAGE_WARN_BPS` 调整，默认分别为 `2000ms` 和 `20bps`；日志回退错误默认仅保留 `300` 秒，可通过 `DASHBOARD_ERROR_MAX_AGE_SEC` 调整。

---

## 关键日志与状态文件

| 文件 | 说明 |
|------|------|
| `logs/live_trading.log` | 主日志，每根 bar 的完整决策诊断 |
| `logs/live_fills.jsonl` | 真实成交结构化记录（含原始订单、PnL、信号快照、触发延迟与分段滑点） |
| `logs/daily_reports/YYYY-MM-DD.md` | 每日成交复盘（按原因、方向归因） |
| `logs/live_trading_state.json` | 持久化状态（最近 bar 时间戳、冷却期） |
| `logs/model_retrain_*.log` | 每次重训详细日志 |
| `logs/runtime_dashboard_*.json` | Dashboard 数据快照 |

---

## 单元测试

```bash
python -m unittest \
  tests.test_backtest_equity \
  tests.test_backtest_intrabar \
  tests.test_strategy_core \
  tests.test_live_runtime_state \
  tests.test_stationary_features \
  tests.test_trade_audit \
  tests.test_rubik_features
```

---

## 已知问题与注意事项

**模型召回率问题**

当前版本在低波动震荡行情中，ML 模型容易因正类样本过少（正类比例约 8%）出现验证集 recall ≈ 0%。表现为：信号概率稳定在 0.30~0.40 附近，触发 `SmallTarget` 或 `CostGate` 而不开仓。这不是 bug，是模型在说"当前信号质量低于手续费门槛"。

解决方向：
1. 降低标签 TP 阈值（`MODEL_LABEL_TAKE_PROFIT=0.010`）以提高正类比例
2. 调整入场阈值（`THRESHOLD_LONG`/`THRESHOLD_SHORT`）与最小仓位（`MIN_SIGNAL_TARGET_RATIO`）

**方向质量模型 pickle 兼容性**

`core/direction_quality.py` 中的 `DirectionQualityModel` 加入了 `__setstate__` 方法，确保旧 pickle 文件（缺少 `direction_regime_calibrators` 属性）在加载时自动补全，不再抛出 `AttributeError`。如遇模型加载报错，检查 Python 版本与 scikit-learn 版本是否兼容。

**单实例保护**

`live_trading_monitor.py` 启动时会写入 `logs/live_trading_monitor.pid` 并检测已有进程，防止 PM2 重启窗口期出现双进程同时下单。若异常退出后 PID 文件残留导致无法启动，手动删除该文件即可。

**参数过多**

`.env` 中有 200+ 配置项，建议每次只调整一个维度（阈值、标签、仓位），不要同时修改多个互相关联的参数。

---

## 风险提示

- 当前为**测试盘（模拟盘）**运行，`LIVE_REQUIRE_SIMULATED_TRADING=1` 是默认保护
- 切换到实盘前，确保在测试盘上稳定运行至少 2 周，并仔细分析每日复盘数据
- 即使策略稳定，仍需关注：交易所接口延迟、滑点扩大、资金费波动、极端行情下的停止损失
- 量化策略的历史回测表现不代表未来收益

---

## 参考链接

- [OKX API Key 管理](https://www.okx.com/account/my-api)
- [OKX API 文档（中文）](https://www.okx.com/docs-v5/zh/#overview)
- [PM2 文档](https://pm2.keymetrics.io/docs/usage/quick-start/)
