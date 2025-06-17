# Quant SOL Project (V2.2 Pro Version)

> 🚀 **多周期、多模型、多因子融合的 OKX 永续合约量化交易系统**

---

## 📦 项目简介

本项目是基于 SOL/USDT 保证金合约市场，使用机器学习（LightGBM）、多周期特征融合、资金流+波动率因子、动态仓位管理（Kelly+多因子）的完整量化交易系统。

✅ 支持回测与实盘自动交易  
✅ 支持多周期数据拉取（5m、15m、1H）  
✅ 支持模型集成与平滑信号  
✅ 支持 OKX API 全自动下单

---

## 🗂 项目目录结构

```
quant_sol_project/
│
├── core/                  # 核心模块
│   ├── okx_api.py         # OKX API封装模块
│   ├── ml_feature_engineering.py  # 多周期特征工程与衍生特征生成
│   ├── position_manager.py  # 动态仓位管理模块 (Kelly + 波动率 + 资金流融合)
│   ├── signal_engine.py   # 多模型融合信号引擎
│
├── train/                 # 训练模块
│   └── train.py           # 完整训练流程
│
├── backtest/              # 回测模块
│   └── backtest.py        # 完整回测流程
│
├── utils/                 # 工具模块
│   └── utils.py           # 日志、Telegram通知、路径配置等
│
├── config/                # 配置模块
│   └── config.py          # 全局配置项，支持 .env 环境变量动态配置
│
├── models/                # 训练后的模型文件及特征列表
│   ├── model_okx.pkl
│   └── feature_list.pkl
│
├── logs/                  # 运行日志
│
├── .env                   # 私密配置 (API KEY, 参数, 策略阈值等)
├── .gitignore             # Git忽略文件
└── README.md              # 项目说明文档（本文件）
```

---

## 🔧 运行环境配置

### 推荐使用 Python 3.9+


### 1️⃣ 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```
### 2️⃣ 配置 .env 文件

```bash
copy .env.example .env
```

---

## 🚀 模型训练
直接运行：


```bash
python train/train.py
```
* 会自动拉取多周期数据，完成特征工程，模型训练与保存
* 训练后的模型文件保存至 models/model_okx.pkl
* 同时保存特征列表 models/feature_list.pkl
---
## 📊 策略回测
直接运行：
```bash
python -m backtest.backtest
```
* 支持多周期全量回测
* 自动加载训练好的模型和特征
* 回测结果含资金曲线、收益、回撤等指标
* 
---
## 🟢 实盘执行

```bash
python run/live_trading_monitor.py
```
---


## ⚠ 注意事项

- 本项目仅供学习与研究用途，请勿直接在实盘大资金环境下使用！
- 请务必做好 API 密钥与资金安全隔离！
- 强烈建议在云服务器测试好逻辑后再投入正式运行。

---

## 📌 后续优化方向

- ✅ 多模型融合 (LightGBM、XGBoost、RandomForest)（现已支持）
- ✅ 多周期特征支持（5m、15m、1h、4h 等）(现已支持）
- ✅ 智能仓位动态管理 (现已支持）
- ⏳ 支持指标监控（资金流、波动率、仓位等）
- ⏳ 可视化监控模块（资金曲线/信号走势）

