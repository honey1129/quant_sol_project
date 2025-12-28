import os
from datetime import datetime
import csv
import joblib
import traceback
from core.strategy_core import StrategyCore
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from core import position_manager, okx_api, ml_feature_engineering, signal_engine
from config import config
from core.reward_risk import RewardRiskEstimator
from utils.utils import log_info, log_error,LOGS_DIR


class Backtester:
    def __init__(self, interval, window):
        self.interval = interval
        self.window = window
        self.in_high_conf = False
        self.hold_bars = 0
        self.peak_price = None

        # 拉取多周期数据以及计算reward_risk
        self.data_dict,self.reward_risk = self._load_data()

        # 特征工程
        merged_df = ml_feature_engineering.merge_multi_period_features(self.data_dict)
        merged_df = ml_feature_engineering.add_advanced_features(merged_df)
        self.data = merged_df

        # 读取训练时的特征列表
        self.feature_cols = joblib.load(config.FEATURE_LIST_PATH)

        # 加载模型与权重
        self.models = signal_engine.load_models(config.MODEL_PATHS)
        self.model_weights = config.MODEL_WEIGHTS

        # 初始化仓位和资金
        self.position = 0
        self.entry_price = 0
        self.balance = config.INITIAL_BALANCE
        self.max_balance = self.balance
        self.trade_log = []
        self.fee_rate = config.FEE_RATE

        # 初始化 position_manager
        self.position_manager = position_manager.PositionManager()
        # ✅ 统一核心策略（以回测为准）
        self.core = StrategyCore(
            self.position_manager,
            threshold_long=config.THRESHOLD_LONG,
            threshold_short=config.THRESHOLD_SHORT,
            take_profit=config.TAKE_PROFIT,
            stop_loss=config.STOP_LOSS,
            min_hold_bars=config.MIN_HOLD_BARS,
            add_threshold=config.ADD_THRESHOLD,
            max_rebalance_ratio=config.MAX_REBALANCE_RATIO,
            min_adjust_amount=float(config.MIN_ADJUST_AMOUNT),
            reward_risk=float(self.reward_risk),
        )

    def _load_data(self):
        log_info(f"从OKX拉取历史数据: {self.interval}, {self.window}根K线")
        client = okx_api.OKXClient()
        trades = client.fetch_recent_closed_trades()
        rr_estimator=RewardRiskEstimator()
        rr_estimator.batch_update(trades)
        reward_risk = rr_estimator.estimate()

        all_data = client.fetch_data()
        return all_data,reward_risk


    def run_backtest(self):

        # ========== 预计算信号 ==========
        self.data[['long_prob', 'short_prob']] = self.data.apply(
            self._predict_row, axis=1, result_type="expand"
        )

        for i in tqdm(range(len(self.data))):
            row = self.data.iloc[i]
            price = row['5m_close']
            long_prob = row['long_prob']
            short_prob = row['short_prob']
            money_flow_ratio = row['money_flow_ratio']
            volatility = row['volatility_15']

            if pd.isna(volatility):
                continue
            volatility = float(volatility)

            # ===== 将回测状态同步进 core =====
            self.core.set_state(self.position, self.entry_price, self.hold_bars)

            out = self.core.on_bar(
                price=price,
                equity=self.balance,
                long_prob=long_prob,
                short_prob=short_prob,
                money_flow_ratio=money_flow_ratio,
                volatility=volatility,
            )

            action = out["action"]
            delta = float(out["delta_qty"])

            if action == "CLOSE":
                pos_to_close = self.position
                entry_price = self.entry_price

                profit = (price - entry_price) * pos_to_close
                self.balance += profit - abs(pos_to_close * price * self.fee_rate)

                act = "平仓" if out.get("reason") == "TP/SL" else "反向平仓"
                self.trade_log.append((row.name, act, price, pos_to_close, self.balance))

            elif action == "OPEN":
                new_pos, _, _ = self.core.get_state()
                self.balance -= abs(new_pos * price * self.fee_rate)

                act = "开多" if new_pos > 0 else "开空"
                self.trade_log.append((row.name, act, price, new_pos, self.balance))

            elif action == "REBALANCE":
                new_pos, _, _ = self.core.get_state()
                self.balance -= abs(delta * price * self.fee_rate)

                act = "加多" if delta > 0 else "减多"
                self.trade_log.append((row.name, act, price, new_pos, self.balance))

            self.position, self.entry_price, self.hold_bars = self.core.get_state()
            self.max_balance = max(self.max_balance, self.balance)


        self._summary()

    def _predict_row(self, row):
        """
        复用实盘信号融合逻辑，保持一致性
        """
        X_row = row[self.feature_cols].values.reshape(1, -1).astype(float)
        X_row = pd.DataFrame(X_row, columns=self.feature_cols)

        weighted_sum = np.zeros(2)
        total_weight = sum(self.model_weights.values())

        for name, model in self.models.items():
            prob = model.predict_proba(X_row)[0]
            weight = self.model_weights.get(name, 1.0)
            weighted_sum += prob * weight

        avg_pred = weighted_sum / total_weight
        long_prob, short_prob = avg_pred[1], avg_pred[0]
        return long_prob, short_prob

    def _summary(self):
        pnl = self.balance - config.INITIAL_BALANCE
        drawdown = (self.balance - self.max_balance) / self.max_balance

        log_info("回测完成 ✅")
        log_info(f"最终资金: {self.balance:.2f} USDT")
        log_info(f"累计收益: {pnl:.2f} USDT ({pnl / config.INITIAL_BALANCE * 100:.2f}%)")
        log_info(f"最大回撤: {drawdown * 100:.2f}%")
        log_info(f"交易次数: {len(self.trade_log)}")
        log_info(f"交易记录示例: {self.trade_log[-5:]}")
        self.dump_trade_log_to_csv(pnl,drawdown)

    def dump_trade_log_to_csv(self,pnl, drawdown):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backtest_log_path = os.path.join(LOGS_DIR, f"backtest_{self.interval}_{ts}.csv")
        with open(backtest_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "action", "price",  "position","balance"])

            writer.writerows([
                (t, a, round(p, 4), round(pos, 4),round(b, 2))
                for (t, a, p,pos, b) in self.trade_log
            ])
            writer.writerow([])
            writer.writerow(["# Summary"])
            writer.writerow(["Final Balance", round(self.balance, 2)])
            writer.writerow(["PnL (USDT)", round(pnl, 2)])
            writer.writerow([
                "Return (%)",
                round(pnl / config.INITIAL_BALANCE * 100, 2)
            ])
            writer.writerow([
                "Max Drawdown (%)",
                round(drawdown * 100, 2)
            ])
            writer.writerow([
                "Trade Count",
                len(self.trade_log)
            ])



if __name__ == '__main__':
    try:
        for interval in config.INTERVALS:
            window = config.WINDOWS.get(interval, 1000)
            log_info(f"\n==== 开始 {interval} 周期回测 ====")
            backtester = Backtester(interval, window)
            backtester.run_backtest()
            time.sleep(1)
    except Exception as e:
        log_error(traceback.format_exc())
