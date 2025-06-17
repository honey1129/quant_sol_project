import joblib
import traceback
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from core import position_manager, okx_api, ml_feature_engineering, signal_engine
from config import config
from utils.utils import log_info, log_error

class Backtester:
    def __init__(self, interval, window):
        self.interval = interval
        self.window = window

        # 拉取多周期数据
        self.data_dict = self._load_data()

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

        # 初始化 position_manager 实例（注意：新版）
        self.position_manager = position_manager.PositionManager()

    def _load_data(self):
        log_info(f"从OKX拉取历史数据: {self.interval}, {self.window}根K线")
        client = okx_api.OKXClient()
        all_data = client.fetch_data()
        return all_data

    def run_backtest(self):
        # 预先批量计算所有信号
        self.data[['long_prob', 'short_prob']] = self.data.apply(self._predict_row, axis=1, result_type="expand")

        for i in tqdm(range(len(self.data))):
            row = self.data.iloc[i]
            price = row['5m_close']  # 以5m为执行价格

            money_flow_ratio = row['money_flow_ratio']
            volatility = row['volatility_15']

            # 止盈止损逻辑
            if self.position != 0:
                pnl_pct = (price - self.entry_price) / self.entry_price if self.position > 0 else (self.entry_price - price) / self.entry_price
                if pnl_pct >= config.TAKE_PROFIT or pnl_pct <= -config.STOP_LOSS:
                    profit = (price - self.entry_price) * self.position
                    self.balance += profit - abs(self.position * price * self.fee_rate)
                    self.trade_log.append((row.name, '平仓', price, self.balance))
                    self.position = 0
                    self.entry_price = 0
                    continue

            # 获取预测信号
            long_prob = row['long_prob']
            short_prob = row['short_prob']

            # 动态仓位计算（核心仓位逻辑完全复用实盘逻辑）
            target_ratio = 0
            if long_prob > config.THRESHOLD_LONG:
                target_ratio = self.position_manager.calculate_target_ratio(long_prob, money_flow_ratio, volatility)
            elif short_prob > config.THRESHOLD_SHORT:
                target_ratio = -self.position_manager.calculate_target_ratio(short_prob, money_flow_ratio, volatility)

            target_position = target_ratio * self.balance / price
            delta = target_position - self.position

            if abs(delta * price) >= config.MIN_ADJUST_AMOUNT:
                self.balance -= abs(delta * price * self.fee_rate)
                self.position += delta
                self.entry_price = price  # ✅ 简化成本更新
                action = '加多' if delta > 0 else '加空'
                self.trade_log.append((row.name, action, price, self.balance))

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
