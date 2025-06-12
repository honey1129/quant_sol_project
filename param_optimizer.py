import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools
import config
from utils import fetch_binance_data, add_indicators, get_feature_columns
import os

# 确保日志目录存在
os.makedirs("logs", exist_ok=True)

# 加载数据
def load_data():
    df = fetch_binance_data()
    df = add_indicators(df)
    features = get_feature_columns()
    df['future_return'] = df['close'].pct_change(3).shift(-3)
    df['target'] = (df['future_return'] > 0).astype(int)
    df.dropna(inplace=True)
    X = df[features].astype(float)
    y = df['target']
    return X, y, df, features

# 训练模型
def train_model(X_train, y_train):
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
    model.fit(X_train, y_train)
    return model

# 简化版回测引擎
def backtest(df, model, features, threshold_long, threshold_short, take_profit, stop_loss):
    balance = config.INITIAL_BALANCE
    position = 0
    entry_price = 0
    balances = []

    for idx, row in df.iterrows():
        X_row = pd.DataFrame([row[features].values], columns=features).astype(float)
        prob = model.predict_proba(X_row)[0]
        long_prob, short_prob = prob[1], prob[0]
        price = row['close']

        if position != 0:
            change_pct = (price - entry_price) / entry_price
            if position > 0:
                if change_pct >= take_profit or change_pct <= -stop_loss:
                    pnl = (price - entry_price) * abs(position)
                    balance += pnl
                    position = 0
                    entry_price = 0
            else:
                change_pct = (entry_price - price) / entry_price
                if change_pct >= take_profit or change_pct <= -stop_loss:
                    pnl = (entry_price - price) * abs(position)
                    balance += pnl
                    position = 0
                    entry_price = 0

        if position == 0:
            size = config.POSITION_SIZE * config.LEVERAGE / price
            if long_prob > threshold_long:
                position = size
                entry_price = price
            elif short_prob > threshold_short:
                position = -size
                entry_price = price

        balances.append(balance)

    df['balance'] = balances
    final_balance = df['balance'].iloc[-1]
    returns = df['balance'].pct_change().dropna()
    cum_return = (final_balance - config.INITIAL_BALANCE) / config.INITIAL_BALANCE
    max_dd = ((df['balance'].cummax() - df['balance']) / df['balance'].cummax()).max()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(24*365) if returns.std() > 0 else 0

    return cum_return, max_dd, sharpe

# 网格搜索核心逻辑
def grid_search():
    X, y, df, features = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = train_model(X_train, y_train)

    threshold_long_range = [0.55, 0.6, 0.65]
    threshold_short_range = [0.35, 0.4, 0.45]
    take_profit_range = [0.02, 0.03, 0.04]
    stop_loss_range = [0.01, 0.015, 0.02]

    results = []

    for tl, ts, tp, sl in itertools.product(threshold_long_range, threshold_short_range, take_profit_range, stop_loss_range):
        cum_return, max_dd, sharpe = backtest(df, model, features, tl, ts, tp, sl)
        results.append({
            'threshold_long': tl,
            'threshold_short': ts,
            'take_profit': tp,
            'stop_loss': sl,
            'return': cum_return,
            'max_dd': max_dd,
            'sharpe': sharpe
        })

    result_df = pd.DataFrame(results)
    result_df.sort_values(by='sharpe', ascending=False, inplace=True)
    print(result_df.head(10))
    result_df.to_csv('logs/param_optimization_results.csv', index=False)
    print("参数优化已完成，结果已保存到 logs/param_optimization_results.csv")

if __name__ == '__main__':
    grid_search()
