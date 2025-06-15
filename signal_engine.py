# signal_engine.py

from data_fetcher import fetch_data
from feature_engineering import add_indicators

def compute_signal():
    data = fetch_data()

    # 依次给每个周期计算指标
    for interval in data:
        data[interval] = add_indicators(data[interval])

    # 1小时周期做趋势过滤
    df_1h = data['1h']
    latest_1h = df_1h.iloc[-1]
    trend_up = (latest_1h['macd'] > 0) and (latest_1h['rsi'] > 50) and (latest_1h['close'] > latest_1h['ma'])
    trend_down = (latest_1h['macd'] < 0) and (latest_1h['rsi'] < 50) and (latest_1h['close'] < latest_1h['ma'])

    # 5分钟周期做择时信号
    df_5m = data['5m']
    latest_5m = df_5m.iloc[-1]
    signal_up = (latest_5m['macd'] > 0) and (latest_5m['rsi'] > 50) and (latest_5m['close'] > latest_5m['ma'])
    signal_down = (latest_5m['macd'] < 0) and (latest_5m['rsi'] < 50) and (latest_5m['close'] < latest_5m['ma'])

    # 多周期融合信号逻辑
    if trend_up and signal_up:
        return 'long'
    elif trend_down and signal_down:
        return 'short'
    else:
        return 'hold'
