import numpy as np

# 你原始的add_features函数，这里直接复用
def add_features(df):
    df = df.copy()
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_30'] = df['close'].ewm(span=30).mean()
    df['ema_60'] = df['close'].ewm(span=60).mean()
    ema_fast = df['close'].ewm(span=12).mean()
    ema_slow = df['close'].ewm(span=26).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['atr_14'] = df[['high', 'low', 'close']].apply(lambda x: x['high'] - x['low'], axis=1)
    df['atr'] = df['atr_14'].rolling(window=14).mean()
    df['volatility_20'] = df['close'].rolling(window=20).std()
    df['rolling_atr_std'] = df['atr'].rolling(window=20).std()
    df['boll_mid'] = df['close'].rolling(window=20).mean()
    df['boll_std'] = df['volatility_20']
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
    df['rsi'] = compute_rsi(df['close'], window=14)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['roc_12'] = df['close'].pct_change(12)
    df['williams_r'] = compute_williams_r(df, window=14)
    df['stoch_k'] = compute_stochastic(df, window=14)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['vwap'] = compute_vwap(df)
    df['return_3'] = df['close'].pct_change(3)
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)
    df.dropna(inplace=True)
    return df

# 技术指标函数：
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_williams_r(df, window=14):
    highest_high = df['high'].rolling(window).max()
    lowest_low = df['low'].rolling(window).min()
    return -100 * (highest_high - df['close']) / (highest_high - lowest_low)

def compute_stochastic(df, window=14):
    low_min = df['low'].rolling(window).min()
    high_max = df['high'].rolling(window).max()
    return 100 * (df['close'] - low_min) / (high_max - low_min)

def compute_vwap(df):
    return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()


# 🚩 多周期融合函数
def merge_multi_period_features(data_dict):
    feature_list = []
    for interval, df in data_dict.items():
        df_features = add_features(df)
        df_features = df_features.add_prefix(f"{interval}_")
        feature_list.append(df_features)

    merged = feature_list[0]
    for df in feature_list[1:]:
        merged = merged.join(df, how="inner")

    merged.dropna(inplace=True)
    return merged
