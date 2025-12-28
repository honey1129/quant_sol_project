# ml_feature_engineering.py

import numpy as np

# 单周期基础特征工程
def add_features(df):
    """
    为单周期K线数据添加一系列常用技术指标特征
    """
    df = df.copy()

    # 各类EMA均线
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_30'] = df['close'].ewm(span=30).mean()
    df['ema_60'] = df['close'].ewm(span=60).mean()

    # MACD指标
    ema_fast = df['close'].ewm(span=12).mean()
    ema_slow = df['close'].ewm(span=26).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # ATR指标
    df['atr_14'] = df[['high', 'low', 'close']].apply(lambda x: x['high'] - x['low'], axis=1)
    df['atr'] = df['atr_14'].rolling(window=14).mean()

    # 波动率与布林带
    df['volatility_20'] = df['close'].rolling(window=20).std()
    df['rolling_atr_std'] = df['atr'].rolling(window=20).std()
    df['boll_mid'] = df['close'].rolling(window=20).mean()
    df['boll_std'] = df['volatility_20']
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']

    # RSI指标
    df['rsi'] = compute_rsi(df['close'], window=14)

    # 动量与变化率指标
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['roc_12'] = df['close'].pct_change(12)

    # 威廉指标、随机指标
    df['williams_r'] = compute_williams_r(df, window=14)
    df['stoch_k'] = compute_stochastic(df, window=14)

    # OBV指标
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # VWAP指标
    df['vwap'] = compute_vwap(df)

    # 收益率特征
    df['return_3'] = df['close'].pct_change(3)
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)

    return df  # ❗ 注意：这里不做 dropna，留到融合时统一处理

# 多周期融合逻辑
def merge_multi_period_features(data_dict):
    """
    融合多周期数据为统一特征表，智能处理缺失值，最大化训练样本量
    """
    feature_list = []

    for interval, df in data_dict.items():
        df_features = add_features(df)
        df_features = df_features.add_prefix(f"{interval}_")
        feature_list.append(df_features)

    # 逐步进行多周期join
    merged = feature_list[0]
    for df in feature_list[1:]:
        merged = merged.join(df, how="outer")  # ⚠ 关键点：使用 outer join 保留更多数据
        merged = merged.sort_index()

    merged = merged.sort_index()
    merged.ffill(inplace=True)
    merged.dropna(inplace=True)

    # 最后做温和缺失裁剪：若缺失超出10%则丢弃该行
    merged.dropna(thresh=int(merged.shape[1] * 0.9), inplace=True)

    return merged

# 多因子衍生特征工程
def add_advanced_features(df):
    """
    融入资金流、波动率、微结构等衍生高阶特征
    """
    # === 资金流指标 ===
    df['money_flow'] = df['5m_close'] * df['5m_volume']
    df['money_flow_ma'] = df['money_flow'].rolling(12).mean()
    df['money_flow_ratio'] = df['money_flow'] / (df['money_flow_ma'] + 1e-6)

    # === 波动率特征 ===
    df['log_return'] = np.log(df['5m_close'] / df['5m_close'].shift(1))
    df['volatility_5'] = df['log_return'].rolling(5).std()
    df['volatility_15'] = df['log_return'].rolling(15).std()

    # === 微结构特征（价差占比） ===
    df['hl_spread'] = (df['5m_high'] - df['5m_low']) / df['5m_close']

    # === 均线乖离率特征 ===
    df['ema_12'] = df['5m_close'].ewm(span=12).mean()
    df['ema_26'] = df['5m_close'].ewm(span=26).mean()
    df['ema_diff'] = (df['5m_close'] - df['ema_12']) / df['ema_12']

    # === 动量特征 ===
    df['momentum_10'] = df['5m_close'] / df['5m_close'].shift(10) - 1

    # === 成交量衍生特征 ===
    df['volume_ma'] = df['5m_volume'].rolling(10).mean()
    df['volume_ratio'] = df['5m_volume'] / (df['volume_ma'] + 1e-6)

    # 补充缺失值
    df.bfill(inplace=True)
    return df


# 各类技术指标工具函数
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
