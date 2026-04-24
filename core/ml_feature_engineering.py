# ml_feature_engineering.py

from datetime import timezone

import numpy as np
import pandas as pd

# 单周期基础特征工程
def add_features(df):
    """
    为单周期K线数据添加一系列常用技术指标特征
    """
    df = df.copy()

    # 各类EMA均线
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['ema_60'] = df['close'].ewm(span=60, adjust=False).mean()

    # MACD指标
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ATR指标（标准True Range + Wilder平滑）
    prev_close = df['close'].shift(1)
    tr_components = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1)
    df['tr'] = tr_components.max(axis=1)
    df['atr_14'] = df['tr']
    df['atr'] = df['tr'].ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

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
    stoch_k, stoch_d, stoch_j = compute_stochastic(df, window=14)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    df['stoch_j'] = stoch_j

    # OBV指标
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # VWAP指标
    df['vwap'] = compute_vwap(df)

    # 收益率特征
    df['return_3'] = df['close'].pct_change(3)
    df['return_5'] = df['close'].pct_change(5)
    df['return_10'] = df['close'].pct_change(10)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df  # ❗ 注意：这里不做 dropna，留到融合时统一处理


def interval_to_timedelta(interval):
    normalized = str(interval).strip()
    if not normalized:
        raise ValueError("interval 不能为空")

    unit = normalized[-1].lower()
    try:
        value = int(normalized[:-1])
    except ValueError as exc:
        raise ValueError(f"无法解析周期: {interval}") from exc

    unit_map = {
        "m": "minutes",
        "h": "hours",
        "d": "days",
    }
    if unit not in unit_map:
        raise ValueError(f"暂不支持的周期单位: {interval}")
    return pd.Timedelta(**{unit_map[unit]: value})


def keep_confirmed_bars(df, interval, now_ts=None):
    """
    只保留已确认收盘的K线。

    - 若交易所返回 confirm 字段，则优先使用；
    - 否则按 timestamp + interval <= now 推断。
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    frame = df.copy()
    frame = frame.sort_index()
    interval_delta = interval_to_timedelta(interval)

    if now_ts is None:
        now_ts = pd.Timestamp.now(tz=timezone.utc)
    now_ts = pd.Timestamp(now_ts)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")

    index_ts = pd.DatetimeIndex(frame.index)
    if index_ts.tz is None:
        index_ts_utc = index_ts.tz_localize("UTC")
    else:
        index_ts_utc = index_ts.tz_convert("UTC")

    inferred_confirmed = pd.Series(
        (index_ts_utc + interval_delta) <= now_ts,
        index=frame.index,
        dtype=bool,
    )

    confirm_col = None
    for candidate in ("confirm", "confirmed", "is_confirmed"):
        if candidate in frame.columns:
            confirm_col = candidate
            break

    if confirm_col is not None:
        raw_confirm = frame[confirm_col]
        normalized_confirm = (
            raw_confirm.astype(str)
            .str.strip()
            .str.lower()
            .map({"1": True, "0": False, "true": True, "false": False})
        )
        frame["is_confirmed"] = normalized_confirm.fillna(False)
    else:
        frame["is_confirmed"] = inferred_confirmed

    frame["is_confirmed"] = frame["is_confirmed"] & inferred_confirmed
    frame = frame[frame["is_confirmed"]].copy()
    return frame

# 多周期融合逻辑
def merge_multi_period_features(data_dict, base_interval=None):
    """
    融合多周期数据为统一特征表，默认以最细周期为基准。

    高周期特征会先整体滞后一根再映射到低周期，避免在高周期K线尚未收盘时
    把该周期特征前向填充到更细粒度bar中。
    """
    if not data_dict:
        return pd.DataFrame()

    ordered_intervals = list(data_dict.keys())
    if base_interval is None:
        base_interval = min(ordered_intervals, key=lambda item: interval_to_timedelta(item))
    if base_interval not in data_dict:
        raise KeyError(f"基础周期 {base_interval} 不存在于 data_dict 中")

    prepared = {}
    for interval in ordered_intervals:
        confirmed_df = keep_confirmed_bars(data_dict[interval], interval)
        if confirmed_df.empty:
            continue
        df_features = add_features(confirmed_df)
        if interval != base_interval:
            df_features = df_features.shift(1)
        prepared[interval] = df_features.add_prefix(f"{interval}_")

    if base_interval not in prepared:
        return pd.DataFrame()

    merged = prepared[base_interval].copy()
    for interval in ordered_intervals:
        if interval == base_interval or interval not in prepared:
            continue
        merged = merged.join(prepared[interval], how="left")

    merged = merged.sort_index()
    merged.ffill(inplace=True)

    # 高周期长窗口特征在样本不足时可能整列为空，不能因此把整个结果表清空。
    merged.dropna(axis=1, how="all", inplace=True)

    # 最后做温和缺失裁剪：若缺失超出10%则丢弃该行
    min_non_na = max(1, int(np.ceil(merged.shape[1] * 0.9)))
    merged.dropna(thresh=min_non_na, inplace=True)

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
    df['ema_12'] = df['5m_close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['5m_close'].ewm(span=26, adjust=False).mean()
    df['ema_diff'] = (df['5m_close'] - df['ema_12']) / df['ema_12']

    # === 动量特征 ===
    df['momentum_10'] = df['5m_close'] / df['5m_close'].shift(10) - 1

    # === 成交量衍生特征 ===
    df['volume_ma'] = df['5m_volume'].rolling(10).mean()
    df['volume_ratio'] = df['5m_volume'] / (df['volume_ma'] + 1e-6)

    # 避免用未来数据回填到过去，缺失值交给调用方统一裁剪。
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# 各类技术指标工具函数
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_williams_r(df, window=14):
    highest_high = df['high'].rolling(window).max()
    lowest_low = df['low'].rolling(window).min()
    return -100 * (highest_high - df['close']) / (highest_high - lowest_low)

def compute_stochastic(df, window=14):
    low_min = df['low'].rolling(window).min()
    high_max = df['high'].rolling(window).max()
    denominator = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df['close'] - low_min) / denominator
    d = k.rolling(3).mean()
    j = 3 * k - 2 * d
    return k, d, j

def compute_vwap(df):
    return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
