# ml_feature_engineering.py

from datetime import timezone

import numpy as np
import pandas as pd

from config import config
from core.regime_filter import derive_market_regime
from core.trend_filter import derive_trend_context


REGIME_TREND_FEATURE_COLUMNS = [
    "trend_bias_num",
    "regime_trend_long",
    "regime_trend_short",
    "regime_range_high_vol",
    "is_high_vol",
    "trend_gap_abs",
]

# 每个周期里随价格水位整体漂移的绝对量级列。树模型直接吃这些列会把
# “当前价格处在某个绝对区间”当成信号，一旦实盘价格离开训练价位带就失效。
# 我们保留这些原始列供下游撮合/趋势判断使用，但用 stationary 派生列喂模型。
_PRICE_LEVEL_SUFFIXES = (
    "open", "high", "low", "close", "vwap",
    "boll_mid", "boll_upper", "boll_lower",
    "ema_10", "ema_20", "ema_30", "ema_60",
)
# 量纲与价格成正比、需除以 close 归一化的列。
_PRICE_SCALED_SUFFIXES = (
    "macd", "macd_signal", "macd_hist",
    "tr", "atr", "atr_14",
    "boll_std", "volatility_20", "rolling_atr_std", "momentum_10",
)

# 喂给模型时需要排除的列：标签、内部布尔标志、以及所有非平稳的绝对量级列。
# 原始列仍保留在 DataFrame 中（下游 predict/backtest/live 直接读取），
# 仅从模型输入里剔除，改用其 stationary 派生版本。
MODEL_FEATURE_EXCLUDE_EXACT = {
    "future_return", "target",
    # 跨周期之外的绝对量级列
    "ema_12", "ema_26",
    "money_flow", "money_flow_ma", "volume_ma",
    # Rubik 原始绝对量级(OI/taker 成交量),只用其平稳派生版,绝对值随市值漂移
    "rubik_open_interest", "rubik_oi_volume",
    "rubik_taker_sell_vol", "rubik_taker_buy_vol",
}
MODEL_FEATURE_EXCLUDE_SUFFIXES = tuple(
    f"_{suffix}" for suffix in (_PRICE_LEVEL_SUFFIXES + _PRICE_SCALED_SUFFIXES)
) + (
    "_obv",            # 累计量，保留 obv_zscore 派生版
    "_volume",         # 绝对成交量，保留 volume_ratio
    "_confirm",        # 收盘确认标志，过滤后近似常数
    "_is_confirmed",
)


def _is_excluded_model_feature(col):
    col = str(col)
    if col in MODEL_FEATURE_EXCLUDE_EXACT:
        return True
    if col.startswith("label_"):
        return True
    return col.endswith(MODEL_FEATURE_EXCLUDE_SUFFIXES)


def model_feature_columns(df):
    """Return the stationary subset of columns to feed the model.

    Absolute price-level / raw-magnitude columns drift with SOL's price and make
    tree models memorize the training-period price band instead of patterns. They
    stay in the DataFrame for downstream matching/trend logic, but are excluded
    here so training and live inference share one stationary feature set.
    """
    return [col for col in df.columns if not _is_excluded_model_feature(col)]

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

    with pd.option_context("future.no_silent_downcasting", True):
        df = df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
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
    with pd.option_context("future.no_silent_downcasting", True):
        merged = merged.ffill().infer_objects(copy=False)

    # 高周期长窗口特征在样本不足时可能整列为空，不能因此把整个结果表清空。
    merged.dropna(axis=1, how="all", inplace=True)

    # 最后做温和缺失裁剪：若缺失超出10%则丢弃该行
    min_non_na = max(1, int(np.ceil(merged.shape[1] * 0.9)))
    merged.dropna(thresh=min_non_na, inplace=True)

    return merged


def add_regime_trend_features(df):
    """
    Add explicit rule-based trend/regime features used by the trading gate.

    These are intentionally derived from the same trend/regime helpers as live trading
    so the model does not have to infer the gate state indirectly from EMA/ATR columns.
    """
    df = df.copy()
    rows = []
    for _, row in df.iterrows():
        close_price = row.get("5m_close")
        atr_value = row.get("5m_atr")
        atr_ratio = None
        if pd.notna(close_price) and pd.notna(atr_value) and float(close_price) > 0:
            atr_ratio = float(atr_value) / float(close_price)

        trend_context = derive_trend_context(
            row,
            interval=config.TREND_FILTER_INTERVAL,
            fast_col=config.TREND_FILTER_FAST_COL,
            slow_col=config.TREND_FILTER_SLOW_COL,
            min_gap=config.TREND_FILTER_MIN_GAP,
        )
        regime_context = derive_market_regime(
            trend_bias=trend_context.get("trend_bias"),
            trend_gap=trend_context.get("trend_gap"),
            volatility=row.get("volatility_15"),
            atr_ratio=atr_ratio,
            money_flow_ratio=row.get("money_flow_ratio"),
            trend_gap_threshold=config.REGIME_TREND_GAP_THRESHOLD,
            high_vol_atr_threshold=config.REGIME_HIGH_VOL_ATR_THRESHOLD,
            high_volatility_threshold=config.REGIME_HIGH_VOLATILITY_THRESHOLD,
            money_flow_extreme_threshold=config.REGIME_MONEY_FLOW_EXTREME_THRESHOLD,
        )

        trend_bias = str(trend_context.get("trend_bias") or "neutral").lower()
        regime = str(regime_context.get("regime") or "unknown").lower()
        trend_gap = trend_context.get("trend_gap")
        try:
            trend_gap_abs = abs(float(trend_gap))
        except (TypeError, ValueError):
            trend_gap_abs = 0.0
        if not np.isfinite(trend_gap_abs):
            trend_gap_abs = 0.0

        rows.append({
            "trend_bias_num": 1.0 if trend_bias == "long" else (-1.0 if trend_bias == "short" else 0.0),
            "regime_trend_long": 1.0 if regime == "trend_long" else 0.0,
            "regime_trend_short": 1.0 if regime == "trend_short" else 0.0,
            "regime_range_high_vol": 1.0 if regime == "range_high_vol" else 0.0,
            "is_high_vol": 1.0 if bool(regime_context.get("is_high_vol")) else 0.0,
            "trend_gap_abs": float(trend_gap_abs),
        })

    feature_df = pd.DataFrame(rows, index=df.index)
    for col in REGIME_TREND_FEATURE_COLUMNS:
        if col not in feature_df:
            feature_df[col] = 0.0
    return pd.concat([df, feature_df[REGIME_TREND_FEATURE_COLUMNS]], axis=1)


# Rubik(OI / taker / 多空比)派生的平稳特征列。绝对量级不进模型,只用变化率/失衡比。
RUBIK_FEATURE_COLUMNS = [
    "rubik_oi_change",        # OI 环比变化率
    "rubik_oi_vol_ratio",     # 成交量/OI,换手强度
    "rubik_taker_imbalance",  # (买-卖)/(买+卖),主动方向压力
    "rubik_taker_buy_share",  # 买量占比
    "rubik_ls_ratio",         # 多空账户比(已是比值,近似平稳)
    "rubik_ls_ratio_change",  # 多空比环比变化
]


def add_rubik_features(df, rubik_data):
    """Merge 1H Rubik stats (OI / taker / long-short) onto the 5m frame.

    Rubik 仅 1H 粒度且只有 ~30 天历史。为防前视:1H 统计 bar 的时间戳 T 代表
    [T, T+1H),要等该小时收盘后才可见,故先把 ts 前移一个周期再 merge_asof,
    与 merge_multi_period_features 对高周期 shift(1) 的口径一致。

    只产出无量纲平稳列(OI 变化率、taker 失衡比、多空比)。绝对 OI/成交量不进模型。
    rubik_data 缺失或为空时,所有 rubik 列填 0(模型当作无信息)。
    """
    df = df.copy()
    # rubik_data 未提供 = 特征关闭。此时不添加任何 rubik 列,避免常数零列污染
    # feature_list(A/B 实测 rubik 特征不改善 OOS,默认关闭)。
    if not rubik_data:
        return df
    for col in RUBIK_FEATURE_COLUMNS:
        df[col] = 0.0

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    # 保留原始行序:base 按 ts 升序排,但记下原位置 __pos__ 以便写回。
    base = pd.DataFrame({"__pos__": np.arange(len(df)), "__ts__": idx})
    base = base.sort_values("__ts__").reset_index(drop=True)

    period = pd.Timedelta(hours=1)  # 1H rubik

    def merge_block(series_df, derive, out_cols):
        if series_df is None or series_df.empty:
            return
        s = series_df.sort_values("ts").reset_index(drop=True).copy()
        s["ts"] = pd.to_datetime(s["ts"], utc=True)
        derive(s)
        s["__avail__"] = s["ts"] + period  # 收盘后才可见,防前视
        merged = pd.merge_asof(
            base, s[["__avail__"] + out_cols],
            left_on="__ts__", right_on="__avail__", direction="backward",
        )
        for col in out_cols:
            vals = np.zeros(len(df), dtype=float)
            vals[merged["__pos__"].to_numpy()] = merged[col].to_numpy()
            df[col] = vals

    def _oi(s):
        s["rubik_oi_change"] = s["open_interest"].pct_change()
        s["rubik_oi_vol_ratio"] = s["oi_volume"] / (s["open_interest"].abs() + 1e-9)

    def _taker(s):
        total = s["taker_buy_vol"] + s["taker_sell_vol"]
        s["rubik_taker_imbalance"] = (s["taker_buy_vol"] - s["taker_sell_vol"]) / (total + 1e-9)
        s["rubik_taker_buy_share"] = s["taker_buy_vol"] / (total + 1e-9)

    def _ls(s):
        s["rubik_ls_ratio"] = s["long_short_ratio"]
        s["rubik_ls_ratio_change"] = s["long_short_ratio"].pct_change()

    merge_block(rubik_data.get("open_interest"), _oi, ["rubik_oi_change", "rubik_oi_vol_ratio"])
    merge_block(rubik_data.get("taker_volume"), _taker, ["rubik_taker_imbalance", "rubik_taker_buy_share"])
    merge_block(rubik_data.get("long_short_ratio"), _ls, ["rubik_ls_ratio", "rubik_ls_ratio_change"])

    df[RUBIK_FEATURE_COLUMNS] = df[RUBIK_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def add_stationary_features(df):
    """Add dimensionless versions of absolute price-level / magnitude columns.

    Tree models trained on raw `ema_60`, `boll_mid`, `vwap`, `macd`, `atr`, `obv`...
    learn the training-period price band rather than transferable shape. These
    derived columns are scale-free, so they stay valid as price drifts. Raw columns
    are kept untouched for downstream matching/trend logic.
    """
    df = df.copy()
    new_cols = {}

    for interval in config.INTERVALS:
        prefix = str(interval)
        close_col = f"{prefix}_close"
        if close_col not in df.columns:
            continue
        close = df[close_col].replace(0, np.nan)

        # 价格水位类 → 相对当周期 close 的距离比（围绕 0 平稳）
        for suffix in _PRICE_LEVEL_SUFFIXES:
            if suffix == "close":
                continue  # close/close-1 恒为 0，无信息
            col = f"{prefix}_{suffix}"
            if col in df.columns:
                new_cols[f"{col}_rel"] = df[col] / close - 1.0

        # 与价格成正比的量级列 → 除以 close 归一化
        for suffix in _PRICE_SCALED_SUFFIXES:
            col = f"{prefix}_{suffix}"
            if col in df.columns:
                new_cols[f"{col}_norm"] = df[col] / close

        # OBV 是累计量，本身随时间发散 → 用差分后的滚动 z-score
        obv_col = f"{prefix}_obv"
        if obv_col in df.columns:
            obv_delta = df[obv_col].diff()
            roll_mean = obv_delta.rolling(20).mean()
            roll_std = obv_delta.rolling(20).std()
            new_cols[f"{obv_col}_zscore"] = (obv_delta - roll_mean) / (roll_std + 1e-9)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


# 多因子衍生特征工程
def add_advanced_features(df, rubik_data=None):
    """
    融入资金流、波动率、微结构等衍生高阶特征

    rubik_data(可选):{"open_interest","taker_volume","long_short_ratio"} -> DataFrame,
    传入则接入 OI/taker/多空比的平稳派生特征;不传则这些列恒为 0(等价于关闭)。
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

    # === 绝对量级特征的无量纲（平稳）版本 ===
    df = add_stationary_features(df)

    df = add_regime_trend_features(df)

    # === Rubik(OI / taker / 多空比)平稳特征,可选 ===
    df = add_rubik_features(df, rubik_data)

    # 避免用未来数据回填到过去，缺失值交给调用方统一裁剪。
    with pd.option_context("future.no_silent_downcasting", True):
        df = df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
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
