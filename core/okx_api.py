import math
import time
import uuid
import pandas as pd
from config import config
import okx.Account as Account
import okx.Trade as Trade
import okx.MarketData as Market
import okx.PublicData as Public
import okx.TradingData as TradingData
from utils.utils import log_info, log_error


def build_client_order_id(symbol, side, pos_side, reduce_only):
    symbol_part = "".join(ch for ch in str(symbol).lower() if ch.isalnum())[:8]
    side_part = str(side).lower()[:1] or "x"
    pos_part = str(pos_side).lower()[:1] or "x"
    reduce_part = "r" if reduce_only else "o"
    unique_part = uuid.uuid4().hex[:16]
    return f"qs{symbol_part}{reduce_part}{side_part}{pos_part}{unique_part}"[:32]


def order_is_acknowledged(order):
    if not order:
        return False
    state = str(order.get("state", "") or "").lower()
    return state in {"live", "partially_filled", "filled"}


def order_is_filled(order):
    if not order:
        return False
    state = str(order.get("state", "") or "").lower()
    return state in {"filled", "partially_filled"}


def floor_size_to_lot(size, lot_size):
    size = float(size)
    lot_size = float(lot_size)
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")
    return round(math.floor(size / lot_size) * lot_size, 6)


def cap_size_by_available_margin(
    size,
    market_price,
    leverage,
    available_usdt,
    lot_size,
    *,
    usage_ratio=0.85,
    min_free_margin_usdt=0.0,
):
    size = floor_size_to_lot(size, lot_size)
    market_price = float(market_price)
    leverage = float(leverage)
    available_usdt = float(available_usdt)
    lot_size = float(lot_size)
    usage_ratio = max(0.0, min(float(usage_ratio), 1.0))
    min_free_margin_usdt = max(0.0, float(min_free_margin_usdt))

    if size <= 0 or market_price <= 0 or leverage <= 0:
        return 0.0, 0.0, 0.0, False

    usable_margin = max(0.0, available_usdt - min_free_margin_usdt) * usage_ratio
    required_margin = size * market_price / leverage
    if required_margin <= usable_margin:
        return size, required_margin, usable_margin, False

    capped_size = floor_size_to_lot(usable_margin * leverage / market_price, lot_size)
    capped_required_margin = capped_size * market_price / leverage if capped_size > 0 else 0.0
    return capped_size, capped_required_margin, usable_margin, True


def is_insufficient_margin_error(result):
    if not isinstance(result, dict):
        return False
    for item in result.get("data", []) or []:
        code = str(item.get("sCode", "") or "")
        message = str(item.get("sMsg", "") or "").lower()
        if code == "51008" or "insufficient" in message and "margin" in message:
            return True
    return False


class OKXClient:
    def __init__(self):
        self.account_api = Account.AccountAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.trade_api = Trade.TradeAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.market_api = Market.MarketAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.public_api = Public.PublicAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True,flag=config.USE_SERVER)
        # Rubik 交易大数据(OI / taker / 多空比)。仅做只读统计,无需签名,但沿用同一 flag。
        self.trading_data_api = TradingData.TradingDataAPI(flag=config.USE_SERVER, debug=False)

    def _call_with_retry(self, label, func, *, max_retry=None, sleep_sec=None, backoff=None):
        max_retry = max(1, int(max_retry if max_retry is not None else config.OKX_API_MAX_RETRY))
        sleep_sec = max(0.0, float(sleep_sec if sleep_sec is not None else config.OKX_API_RETRY_SLEEP_SEC))
        backoff = max(1.0, float(backoff if backoff is not None else config.OKX_API_RETRY_BACKOFF))
        delay = sleep_sec
        last_error = None

        for attempt in range(max_retry):
            try:
                return func()
            except Exception as exc:
                last_error = exc
                log_error(f"⚠ {label}失败，第{attempt + 1}/{max_retry}次: {exc}")
                if attempt < max_retry - 1 and delay > 0:
                    time.sleep(delay)
                    delay *= backoff

        raise last_error

    def get_account_config(self):
        result = self._call_with_retry(
            "获取账户配置",
            self.account_api.get_account_config,
        )
        if result.get("code") != "0":
            raise RuntimeError(f"获取账户配置失败: {result}")
        data = result.get("data", [])
        if not data:
            raise RuntimeError("账户配置为空，无法继续")
        return data[0]

    def list_pending_orders(self):
        result = self._call_with_retry(
            "获取挂单列表",
            lambda: self.trade_api.get_order_list(
                instType="SWAP",
                instId=config.SYMBOL,
            ),
        )
        if result.get("code") != "0":
            raise RuntimeError(f"获取挂单列表失败: {result}")
        return result.get("data", [])

    def get_order_by_client_id(self, cl_ord_id):
        if not cl_ord_id:
            return None
        try:
            result = self._call_with_retry(
                "按 clOrdId 查询订单",
                lambda: self.trade_api.get_order(instId=config.SYMBOL, clOrdId=cl_ord_id),
            )
        except Exception as exc:
            log_error(f"按 clOrdId 查询订单失败: {cl_ord_id} -> {exc}")
            return None

        if result.get("code") != "0":
            return None
        data = result.get("data", [])
        return data[0] if data else None

    def fetch_order_fills(self, ord_id):
        if not ord_id:
            return []
        try:
            result = self._call_with_retry(
                "查询订单成交明细",
                lambda: self.trade_api.get_fills(
                    instType="SWAP",
                    instId=config.SYMBOL,
                    ordId=ord_id,
                ),
            )
        except Exception as exc:
            log_error(f"查询订单成交明细失败: ordId={ord_id} -> {exc}")
            return []

        if result.get("code") != "0":
            log_error(f"查询订单成交明细失败: ordId={ord_id}, result={result}")
            return []
        return result.get("data", []) or []

    def _attach_order_fills(self, order):
        if not order:
            return order
        if "_fills" in order:
            return order
        if not hasattr(self, "trade_api"):
            enriched = dict(order)
            enriched["_fills"] = []
            return enriched
        try:
            fills = self.fetch_order_fills(order.get("ordId"))
        except Exception:
            fills = []
        enriched = dict(order)
        enriched["_fills"] = fills
        return enriched

    def wait_until_filled(self, cl_ord_id, timeout_sec=5.0, poll_interval_sec=0.3):
        if not cl_ord_id:
            return None
        deadline = time.monotonic() + float(timeout_sec)
        last_order = None
        while True:
            order = self.get_order_by_client_id(cl_ord_id)
            if order is not None:
                last_order = order
                if order_is_filled(order):
                    return self._attach_order_fills(order)
                state = str(order.get("state", "") or "").lower()
                if state in {"canceled", "rejected", "failed", "mmp_canceled"}:
                    return None
            if time.monotonic() >= deadline:
                return None
            time.sleep(poll_interval_sec)

    def cancel_pending_orders(self):
        pending_orders = self.list_pending_orders()
        if not pending_orders:
            return []

        canceled = []
        for order in pending_orders:
            ord_id = order.get("ordId", "")
            cl_ord_id = order.get("clOrdId", "")
            try:
                result = self.trade_api.cancel_order(
                    instId=config.SYMBOL,
                    ordId=ord_id,
                    clOrdId=cl_ord_id,
                )
                if result.get("code") == "0":
                    canceled.append({
                        "ordId": ord_id,
                        "clOrdId": cl_ord_id,
                        "state": order.get("state", ""),
                    })
                else:
                    log_error(f"撤销挂单失败: ordId={ord_id}, clOrdId={cl_ord_id}, result={result}")
            except Exception as exc:
                log_error(f"撤销挂单异常: ordId={ord_id}, clOrdId={cl_ord_id}, err={exc}")

        if canceled:
            log_info(f"已清理挂单 {len(canceled)} 笔")
        return canceled

    def _extract_leverage_by_side(self, leverage_rows):
        leverage_map = {}
        for row in leverage_rows:
            pos_side = str(row.get("posSide", "") or "").lower() or "both"
            try:
                leverage_map[pos_side] = float(row.get("lever", 0) or 0)
            except (TypeError, ValueError):
                continue
        return leverage_map

    def ensure_trading_ready(self):
        if bool(config.LIVE_REQUIRE_SIMULATED_TRADING) and str(config.USE_SERVER) != "1":
            raise RuntimeError("当前配置不是 OKX 模拟盘(USE_SERVER=1)，已阻止启动交易监控")

        if not config.OKX_API_KEY or not config.OKX_SECRET or not config.OKX_PASSWORD:
            raise RuntimeError("OKX API 凭证缺失，无法启动交易监控")

        account_config = self.get_account_config()
        pos_mode = str(account_config.get("posMode", "") or "").lower()

        if bool(config.LIVE_RECONCILE_PENDING_ORDERS):
            self.cancel_pending_orders()

        if pos_mode != "long_short_mode":
            if not bool(config.LIVE_AUTO_SET_POSITION_MODE):
                raise RuntimeError(f"账户持仓模式为 {pos_mode or 'unknown'}，与策略要求的 long_short_mode 不一致")

            long_pos, short_pos = self.get_position()
            if long_pos["size"] > 0 or short_pos["size"] > 0:
                raise RuntimeError("持仓模式不是 long_short_mode，且当前已有持仓，拒绝自动切换")

            result = self._call_with_retry(
                "自动设置持仓模式",
                lambda: self.account_api.set_position_mode("long_short_mode"),
            )
            if result.get("code") != "0":
                raise RuntimeError(f"自动设置持仓模式失败: {result}")
            log_info("已自动切换到账户双向持仓模式 long_short_mode")
            time.sleep(0.5)
            account_config = self.get_account_config()
            pos_mode = str(account_config.get("posMode", "") or "").lower()
            if pos_mode != "long_short_mode":
                raise RuntimeError(f"持仓模式校验失败，当前为 {pos_mode or 'unknown'}")

        leverage_rows = self._call_with_retry(
            "获取杠杆配置",
            lambda: self.account_api.get_leverage(mgnMode="cross", instId=config.SYMBOL),
        )
        if leverage_rows.get("code") != "0":
            raise RuntimeError(f"获取杠杆配置失败: {leverage_rows}")
        leverage_map = self._extract_leverage_by_side(leverage_rows.get("data", []))
        target_leverage = float(config.LEVERAGE)

        if bool(config.LIVE_AUTO_SET_LEVERAGE):
            for pos_side in ("long", "short"):
                current_leverage = leverage_map.get(pos_side, leverage_map.get("both", 0.0))
                if abs(current_leverage - target_leverage) > 1e-9:
                    result = self._call_with_retry(
                        f"设置 {pos_side} 杠杆",
                        lambda pos_side=pos_side: self.account_api.set_leverage(
                            lever=str(config.LEVERAGE),
                            mgnMode="cross",
                            instId=config.SYMBOL,
                            posSide=pos_side,
                        ),
                    )
                    if result.get("code") != "0":
                        raise RuntimeError(f"设置 {pos_side} 杠杆失败: {result}")
                    log_info(f"已自动设置 {pos_side} 杠杆为 {config.LEVERAGE}x")
        else:
            for pos_side in ("long", "short"):
                current_leverage = leverage_map.get(pos_side, leverage_map.get("both", 0.0))
                if abs(current_leverage - target_leverage) > 1e-9:
                    raise RuntimeError(
                        f"{pos_side} 杠杆当前为 {current_leverage}x，目标为 {target_leverage}x，且已关闭自动设置"
                    )

        log_info(
            f"交易环境校验完成: simulated={str(config.USE_SERVER) == '1'}, "
            f"pos_mode={pos_mode}, leverage_target={config.LEVERAGE}x"
        )

    # 获取当前账户余额等信息
    def get_account_balance(self):
        result = self._call_with_retry(
            "获取账户余额",
            self.account_api.get_account_balance,
        )
        if result.get("code") != "0":
            raise RuntimeError(f"获取账户余额失败: {result}")

        total_eq_raw = result['data'][0].get('totalEq', '0')
        total_eq = float(total_eq_raw) if total_eq_raw not in ['', None] else 0.0
        result['data'][0]['totalEq'] = total_eq

        details = result['data'][0].get('details', [])
        usdt_detail = next((d for d in details if d.get('ccy') == 'USDT'), None)

        if usdt_detail:
            avail_eq_raw = usdt_detail.get('availEq', '0')
            avail_eq = float(avail_eq_raw) if avail_eq_raw not in ['', None] else 0.0
        else:
            avail_eq = 0.0

        result['data'][0]['availEq'] = avail_eq

        return result

    # 获取SYMBOL当前最新仓位
    def get_position(self):
        result = self._call_with_retry(
            "获取仓位",
            lambda: self.account_api.get_positions(instType='SWAP', instId=config.SYMBOL),
        )
        if result.get("code") != "0":
            raise RuntimeError(f"获取仓位失败: {result}")
        positions = result['data']

        long_position = {'size': 0.0, 'entry_price': 0.0}
        short_position = {'size': 0.0, 'entry_price': 0.0}

        for pos in positions:
            pos_side = pos.get('posSide', '')
            size_raw = pos.get('pos', '0')
            avgPx_raw = pos.get('avgPx', '0')

            size = float(size_raw) if size_raw not in ['', None] else 0.0
            avg_price = float(avgPx_raw) if avgPx_raw not in ['', None] else 0.0

            if pos_side == 'long':
                long_position['size'] = size
                long_position['entry_price'] = avg_price

            elif pos_side == 'short':
                short_position['size'] = size
                short_position['entry_price'] = avg_price

        return long_position, short_position

    # 获取SYMBOL当前最新价格(以usdt计价)
    def get_price(self, max_retry=3, sleep_sec=1):
        for attempt in range(max_retry):
            try:
                data = self.market_api.get_ticker(instId=config.SYMBOL)
                price_raw = data['data'][0].get('last', '0')
                if price_raw in ['', None]:
                    raise Exception("❌ last价格字段为空")
                last_price = float(price_raw)
                return last_price
            except Exception as e:
                log_error(f"⚠ 获取价格失败，第{attempt + 1}次重试: {e}")
                time.sleep(sleep_sec)
        raise Exception("❌ 超过最大重试次数，get_price() 彻底失败")

    # 获取最近已平仓交易的真实收益率（计算reward_risk用）
    def fetch_recent_closed_trades(self, limit=50):
        result = self._call_with_retry(
            "获取历史平仓",
            lambda: self.account_api.get_positions_history(instType="SWAP", instId=config.SYMBOL, limit=str(limit)),
        )
        if result.get("code") != "0":
            log_error(f"获取历史平仓失败: {result}")
            return []
        trades = []
        for item in result.get("data", []):
            try:
                open_px = float(item.get("openAvgPx", 0))
                close_px = float(item.get("closeAvgPx", 0))
                size = abs(float(item.get("closeTotalPos", 0)))
                realized_pnl = float(item.get("realizedPnl", 0))
                fee = float(item.get("fee", 0))

                if open_px <= 0 or close_px <= 0 or size <= 0:
                    continue

                avg_px = (open_px + close_px) / 2
                notional = size * avg_px
                if notional <= 0:
                    continue
                net_pnl = realized_pnl + fee
                trade_return = net_pnl / notional
                trades.append(trade_return)

            except Exception:
                continue

        return trades

    # OKX 历史K线完整拉取函数：支持自动分页、稳定拉取大规模历史数据
    def fetch_ohlcv(self,symbol=config.SYMBOL, bar="1H", max_limit=2000, max_retry=3, sleep_sec=1):
        all_data = []
        next_after = ''

        while len(all_data) < max_limit:
            remaining = max_limit - len(all_data)
            limit = min(300, remaining)
            batch = None
            for attempt in range(max_retry):
                try:
                    response = self.market_api.get_history_candlesticks(
                        instId=symbol,
                        bar=bar,
                        limit=limit,
                        after=next_after
                    )
                    batch = response['data']
                    break
                except Exception as e:
                    print(f"⚠️ 拉取K线失败，重试中 ({attempt + 1}/{max_retry}): {e}")
                    time.sleep(sleep_sec)
            else:
                print("❌ 超过最大重试次数，跳过当前分页")
                break

            if not batch:
                break

            batch_sorted = sorted(batch, key=lambda x: int(x[0]))  # 时间升序
            all_data.extend(batch_sorted)

            if len(batch) < limit:
                break  # 没有更多了

            # ✅ 翻页核心逻辑：用最早时间戳向前翻页
            next_after = str(batch_sorted[0][0])

            time.sleep(0.2)  # 防止API限速

        if not all_data:
            raise Exception("❌ 无法拉取任何K线数据，请检查API权限/网络")

        # 转换为DataFrame。分页结果可能按“最近批次在前、历史批次在后”拼接，
        # 这里统一按时间正序排序，并去重，避免滚动特征被乱序数据污染。
        normalized_rows = []
        for row in all_data:
            normalized_rows.append({
                "timestamp": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "confirm": row[8] if len(row) > 8 else "1",
            })

        df = pd.DataFrame(normalized_rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms', utc=True)
        df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        df.sort_values('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df['confirm'] = df['confirm'].astype(str)
        df.reset_index(drop=True, inplace=True)
        return df

    # 批量获取多个周期的k线数据
    def fetch_data(self):
        data_dict = {}
        for interval in config.INTERVALS:
            df = self.fetch_ohlcv(config.SYMBOL, bar=interval, max_limit=config.WINDOWS[interval])
            df.set_index("timestamp", inplace=True)
            data_dict[interval] = df
            time.sleep(0.3)
        return data_dict

    def fetch_funding_rate_history(self, symbol=config.SYMBOL, max_records=None, max_retry=3, sleep_sec=1):
        if max_records is None:
            max_records = config.BACKTEST_FUNDING_HISTORY_LIMIT

        all_rows = []
        next_after = ''

        while len(all_rows) < max_records:
            remaining = max_records - len(all_rows)
            limit = min(100, remaining)
            batch = None

            for attempt in range(max_retry):
                try:
                    response = self.public_api.funding_rate_history(
                        instId=symbol,
                        after=next_after,
                        limit=str(limit),
                    )
                    batch = response.get('data', [])
                    break
                except Exception as e:
                    print(f"⚠️ 拉取 funding 历史失败，重试中 ({attempt + 1}/{max_retry}): {e}")
                    time.sleep(sleep_sec)
            else:
                print("❌ 超过最大重试次数，跳过 funding 历史分页")
                break

            if not batch:
                break

            all_rows.extend(batch)

            if len(batch) < limit:
                break

            oldest_time = min(int(item['fundingTime']) for item in batch)
            next_after = str(oldest_time)
            time.sleep(0.2)

        if not all_rows:
            return pd.DataFrame(columns=['funding_time', 'funding_rate'])

        df = pd.DataFrame(all_rows)
        df['funding_time'] = pd.to_datetime(df['fundingTime'].astype(float), unit='ms', utc=True)
        rate_col = 'realizedRate' if 'realizedRate' in df.columns else 'fundingRate'
        df['funding_rate'] = df[rate_col].astype(float)
        df.drop_duplicates(subset=['funding_time'], keep='last', inplace=True)
        df.sort_values('funding_time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df[['funding_time', 'funding_rate']]

    @staticmethod
    def _ccy_from_symbol(symbol):
        # Rubik 统计接口按币种(ccy)而非合约 instId 查询,例如 SOL-USDT-SWAP -> SOL。
        return str(symbol).split('-')[0]

    def _fetch_rubik_series(self, label, fetch_fn, value_cols, *, period='1H', max_retry=3, sleep_sec=1):
        """通用 Rubik 历史拉取。

        Rubik 端点单次返回约 720 行(1H 约 30 天)、时间倒序、每行是数组。
        这里统一成时间正序的 DataFrame,首列为 ts,其余按 value_cols 命名。
        无历史(快照型)或拉取失败时返回空表,调用方需容忍缺列。
        """
        rows = None
        for attempt in range(max_retry):
            try:
                resp = fetch_fn()
                if str(resp.get('code')) != '0':
                    raise ValueError(f"code={resp.get('code')} msg={resp.get('msg')}")
                rows = resp.get('data', [])
                break
            except Exception as e:
                print(f"⚠️ 拉取 {label} 失败，重试中 ({attempt + 1}/{max_retry}): {e}")
                time.sleep(sleep_sec)
        else:
            print(f"❌ 超过最大重试次数，跳过 {label}")
            rows = None

        cols = ['ts'] + list(value_cols)
        if not rows:
            return pd.DataFrame(columns=cols)

        # 防御:实际列数可能与预期不符(OKX 偶尔加列),按最小长度对齐。
        width = min(len(cols), len(rows[0]))
        df = pd.DataFrame([r[:width] for r in rows], columns=cols[:width])
        df['ts'] = pd.to_datetime(df['ts'].astype('int64'), unit='ms', utc=True)
        for col in cols[1:width]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.drop_duplicates(subset=['ts'], keep='last', inplace=True)
        df.sort_values('ts', inplace=True)  # Rubik 默认倒序,这里转正序
        df.reset_index(drop=True, inplace=True)
        return df

    def fetch_open_interest_history(self, symbol=config.SYMBOL, period='1H', max_retry=3, sleep_sec=1):
        """合约持仓量(OI)+ 成交量历史。行格式 [ts, oi, volume]。"""
        ccy = self._ccy_from_symbol(symbol)
        return self._fetch_rubik_series(
            "OI 历史",
            lambda: self.trading_data_api.get_contracts_interest_volume(ccy=ccy, period=period),
            ['open_interest', 'oi_volume'],
            period=period, max_retry=max_retry, sleep_sec=sleep_sec,
        )

    def fetch_taker_volume_history(self, symbol=config.SYMBOL, period='1H', max_retry=3, sleep_sec=1):
        """主动买卖(taker)成交量历史。OKX 行格式 [ts, sellVol, buyVol]。"""
        ccy = self._ccy_from_symbol(symbol)
        return self._fetch_rubik_series(
            "taker 成交量历史",
            lambda: self.trading_data_api.get_taker_volume(ccy=ccy, instType='CONTRACTS', period=period),
            ['taker_sell_vol', 'taker_buy_vol'],
            period=period, max_retry=max_retry, sleep_sec=sleep_sec,
        )

    def fetch_long_short_ratio_history(self, symbol=config.SYMBOL, period='1H', max_retry=3, sleep_sec=1):
        """多空账户比历史。行格式 [ts, ratio]。"""
        ccy = self._ccy_from_symbol(symbol)
        return self._fetch_rubik_series(
            "多空比历史",
            lambda: self.trading_data_api.get_long_short_ratio(ccy=ccy, period=period),
            ['long_short_ratio'],
            period=period, max_retry=max_retry, sleep_sec=sleep_sec,
        )

    def fetch_rubik_data(self, symbol=config.SYMBOL, period='1H'):
        """一次性拉取 OI / taker / 多空比,返回 add_rubik_features 期望的 dict。"""
        return {
            "open_interest": self.fetch_open_interest_history(symbol=symbol, period=period),
            "taker_volume": self.fetch_taker_volume_history(symbol=symbol, period=period),
            "long_short_ratio": self.fetch_long_short_ratio_history(symbol=symbol, period=period),
        }

    ### 封装开仓/平仓逻辑(按usdt开仓)
    def place_order_with_leverage(self, side, posSide, usd_amount, leverage, reduce_only=False, max_retry=3, sleep_sec=1, fill_timeout_sec=5.0):
        if not isinstance(usd_amount, (int, float)):
            try:
                usd_amount = float(usd_amount)
            except Exception:
                raise Exception(f"❌ usd_amount 类型异常: 传入了无法转换的值 '{usd_amount}'")
        cl_ord_id = build_client_order_id(config.SYMBOL, side, posSide, reduce_only)
        cached_size = None
        for attempt in range(max_retry):
            # 重试时先查既有订单：避免重复下单（幂等保护）
            existing_order = self.get_order_by_client_id(cl_ord_id)
            if order_is_filled(existing_order):
                enriched = self._attach_order_fills(existing_order)
                log_info(f"✅ 订单已成交（查询确认）: clOrdId={cl_ord_id}, state={existing_order.get('state')}")
                return enriched
            if order_is_acknowledged(existing_order):
                final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                if final_order is not None:
                    log_info(f"✅ 订单已受理后成交: clOrdId={cl_ord_id}")
                    return final_order

            try:
                # 使用缓存 size 保证幂等（重试时 size 与首次相同）
                if cached_size is None:
                    market_price = self.get_price()

                    # ✅ 资金安全校验 (账户可用保证金检查)
                    if not reduce_only:
                        account_info = self.get_account_balance()
                        available_usdt = float(account_info['data'][0]['availEq'])
                        usable_margin = max(
                            0.0,
                            available_usdt - float(config.LIVE_MIN_FREE_MARGIN_USDT),
                        ) * max(0.0, min(float(config.LIVE_MARGIN_USAGE_RATIO), 1.0))

                        if usd_amount > usable_margin:
                            log_info(
                                f"保证金约束: requested_margin={usd_amount:.2f} USDT, "
                                f"capped_margin={usable_margin:.2f} USDT, avail={available_usdt:.2f} USDT"
                            )
                            usd_amount = usable_margin

                        if usd_amount <= 0:
                            log_error(f"❌ 保证金不足: 可用 {available_usdt:.2f} USDT，取消下单")
                            return False

                    # ✅ 直接读取写死的合约参数
                    lot_size = config.LOT_SIZE
                    tick_size = config.TICK_SIZE

                    # ✅ 合法计算下单数量（注意保险性精度控制）
                    order_value = usd_amount * leverage
                    raw_size = order_value / market_price
                    size = math.floor(raw_size / lot_size) * lot_size
                    size = round(size, 6)

                    if size < lot_size:
                        if reduce_only:
                            log_info(f"🟡 平仓 size={size} 小于最小下单单位 {lot_size}，自动跳过")
                            return False
                        else:
                            raise Exception(f"⚠ 下单失败: 开仓 size={size} 小于最小下单单位 {lot_size}")

                    cached_size = size
                else:
                    size = cached_size

                # ✅ 发单
                result = self.trade_api.place_order(
                    instId=config.SYMBOL,
                    tdMode="cross",
                    side=side,
                    posSide=posSide,
                    ordType="market",
                    sz=str(size),
                    reduceOnly=reduce_only,
                    clOrdId=cl_ord_id,
                )

                if result['code'] == "0":
                    order_id = result['data'][0]['ordId']
                    log_info(
                        f"✅ 下单已提交: {side} {posSide} 杠杆: {leverage}x, 本金: {usd_amount} USD, 下单数量: {size} {config.SYMBOL}, 订单ID: {order_id}, clOrdId: {cl_ord_id}")
                    final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                    if final_order is not None:
                        log_info(f"✅ 下单已成交: ordId={order_id}, state={final_order.get('state')}")
                        return final_order
                    log_error(f"⚠ 下单提交后 {fill_timeout_sec}s 内未确认成交，进入下一轮校验: clOrdId={cl_ord_id}")
                    time.sleep(sleep_sec)
                else:
                    existing_order = self.get_order_by_client_id(cl_ord_id)
                    if order_is_filled(existing_order):
                        log_info(f"✅ 下单响应异常但订单已成交: clOrdId={cl_ord_id}")
                        return self._attach_order_fills(existing_order)
                    if order_is_acknowledged(existing_order):
                        final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                        if final_order is not None:
                            log_info(f"✅ 下单响应异常但订单已成交: clOrdId={cl_ord_id}")
                            return final_order
                    # ✅ 保险：防止无 data 崩溃
                    error_data = result.get('data', [{}])[0]
                    error_code = error_data.get('sCode', '')
                    error_msg = error_data.get('sMsg', '')
                    log_error(f"❌ 下单失败: 错误码 {error_code}, 原因: {error_msg}")
                    if is_insufficient_margin_error(result):
                        return False
                    time.sleep(sleep_sec)

            except Exception as e:
                existing_order = self.get_order_by_client_id(cl_ord_id)
                if order_is_filled(existing_order):
                    log_info(f"✅ 下单异常但订单已成交: clOrdId={cl_ord_id}")
                    return self._attach_order_fills(existing_order)
                if order_is_acknowledged(existing_order):
                    final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                    if final_order is not None:
                        log_info(f"✅ 下单异常但订单已成交: clOrdId={cl_ord_id}")
                        return final_order
                log_error(f"⚠ 下单异常({attempt + 1}): {e}")
                time.sleep(sleep_sec)

        # 超过重试次数后失败
        raise Exception("❌ 超过最大重试次数，下单失败")

    # 开多仓(按usdt)
    def open_long(self, usd_amount, leverage):
        return self.place_order_with_leverage("buy", "long", usd_amount, leverage, reduce_only=False)

    # 平多仓(按usdt)
    def close_long(self, usd_amount, leverage):
        long_pos, _ = self.get_position()
        if long_pos['size'] == 0:
            log_info("🟢 无多仓位，跳过平多")
            return False
        return self.place_order_with_leverage("sell", "long", usd_amount, leverage, reduce_only=True)

    # 开空仓(按usdt)
    def open_short(self, usd_amount, leverage):
        return self.place_order_with_leverage("sell", "short", usd_amount, leverage, reduce_only=False)

    # 平空仓(按usdt)
    def close_short(self, usd_amount, leverage):
        _, short_pos = self.get_position()
        if short_pos['size'] == 0:
            log_info("🟢 无空仓位，跳过平空")
            return False
        return self.place_order_with_leverage("buy", "short", usd_amount, leverage, reduce_only=True)

    ### 封装开仓/平仓逻辑(按size开仓)
    def place_order_with_size(self, side, posSide, size, leverage, reduce_only=False, max_retry=3, sleep_sec=1, fill_timeout_sec=5.0):
        """
        按"sz=size"直接下单，避免 usd_amount->size 二次floor，确保与回测 delta_qty 精确对齐。
        """
        if not isinstance(size, (int, float)):
            try:
                size = float(size)
            except Exception:
                raise Exception(f"❌ size 类型异常: '{size}'")

        lot_size = float(config.LOT_SIZE)
        size = floor_size_to_lot(size, lot_size)
        cl_ord_id = build_client_order_id(config.SYMBOL, side, posSide, reduce_only)

        if size < lot_size:
            if reduce_only:
                log_info(f"🟡 reduceOnly 平仓 size={size} 小于最小下单单位 {lot_size}，自动跳过")
                return False
            else:
                raise Exception(f"⚠ 下单失败: 开仓 size={size} 小于最小下单单位 {lot_size}")

        for attempt in range(max_retry):
            # 重试时先查既有订单：避免重复下单（幂等保护）
            existing_order = self.get_order_by_client_id(cl_ord_id)
            if order_is_filled(existing_order):
                log_info(f"✅ 订单已成交（查询确认，sz模式）: clOrdId={cl_ord_id}, state={existing_order.get('state')}")
                return self._attach_order_fills(existing_order)
            if order_is_acknowledged(existing_order):
                final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                if final_order is not None:
                    log_info(f"✅ 订单已受理后成交(sz模式): clOrdId={cl_ord_id}")
                    return final_order

            try:
                if not reduce_only:
                    market_price = self.get_price()

                    # 保证金检查：估算 required_margin = 名义价值 / leverage = size*price/leverage。
                    # 若目标 size 超过可用保证金，按可用额度截断，避免同一类信号反复失败重试。
                    account_info = self.get_account_balance()
                    available_usdt = float(account_info['data'][0]['availEq'])
                    capped_size, required_margin, usable_margin, was_capped = cap_size_by_available_margin(
                        size,
                        market_price,
                        leverage,
                        available_usdt,
                        lot_size,
                        usage_ratio=config.LIVE_MARGIN_USAGE_RATIO,
                        min_free_margin_usdt=config.LIVE_MIN_FREE_MARGIN_USDT,
                    )

                    if was_capped:
                        log_info(
                            f"保证金约束: requested_sz={size}, capped_sz={capped_size}, "
                            f"usable_margin={usable_margin:.2f} USDT, avail={available_usdt:.2f} USDT"
                        )
                        size = capped_size

                    if size < lot_size:
                        log_error(
                            f"❌ 保证金不足: 截断后 size={size} 小于最小下单单位 {lot_size}, "
                            f"usable_margin={usable_margin:.2f} USDT, 可用 {available_usdt:.2f} USDT，取消下单"
                        )
                        return False

                result = self.trade_api.place_order(
                    instId=config.SYMBOL,
                    tdMode="cross",
                    side=side,
                    posSide=posSide,
                    ordType="market",
                    sz=str(size),
                    reduceOnly=reduce_only,
                    clOrdId=cl_ord_id,
                )

                if result['code'] == "0":
                    order_id = result['data'][0]['ordId']
                    log_info(
                        f"✅ 下单已提交(sz模式): {side} {posSide} {leverage}x, sz={size}, reduceOnly={reduce_only}, ordId={order_id}, clOrdId={cl_ord_id}")
                    final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                    if final_order is not None:
                        log_info(f"✅ 下单已成交(sz模式): ordId={order_id}, state={final_order.get('state')}")
                        return final_order
                    log_error(f"⚠ 下单提交后 {fill_timeout_sec}s 内未确认成交(sz模式)，进入下一轮校验: clOrdId={cl_ord_id}")
                    time.sleep(sleep_sec)
                else:
                    existing_order = self.get_order_by_client_id(cl_ord_id)
                    if order_is_filled(existing_order):
                        log_info(f"✅ 下单响应异常但订单已成交(sz模式): clOrdId={cl_ord_id}")
                        return self._attach_order_fills(existing_order)
                    if order_is_acknowledged(existing_order):
                        final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                        if final_order is not None:
                            log_info(f"✅ 下单响应异常但订单已成交(sz模式): clOrdId={cl_ord_id}")
                            return final_order
                    error_data = result.get('data', [{}])[0]
                    error_code = error_data.get('sCode', '')
                    error_msg = error_data.get('sMsg', '')
                    log_error(f"❌ 下单失败(sz模式): 错误码 {error_code}, 原因: {error_msg}")
                    if is_insufficient_margin_error(result):
                        return False
                    time.sleep(sleep_sec)

            except Exception as e:
                existing_order = self.get_order_by_client_id(cl_ord_id)
                if order_is_filled(existing_order):
                    log_info(f"✅ 下单异常但订单已成交(sz模式): clOrdId={cl_ord_id}")
                    return self._attach_order_fills(existing_order)
                if order_is_acknowledged(existing_order):
                    final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                    if final_order is not None:
                        log_info(f"✅ 下单异常但订单已成交(sz模式): clOrdId={cl_ord_id}")
                        return final_order
                log_error(f"⚠ 下单异常(sz模式)({attempt + 1}): {e}")
                time.sleep(sleep_sec)

        raise Exception("❌ 超过最大重试次数，下单失败(sz模式)")

    def open_long_sz(self, sz, leverage):
        return self.place_order_with_size("buy", "long", sz, leverage, reduce_only=False)

    def close_long_sz(self, sz, leverage):
        long_pos, _ = self.get_position()
        if long_pos['size'] <= 0:
            log_info("🟢 无多仓位，跳过平多")
            return False
        close_size = min(float(sz), float(long_pos['size']))
        return self.place_order_with_size("sell", "long", close_size, leverage, reduce_only=True)

    def open_short_sz(self, sz, leverage):
        return self.place_order_with_size("sell", "short", sz, leverage, reduce_only=False)

    def close_short_sz(self, sz, leverage):
        _, short_pos = self.get_position()
        if short_pos['size'] <= 0:
            log_info("🟢 无空仓位，跳过平空")
            return False
        close_size = min(float(sz), float(short_pos['size']))
        return self.place_order_with_size("buy", "short", close_size, leverage, reduce_only=True)



if __name__ == '__main__':
    client = OKXClient()
    result = client.fetch_data()
    print(result)
