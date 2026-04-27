import math
import time
import uuid
import pandas as pd
from config import config
import okx.Account as Account
import okx.Trade as Trade
import okx.MarketData as Market
import okx.PublicData as Public
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


class OKXClient:
    def __init__(self):
        self.account_api = Account.AccountAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.trade_api = Trade.TradeAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.market_api = Market.MarketAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.public_api = Public.PublicAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True,flag=config.USE_SERVER)

    def get_account_config(self):
        result = self.account_api.get_account_config()
        if result.get("code") != "0":
            raise RuntimeError(f"获取账户配置失败: {result}")
        data = result.get("data", [])
        if not data:
            raise RuntimeError("账户配置为空，无法继续")
        return data[0]

    def list_pending_orders(self):
        result = self.trade_api.get_order_list(
            instType="SWAP",
            instId=config.SYMBOL,
        )
        if result.get("code") != "0":
            raise RuntimeError(f"获取挂单列表失败: {result}")
        return result.get("data", [])

    def get_order_by_client_id(self, cl_ord_id):
        if not cl_ord_id:
            return None
        try:
            result = self.trade_api.get_order(instId=config.SYMBOL, clOrdId=cl_ord_id)
        except Exception as exc:
            log_error(f"按 clOrdId 查询订单失败: {cl_ord_id} -> {exc}")
            return None

        if result.get("code") != "0":
            return None
        data = result.get("data", [])
        return data[0] if data else None

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
                    return order
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

            result = self.account_api.set_position_mode("long_short_mode")
            if result.get("code") != "0":
                raise RuntimeError(f"自动设置持仓模式失败: {result}")
            log_info("已自动切换到账户双向持仓模式 long_short_mode")
            time.sleep(0.5)
            account_config = self.get_account_config()
            pos_mode = str(account_config.get("posMode", "") or "").lower()
            if pos_mode != "long_short_mode":
                raise RuntimeError(f"持仓模式校验失败，当前为 {pos_mode or 'unknown'}")

        leverage_rows = self.account_api.get_leverage(mgnMode="cross", instId=config.SYMBOL)
        if leverage_rows.get("code") != "0":
            raise RuntimeError(f"获取杠杆配置失败: {leverage_rows}")
        leverage_map = self._extract_leverage_by_side(leverage_rows.get("data", []))
        target_leverage = float(config.LEVERAGE)

        if bool(config.LIVE_AUTO_SET_LEVERAGE):
            for pos_side in ("long", "short"):
                current_leverage = leverage_map.get(pos_side, leverage_map.get("both", 0.0))
                if abs(current_leverage - target_leverage) > 1e-9:
                    result = self.account_api.set_leverage(
                        lever=str(config.LEVERAGE),
                        mgnMode="cross",
                        instId=config.SYMBOL,
                        posSide=pos_side,
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
        result = self.account_api.get_account_balance()

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
        positions = self.account_api.get_positions(instType='SWAP', instId=config.SYMBOL)['data']

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
        result = self.account_api.get_positions_history(instType="SWAP", instId=config.SYMBOL, limit=str(limit))
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
                log_info(f"✅ 订单已成交（查询确认）: clOrdId={cl_ord_id}, state={existing_order.get('state')}")
                return True
            if order_is_acknowledged(existing_order):
                final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                if final_order is not None:
                    log_info(f"✅ 订单已受理后成交: clOrdId={cl_ord_id}")
                    return True

            try:
                # 使用缓存 size 保证幂等（重试时 size 与首次相同）
                if cached_size is None:
                    market_price = self.get_price()

                    # ✅ 资金安全校验 (账户可用保证金检查)
                    if not reduce_only:
                        account_info = self.get_account_balance()
                        available_usdt = float(account_info['data'][0]['availEq'])
                        required_margin = usd_amount  # cross模式下，本金即为保证金需求

                        if required_margin > available_usdt:
                            log_error(f"❌ 保证金不足: 需 {required_margin} USDT，可用 {available_usdt} USDT，取消下单")
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
                        return True
                    log_error(f"⚠ 下单提交后 {fill_timeout_sec}s 内未确认成交，进入下一轮校验: clOrdId={cl_ord_id}")
                    time.sleep(sleep_sec)
                else:
                    existing_order = self.get_order_by_client_id(cl_ord_id)
                    if order_is_filled(existing_order):
                        log_info(f"✅ 下单响应异常但订单已成交: clOrdId={cl_ord_id}")
                        return True
                    if order_is_acknowledged(existing_order):
                        final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                        if final_order is not None:
                            log_info(f"✅ 下单响应异常但订单已成交: clOrdId={cl_ord_id}")
                            return True
                    # ✅ 保险：防止无 data 崩溃
                    error_data = result.get('data', [{}])[0]
                    error_code = error_data.get('sCode', '')
                    error_msg = error_data.get('sMsg', '')
                    log_error(f"❌ 下单失败: 错误码 {error_code}, 原因: {error_msg}")
                    time.sleep(sleep_sec)

            except Exception as e:
                existing_order = self.get_order_by_client_id(cl_ord_id)
                if order_is_filled(existing_order):
                    log_info(f"✅ 下单异常但订单已成交: clOrdId={cl_ord_id}")
                    return True
                if order_is_acknowledged(existing_order):
                    final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                    if final_order is not None:
                        log_info(f"✅ 下单异常但订单已成交: clOrdId={cl_ord_id}")
                        return True
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
        size = math.floor(size / lot_size) * lot_size
        size = round(size, 6)
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
                return True
            if order_is_acknowledged(existing_order):
                final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                if final_order is not None:
                    log_info(f"✅ 订单已受理后成交(sz模式): clOrdId={cl_ord_id}")
                    return True

            try:
                if not reduce_only:
                    market_price = self.get_price()

                    # 保证金检查：估算 required_margin = 名义价值 / leverage = size*price/leverage
                    account_info = self.get_account_balance()
                    available_usdt = float(account_info['data'][0]['availEq'])
                    required_margin = (size * market_price) / float(leverage)

                    if required_margin > available_usdt:
                        log_error(f"❌ 保证金不足: 需 {required_margin:.2f} USDT，可用 {available_usdt:.2f} USDT，取消下单")
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
                        return True
                    log_error(f"⚠ 下单提交后 {fill_timeout_sec}s 内未确认成交(sz模式)，进入下一轮校验: clOrdId={cl_ord_id}")
                    time.sleep(sleep_sec)
                else:
                    existing_order = self.get_order_by_client_id(cl_ord_id)
                    if order_is_filled(existing_order):
                        log_info(f"✅ 下单响应异常但订单已成交(sz模式): clOrdId={cl_ord_id}")
                        return True
                    if order_is_acknowledged(existing_order):
                        final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                        if final_order is not None:
                            log_info(f"✅ 下单响应异常但订单已成交(sz模式): clOrdId={cl_ord_id}")
                            return True
                    error_data = result.get('data', [{}])[0]
                    error_code = error_data.get('sCode', '')
                    error_msg = error_data.get('sMsg', '')
                    log_error(f"❌ 下单失败(sz模式): 错误码 {error_code}, 原因: {error_msg}")
                    time.sleep(sleep_sec)

            except Exception as e:
                existing_order = self.get_order_by_client_id(cl_ord_id)
                if order_is_filled(existing_order):
                    log_info(f"✅ 下单异常但订单已成交(sz模式): clOrdId={cl_ord_id}")
                    return True
                if order_is_acknowledged(existing_order):
                    final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                    if final_order is not None:
                        log_info(f"✅ 下单异常但订单已成交(sz模式): clOrdId={cl_ord_id}")
                        return True
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
