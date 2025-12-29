import math
import time
import pandas as pd
from config import config
import okx.Account as Account
import okx.Trade as Trade
import okx.MarketData as Market
import okx.PublicData as Public
from utils.utils import log_info, log_error

class OKXClient:
    def __init__(self):
        self.account_api = Account.AccountAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.trade_api = Trade.TradeAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.market_api = Market.MarketAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.public_api = Public.PublicAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True,flag=config.USE_SERVER)

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
            limit = min(100, remaining)
            batch = None
            for attempt in range(max_retry):
                try:
                    response = self.market_api.get_candlesticks(
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

        # 转换为DataFrame
        all_data = list(reversed(all_data))  # 最终按时间升序
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame([row[:6] for row in all_data], columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
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

    ### 封装开仓/平仓逻辑(按usdt开仓)
    def place_order_with_leverage(self, side, posSide, usd_amount, leverage,reduce_only=False, max_retry=3, sleep_sec=1):
        if not isinstance(usd_amount, (int, float)):
            try:
                usd_amount = float(usd_amount)
            except Exception:
                raise Exception(f"❌ usd_amount 类型异常: 传入了无法转换的值 '{usd_amount}'")
        for attempt in range(max_retry):
            try:
                market_price = self.get_price()

                # ✅ 资金安全校验 (账户可用保证金检查)
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

                # ✅ 发单
                result = self.trade_api.place_order(
                    instId=config.SYMBOL,
                    tdMode="cross",
                    side=side,
                    posSide=posSide,
                    ordType="market",
                    sz=str(size),
                    reduceOnly=reduce_only
                )

                if result['code'] == "0":
                    order_id = result['data'][0]['ordId']
                    log_info(
                        f"✅ 下单成功: {side} {posSide} 杠杆: {leverage}x, 本金: {usd_amount} USD, 下单数量: {size} {config.SYMBOL}, 订单ID: {order_id}")
                    return True
                else:
                    # ✅ 保险：防止无 data 崩溃
                    error_data = result.get('data', [{}])[0]
                    error_code = error_data.get('sCode', '')
                    error_msg = error_data.get('sMsg', '')
                    log_error(f"❌ 下单失败: 错误码 {error_code}, 原因: {error_msg}")
                    time.sleep(sleep_sec)

            except Exception as e:
                log_error(f"⚠ 下单异常({attempt + 1}): {e}")
                time.sleep(sleep_sec)

        # 超过重试次数后失败
        raise Exception("❌ 超过最大重试次数，下单失败")

    # 开多仓(按usdt)
    def open_long(self, usd_amount, leverage):
        self.place_order_with_leverage("buy", "long", usd_amount, leverage, reduce_only=False)

    # 平多仓(按usdt)
    def close_long(self, usd_amount, leverage):
        long_pos, _ = self.get_position()
        if long_pos['size'] == 0:
            log_info("🟢 无多仓位，跳过平多")
            return
        self.place_order_with_leverage("sell", "long", usd_amount, leverage, reduce_only=True)

    # 开空仓(按usdt)
    def open_short(self, usd_amount, leverage):
        self.place_order_with_leverage("sell", "short", usd_amount, leverage, reduce_only=False)

    # 平空仓(按usdt)
    def close_short(self, usd_amount, leverage):
        _, short_pos = self.get_position()
        if short_pos['size'] == 0:
            log_info("🟢 无空仓位，跳过平空")
            return
        self.place_order_with_leverage("buy", "short", usd_amount, leverage, reduce_only=True)

    ### 封装开仓/平仓逻辑(按size开仓)
    def place_order_with_size(self, side, posSide, size, leverage, reduce_only=False, max_retry=3, sleep_sec=1):
        """
        按“sz=size”直接下单，避免 usd_amount->size 二次floor，确保与回测 delta_qty 精确对齐。
        """
        if not isinstance(size, (int, float)):
            try:
                size = float(size)
            except Exception:
                raise Exception(f"❌ size 类型异常: '{size}'")

        lot_size = float(config.LOT_SIZE)
        size = math.floor(size / lot_size) * lot_size
        size = round(size, 6)

        if size < lot_size:
            if reduce_only:
                log_info(f"🟡 reduceOnly 平仓 size={size} 小于最小下单单位 {lot_size}，自动跳过")
                return False
            else:
                raise Exception(f"⚠ 下单失败: 开仓 size={size} 小于最小下单单位 {lot_size}")

        for attempt in range(max_retry):
            try:
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
                    reduceOnly=reduce_only
                )

                if result['code'] == "0":
                    order_id = result['data'][0]['ordId']
                    log_info(
                        f"✅ 下单成功(sz模式): {side} {posSide} {leverage}x, sz={size}, reduceOnly={reduce_only}, ordId={order_id}")
                    return True
                else:
                    error_data = result.get('data', [{}])[0]
                    error_code = error_data.get('sCode', '')
                    error_msg = error_data.get('sMsg', '')
                    log_error(f"❌ 下单失败(sz模式): 错误码 {error_code}, 原因: {error_msg}")
                    time.sleep(sleep_sec)

            except Exception as e:
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
        return self.place_order_with_size("sell", "long", sz, leverage, reduce_only=True)

    def open_short_sz(self, sz, leverage):
        return self.place_order_with_size("sell", "short", sz, leverage, reduce_only=False)

    def close_short_sz(self, sz, leverage):
        _, short_pos = self.get_position()
        if short_pos['size'] <= 0:
            log_info("🟢 无空仓位，跳过平空")
            return False
        return self.place_order_with_size("buy", "short", sz, leverage, reduce_only=True)



if __name__ == '__main__':
    client = OKXClient()
    result = client.fetch_data()
    print(result)