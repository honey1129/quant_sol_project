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

    def get_account_balance(self):
        result = self.account_api.get_account_balance()

        # 防御性提取 totalEq（全账户权益）
        total_eq_raw = result['data'][0].get('totalEq', '0')
        total_eq = float(total_eq_raw) if total_eq_raw not in ['', None] else 0.0
        result['data'][0]['totalEq'] = total_eq

        # ✅ 重点：提取 USDT 子账户详情
        details = result['data'][0].get('details', [])
        usdt_detail = next((d for d in details if d.get('ccy') == 'USDT'), None)

        if usdt_detail:
            avail_eq_raw = usdt_detail.get('availEq', '0')
            avail_eq = float(avail_eq_raw) if avail_eq_raw not in ['', None] else 0.0
        else:
            avail_eq = 0.0

        result['data'][0]['availEq'] = avail_eq  # 重写主返回的 availEq 供后续兼容逻辑

        return result

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

    def fetch_ohlcv(self,symbol=config.SYMBOL, bar="1H", max_limit=2000, max_retry=3, sleep_sec=1):
        """
        OKX 历史K线完整拉取函数：支持自动分页、稳定拉取大规模历史数据
        """
        all_data = []
        next_after = ''  # ✅ 注意：首次使用空字符串

        while len(all_data) < max_limit:
            remaining = max_limit - len(all_data)
            limit = min(100, remaining)
            batch = None  # ✅ 提前初始化
            # 带重试逻辑
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

    def fetch_data(self):
        data_dict = {}
        for interval in config.INTERVALS:
            df = self.fetch_ohlcv(config.SYMBOL, bar=interval, max_limit=config.WINDOWS[interval])
            df.set_index("timestamp", inplace=True)
            data_dict[interval] = df
            time.sleep(0.3)
        return data_dict

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



    ### 封装开仓/平仓逻辑（实盘高复用接口）
    # 开多仓
    def open_long(self, usd_amount, leverage):
        self.place_order_with_leverage("buy", "long", usd_amount, leverage, reduce_only=False)

    # 平多仓
    def close_long(self, usd_amount, leverage):
        long_pos, _ = self.get_position()
        if long_pos['size'] == 0:
            log_info("🟢 无多仓位，跳过平多")
            return
        self.place_order_with_leverage("sell", "long", usd_amount, leverage, reduce_only=True)


    # 开空仓
    def open_short(self, usd_amount, leverage):
        self.place_order_with_leverage("sell", "short", usd_amount, leverage, reduce_only=False)

    # 平空仓
    def close_short(self, usd_amount, leverage):
        _, short_pos = self.get_position()
        if short_pos['size'] == 0:
            log_info("🟢 无空仓位，跳过平空")
            return
        self.place_order_with_leverage("buy", "short", usd_amount, leverage, reduce_only=True)

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


if __name__ == '__main__':
    client = OKXClient()
    result = client.fetch_data()
    print(result)