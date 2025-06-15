import math
import time
import pandas as pd
import config
import okx.Account as Account
import okx.Trade as Trade
import okx.MarketData as Market
import okx.PublicData as Public
from utils import log_info, log_error

class OKXClient:
    def __init__(self):
        self.account_api = Account.AccountAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.trade_api = Trade.TradeAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.market_api = Market.MarketAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.public_api = Public.PublicAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True,flag=config.USE_SERVER)

    def get_account_balance(self):
        result = self.account_api.get_account_balance()
        return result

    def get_position(self):
        positions = self.account_api.get_positions(instType='SWAP', instId=config.SYMBOL)['data']
        for pos in positions:
            size = float(pos['pos'])
            avg_price = float(pos['avgPx']) if pos['avgPx'] else 0
            if size > 0:
                return 'long', size, avg_price
            elif size < 0:
                return 'short', abs(size), avg_price
        return 'none', 0, 0

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

    def place_order_with_leverage(self, side, posSide, usd_amount, leverage, max_retry=3, sleep_sec=1):
        for attempt in range(max_retry):
            try:
                market_price = self.get_price()

                # 关键1：实时获取合约信息
                instrument = self.public_api.get_instruments(instType="SWAP", instId=config.SYMBOL)
                lot_size = float(instrument['data'][0]['lotSz'])
                tick_size = float(instrument['data'][0]['tickSz'])

                # 计算目标下单币数量
                order_value = usd_amount * leverage
                raw_size = order_value / market_price

                # 关键2：做合法精度修正 (向下取整)
                size = math.floor(raw_size / lot_size) * lot_size
                size = round(size, 6)  # 保险性控制小数位

                if size < lot_size:
                    raise Exception(f"⚠ 下单失败：换算后 size = {size} 小于最小下单单位 lot_size = {lot_size}")

                result = self.trade_api.place_order(
                    instId=config.SYMBOL,
                    tdMode="cross",
                    side=side,
                    posSide=posSide,
                    ordType="market",
                    sz=str(size)
                )

                if result['code'] == "0":
                    order_id = result['data'][0]['ordId']
                    log_info(
                        f"✅ 下单成功: {side} {posSide} 杠杆: {leverage}x, 本金: {usd_amount} USD, 下单数量: {size} {config.SYMBOL}, 订单ID: {order_id}")
                    return True
                else:
                    error_code = result['data'][0]['sCode']
                    error_msg = result['data'][0]['sMsg']
                    log_error(f"❌ 下单失败: 错误码 {error_code}, 原因: {error_msg}")
                    time.sleep(sleep_sec)
            except Exception as e:
                log_error(f"⚠ 下单异常({attempt + 1}): {e}")
                time.sleep(sleep_sec)
        raise Exception("❌ 超过最大重试次数，下单失败")



    ### 封装开仓/平仓逻辑（实盘高复用接口）
    def open_long(self, usd_amount, leverage):
        self.place_order_with_leverage("buy", "long", usd_amount, leverage)

    def open_short(self, usd_amount, leverage):
        self.place_order_with_leverage("sell", "short", usd_amount, leverage)

    def close_long(self, usd_amount, leverage):
        self.place_order_with_leverage("sell", "long", usd_amount, leverage)

    def close_short(self, usd_amount, leverage):
        self.place_order_with_leverage("buy", "short", usd_amount, leverage)


    def get_price(self,max_retry=3, sleep_sec=1):
        for attempt in range(max_retry):
            try:
                data = self.market_api.get_ticker(instId=config.SYMBOL)
                last_price = float(data['data'][0]['last'])
                return last_price
            except Exception as e:
                log_error(f"⚠ 获取价格失败，第{attempt + 1}次重试: {e}")
                time.sleep(sleep_sec)
        raise Exception("❌ 超过最大重试次数，get_price() 彻底失败")


if __name__ == '__main__':
    client = OKXClient()
    result = client.fetch_data()
    print(result)