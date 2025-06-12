import math
import time
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

    # 每次投入 50 USDT 本金，使用 3 倍杠杆
    usd_amount = 50
    leverage = 3

    # # 开多示例：
    # client.open_long(usd_amount, leverage)

    # 开空示例：
    client.open_short(usd_amount, leverage)

