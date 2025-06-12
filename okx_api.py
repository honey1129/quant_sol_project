import time
import config
import okx.Account as Account
import okx.Trade as Trade
import okx.MarketData as Market

class OKXClient:
    def __init__(self):
        self.account_api = Account.AccountAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.trade_api = Trade.TradeAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.market_api = Market.MarketAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)

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

    def open_long(self, size):
        self.trade_api.place_order(instId=config.SYMBOL, tdMode='cross', side='buy', posSide='long', ordType='market', sz=str(size))

    def open_short(self, size):
        self.trade_api.place_order(instId=config.SYMBOL, tdMode='cross', side='sell', posSide='short', ordType='market', sz=str(size))

    def close_long(self, size):
        self.trade_api.place_order(instId=config.SYMBOL, tdMode='cross', side='sell', posSide='long', ordType='market', sz=str(size))

    def close_short(self, size):
        self.trade_api.place_order(instId=config.SYMBOL, tdMode='cross', side='buy', posSide='short', ordType='market', sz=str(size))

    def get_price(self,max_retry=3, sleep_sec=1):
        for attempt in range(max_retry):
            try:
                data = self.market_api.get_ticker(instId=config.SYMBOL)
                last_price = float(data['data'][0]['last'])
                return last_price
            except Exception as e:
                print(f"⚠ 获取价格失败，第{attempt + 1}次重试: {e}")
                time.sleep(sleep_sec)
        raise Exception("❌ 超过最大重试次数，get_price() 彻底失败")
