import ccxt
import config
from utils import send_telegram

def init_okx():
    try:
        exchange = ccxt.okx({
            'apiKey': config.OKX_API_KEY,
            'secret': config.OKX_SECRET,
            'password': config.OKX_PASSWORD,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap',
                        'defaultMarket': 'swap'}
        })

        if config.USE_SERVER == "1":
            exchange.set_sandbox_mode(True)
            send_telegram("✅ 当前运行环境：OKX 模拟盘")
        else:
            send_telegram("✅ 当前运行环境：OKX 实盘")

        return exchange

    except Exception as e:
        send_telegram(f"❌ 交易所初始化失败: {e}")
        raise e
