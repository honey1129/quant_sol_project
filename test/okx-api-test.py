import ccxt
import config

exchange = ccxt.okx({
    'apiKey': config.OKX_API_KEY,
    'secret': config.OKX_SECRET,
    'password': config.OKX_PASSWORD,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})

if config.USE_SANDBOX:
    exchange.set_sandbox_mode(True)

print(exchange.fetch_balance())
