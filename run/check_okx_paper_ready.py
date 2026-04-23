from core.okx_api import OKXClient
from config import config


def main():
    client = OKXClient()
    client.ensure_trading_ready()
    long_pos, short_pos = client.get_position()
    pending_orders = client.list_pending_orders()
    balance = client.get_account_balance()["data"][0]

    print("paper_ready_ok")
    print(f"symbol={config.SYMBOL}")
    print(f"use_server={config.USE_SERVER}")
    print(f"leverage={config.LEVERAGE}")
    print(f"long_size={float(long_pos['size'])}")
    print(f"short_size={float(short_pos['size'])}")
    print(f"pending_orders={len(pending_orders)}")
    print(f"availEq={float(balance.get('availEq', 0) or 0)}")
    print(f"totalEq={float(balance.get('totalEq', 0) or 0)}")


if __name__ == "__main__":
    main()
