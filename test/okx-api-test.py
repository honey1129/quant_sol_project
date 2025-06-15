from okx_api import OKXClient

client = OKXClient()

def test_instruments():
    try:
        instrument = client.public_api.get_instruments(instType="SWAP", instId="SOL-USDT-SWAP")
        print("✅ 接口调用成功")

        if not instrument['data']:
            print("❌ data 为空")
            return

        data_item = instrument['data'][0]
        print(f"lotSz: {data_item.get('lotSz')}")
        print(f"tickSz: {data_item.get('tickSz')}")
        print(f"完整返回数据: {data_item}")

        # 额外测试是否可以安全转 float
        lotSz_raw = data_item.get('lotSz', None)
        tickSz_raw = data_item.get('tickSz', None)

        if not lotSz_raw or not tickSz_raw:
            print("❌ lotSz 或 tickSz 为空字符串")
            return

        lot_size = float(lotSz_raw)
        tick_size = float(tickSz_raw)
        print(f"✅ 转换成功 lot_size: {lot_size}, tick_size: {tick_size}")

    except Exception as e:
        print(f"❌ 测试异常: {e}")

if __name__ == '__main__':
    test_instruments()
