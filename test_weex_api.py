#!/usr/bin/env python3
"""
WEEX API 接口连通性测试脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.okx_api import OKXClient
from config import config
from core.weex_api import convert_symbol_to_weex_format

def test_api_connection():
    """测试API连接"""
    print("=" * 60)
    print("🔍 WEEX API 接口连通性测试")
    print("=" * 60)
    
    # 检查配置
    print("\n1️⃣ 检查配置:")
    print(f"   交易所: {config.EXCHANGE}")
    print(f"   API_KEY: {config.OKX_API_KEY[:15] if config.WEEX_API_KEY else '未设置'}...")
    print(f"   SECRET: {'已设置' if config.WEEX_SECRET else '未设置'}")
    print(f"   PASSWORD: {'已设置' if config.WEEX_PASSWORD else '未设置'}")
    print(f"   SYMBOL: {config.SYMBOL}")
    print(f"   WEEX格式: {convert_symbol_to_weex_format(config.SYMBOL)}")
    print(f"   BASE_URL: {config.WEEX_BASE_URL if hasattr(config, 'WEEX_BASE_URL') else '默认'}")
    
    if config.EXCHANGE != 'WEEX':
        print("\n⚠️  警告: EXCHANGE配置不是WEEX，当前为:", config.EXCHANGE)
        print("   请在.env中设置: EXCHANGE=WEEX")
        return
    
    # 初始化客户端
    print("\n2️⃣ 初始化客户端...")
    try:
        client = OKXClient()
        print("   ✅ 客户端初始化成功")
    except Exception as e:
        print(f"   ❌ 客户端初始化失败: {e}")
        return
    
    # 测试1: 获取价格
    print("\n3️⃣ 测试获取价格接口:")
    try:
        price = client.get_price()
        print(f"   ✅ 价格获取成功: {price}")
    except Exception as e:
        print(f"   ❌ 价格获取失败: {e}")
    
    # 测试2: 获取账户余额
    print("\n4️⃣ 测试账户余额接口:")
    try:
        balance = client.get_account_balance()
        if balance and 'data' in balance:
            total_eq = balance['data'][0].get('totalEq', 0)
            avail_eq = balance['data'][0].get('availEq', 0)
            print(f"   ✅ 账户余额获取成功")
            print(f"      总权益: {total_eq} USDT")
            print(f"      可用余额: {avail_eq} USDT")
        else:
            print(f"   ⚠️  响应格式异常: {balance}")
    except Exception as e:
        print(f"   ❌ 账户余额获取失败: {e}")
    
    # 测试3: 获取持仓
    print("\n5️⃣ 测试持仓查询接口:")
    try:
        long_pos, short_pos = client.get_position()
        print(f"   ✅ 持仓查询成功")
        print(f"      多仓: {long_pos['size']} 张, 成本价: {long_pos['entry_price']}")
        print(f"      空仓: {short_pos['size']} 张, 成本价: {short_pos['entry_price']}")
    except Exception as e:
        print(f"   ❌ 持仓查询失败: {e}")
    
    # 测试4: 获取K线数据
    print("\n6️⃣ 测试K线数据接口:")
    try:
        df = client.fetch_ohlcv(config.SYMBOL, bar='5m', max_limit=10)
        print(f"   ✅ K线数据获取成功")
        print(f"      数据条数: {len(df)}")
        print(f"      最新价格: {df['close'].iloc[-1]}")
        print(f"      时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")
    except Exception as e:
        print(f"   ❌ K线数据获取失败: {e}")
    
    # 测试5: 获取多周期数据
    print("\n7️⃣ 测试多周期数据接口:")
    try:
        data_dict = client.fetch_data()
        print(f"   ✅ 多周期数据获取成功")
        for interval, df in data_dict.items():
            print(f"      {interval}: {len(df)} 条K线")
    except Exception as e:
        print(f"   ❌ 多周期数据获取失败: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    test_api_connection()

