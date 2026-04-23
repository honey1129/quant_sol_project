#!/usr/bin/env python3
"""
使用Python生成curl命令来调用WEEX API账户余额接口
"""
import os
import time
import hmac
import hashlib
import base64
import json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量读取配置
API_KEY = os.getenv('WEEX_API_KEY')
API_SECRET = os.getenv('WEEX_SECRET')
PASSPHRASE = os.getenv('WEEX_PASSWORD')
BASE_URL = os.getenv('WEEX_BASE_URL', 'https://api-contract.weex.com')

def generate_signature(timestamp, method, request_path, body='', api_secret=None):
    """生成WEEX API签名"""
    api_secret = api_secret or API_SECRET
    message = str(timestamp) + method + request_path + body
    mac = hmac.new(
        bytes(api_secret, encoding='utf8'),
        bytes(message, encoding='utf8'),
        digestmod=hashlib.sha256
    )
    return base64.b64encode(mac.digest()).decode()

def generate_curl_command():
    """生成curl命令"""
    # 生成时间戳（13位毫秒）
    timestamp = str(int(time.time() * 1000))
    
    # 请求参数
    method = 'GET'
    request_path = '/capi/v2/account/assets'
    body = ''
    
    # 生成签名
    signature = generate_signature(timestamp, method, request_path, body)
    
    # 构造URL
    url = f"{BASE_URL}{request_path}"
    
    # 生成curl命令
    curl_cmd = f"""curl -X GET "{url}" \\
  -H "ACCESS-KEY: {API_KEY}" \\
  -H "ACCESS-SIGN: {signature}" \\
  -H "ACCESS-PASSPHRASE: {PASSPHRASE}" \\
  -H "ACCESS-TIMESTAMP: {timestamp}" \\
  -H "Content-Type: application/json" \\
  -H "locale: zh-CN" """
    
    return curl_cmd, timestamp, signature

def test_with_requests():
    """使用requests库直接调用（对比验证）"""
    import requests
    
    timestamp = str(int(time.time() * 1000))
    method = 'GET'
    request_path = '/capi/v2/account/assets'
    body = ''
    
    signature = generate_signature(timestamp, method, request_path, body)
    
    url = f"{BASE_URL}{request_path}"
    headers = {
        'ACCESS-KEY': API_KEY,
        'ACCESS-SIGN': signature,
        'ACCESS-PASSPHRASE': PASSPHRASE,
        'ACCESS-TIMESTAMP': timestamp,
        'Content-Type': 'application/json',
        'locale': 'zh-CN'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        return response.json(), response.status_code
    except Exception as e:
        return {'error': str(e)}, 0

if __name__ == '__main__':
    print("=" * 70)
    print("🔍 WEEX API 账户余额查询 - curl命令生成器")
    print("=" * 70)
    
    # 检查配置
    if not API_KEY or not API_SECRET or not PASSPHRASE:
        print("\n❌ 错误: 请先配置API密钥")
        print("   在.env文件中设置:")
        print("   OKX_API_KEY=your_api_key")
        print("   OKX_SECRET=your_secret")
        print("   OKX_PASSWORD=your_passphrase")
        exit(1)
    
    print(f"\n配置信息:")
    print(f"  API_KEY: {API_KEY[:15]}...")
    print(f"  BASE_URL: {BASE_URL}")
    
    # 生成curl命令
    curl_cmd, timestamp, signature = generate_curl_command()
    
    print(f"\n1️⃣ 时间戳: {timestamp}")
    print(f"2️⃣ 签名: {signature[:30]}...")
    
    print(f"\n3️⃣ 生成的curl命令:")
    print("-" * 70)
    print(curl_cmd)
    print("-" * 70)
    
    # 询问是否执行
    print(f"\n4️⃣ 使用requests库直接调用（验证）:")
    print("-" * 70)
    try:
        response_data, status_code = test_with_requests()
        print(f"HTTP状态码: {status_code}")
        print(f"响应内容:")
        print(json.dumps(response_data, indent=2, ensure_ascii=False))
        
        # 解析结果
        if 'data' in response_data and response_data['data']:
            print(f"\n5️⃣ 解析结果:")
            for asset in response_data['data']:
                currency = asset.get('currency') or asset.get('coin', '')
                balance = asset.get('balance') or asset.get('total', '0')
                available = asset.get('available') or asset.get('free', '0')
                print(f"  {currency}: 总余额={balance}, 可用={available}")
    except ImportError:
        print("⚠️  requests库未安装，跳过直接调用")
        print("   可以安装: pip install requests")
    except Exception as e:
        print(f"❌ 调用失败: {e}")
    
    print("\n" + "=" * 70)
    print("💡 提示:")
    print("   1. 复制上面的curl命令到终端执行")
    print("   2. 或者运行: bash test_curl_account.sh")
    print("=" * 70)

