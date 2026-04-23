#!/usr/bin/env python3
"""
WEEX API 网络连接诊断脚本
"""
import sys
import os
import socket
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.weex_api import convert_symbol_to_weex_format

def test_network_connectivity():
    """测试网络连接"""
    print("=" * 60)
    print("🔍 WEEX API 网络连接诊断")
    print("=" * 60)
    
    base_url = config.WEEX_BASE_URL if hasattr(config, 'WEEX_BASE_URL') else "https://api-contract.weex.com"
    host = base_url.replace('https://', '').replace('http://', '').split('/')[0]
    
    print(f"\n1️⃣ 检查域名解析:")
    try:
        ip = socket.gethostbyname(host)
        print(f"   ✅ 域名解析成功: {host} -> {ip}")
    except Exception as e:
        print(f"   ❌ 域名解析失败: {e}")
        print(f"   💡 可能原因: DNS问题或网络不通")
        return
    
    print(f"\n2️⃣ 测试TCP连接:")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((ip, 443))
        sock.close()
        if result == 0:
            print(f"   ✅ TCP连接成功 (端口443)")
        else:
            print(f"   ❌ TCP连接失败 (错误码: {result})")
            print(f"   💡 可能原因: 防火墙阻止或IP被屏蔽")
    except Exception as e:
        print(f"   ❌ TCP连接测试失败: {e}")
    
    print(f"\n3️⃣ 测试HTTP连接 (无认证):")
    try:
        # 测试公共端点（可能不需要认证）
        test_urls = [
            f"{base_url}/capi/v2/public/info",
            f"{base_url}/capi/v2/market/ticker?symbol=cmt_btcusdt",
        ]
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=5)
                print(f"   ✅ {url}")
                print(f"      状态码: {response.status_code}")
                if response.status_code == 200:
                    print(f"      响应长度: {len(response.text)} 字符")
                    if len(response.text) < 200:
                        print(f"      响应内容: {response.text[:100]}")
                elif response.status_code == 521:
                    print(f"      ⚠️  521错误: Cloudflare错误，可能是IP被阻止")
                break
            except requests.exceptions.Timeout:
                print(f"   ❌ {url} - 连接超时")
            except requests.exceptions.ConnectionError as e:
                print(f"   ❌ {url} - 连接错误: {e}")
            except Exception as e:
                print(f"   ❌ {url} - 错误: {e}")
    except Exception as e:
        print(f"   ❌ HTTP测试失败: {e}")
    
    print(f"\n4️⃣ 检查当前IP地址:")
    try:
        ip_check = requests.get('https://api.ipify.org?format=json', timeout=5)
        current_ip = ip_check.json().get('ip', 'Unknown')
        print(f"   ✅ 当前公网IP: {current_ip}")
        print(f"   💡 请确认此IP是否在WEEX API白名单中")
    except Exception as e:
        print(f"   ⚠️  无法获取IP地址: {e}")
    
    print(f"\n5️⃣ 配置检查:")
    print(f"   交易所: {config.EXCHANGE}")
    print(f"   BASE_URL: {base_url}")
    print(f"   SYMBOL: {config.SYMBOL}")
    print(f"   WEEX格式: {convert_symbol_to_weex_format(config.SYMBOL)}")
    print(f"   API_KEY: {'已设置' if config.OKX_API_KEY else '未设置'}")
    
    print("\n" + "=" * 60)
    print("💡 诊断建议:")
    print("=" * 60)
    print("1. 如果域名解析失败: 检查DNS设置或网络连接")
    print("2. 如果TCP连接失败: 检查防火墙或使用VPN")
    print("3. 如果返回521错误: IP可能被WEEX阻止，需要添加到白名单")
    print("4. 如果连接超时: 可能需要使用代理或VPN")
    print("5. 检查WEEX账户中的API IP白名单设置")

if __name__ == '__main__':
    test_network_connectivity()

