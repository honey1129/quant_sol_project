#!/bin/bash
# WEEX API 账户余额查询 - curl方式调用
# 使用方法: bash test_curl_account.sh

# 从.env文件读取配置（需要先安装jq或手动设置）
# 或者直接在这里设置
API_KEY="${OKX_API_KEY:-your_api_key_here}"
API_SECRET="${OKX_SECRET:-your_api_secret_here}"
PASSPHRASE="${OKX_PASSWORD:-your_passphrase_here}"
BASE_URL="${WEEX_BASE_URL:-https://api-contract.weex.com}"

# 如果没有设置，尝试从.env读取
if [ "$API_KEY" == "your_api_key_here" ]; then
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
        API_KEY="${OKX_API_KEY}"
        API_SECRET="${OKX_SECRET}"
        PASSPHRASE="${OKX_PASSWORD}"
    fi
fi

# 检查配置
if [ "$API_KEY" == "your_api_key_here" ] || [ -z "$API_KEY" ]; then
    echo "❌ 错误: 请设置API密钥"
    echo "   方法1: 在.env文件中设置 OKX_API_KEY, OKX_SECRET, OKX_PASSWORD"
    echo "   方法2: 在脚本中直接设置环境变量"
    exit 1
fi

echo "============================================================"
echo "🔍 WEEX API 账户余额查询 (curl方式)"
echo "============================================================"
echo ""
echo "配置信息:"
echo "  API_KEY: ${API_KEY:0:15}..."
echo "  BASE_URL: $BASE_URL"
echo ""

# 1. 生成时间戳（13位毫秒）
TIMESTAMP=$(python3 -c "import time; print(int(time.time() * 1000))")
if [ $? -ne 0 ]; then
    # 如果Python不可用，使用date命令（10位秒，需要乘以1000）
    TIMESTAMP=$(($(date +%s) * 1000))
fi

echo "1️⃣ 生成时间戳: $TIMESTAMP"

# 2. 构造请求路径
METHOD="GET"
REQUEST_PATH="/capi/v2/account/assets"
BODY=""

# 3. 构造签名字符串: timestamp + method + request_path + body
MESSAGE="${TIMESTAMP}${METHOD}${REQUEST_PATH}${BODY}"

echo "2️⃣ 签名字符串: $MESSAGE"

# 4. 生成HMAC-SHA256签名
# 使用Python生成签名（更可靠）
SIGNATURE=$(python3 <<EOF
import hmac
import hashlib
import base64
import sys

api_secret = "$API_SECRET"
message = "$MESSAGE"

mac = hmac.new(
    bytes(api_secret, encoding='utf8'),
    bytes(message, encoding='utf8'),
    digestmod=hashlib.sha256
)
signature = base64.b64encode(mac.digest()).decode()
print(signature)
EOF
)

if [ $? -ne 0 ]; then
    echo "❌ 签名生成失败，请确保已安装Python3"
    exit 1
fi

echo "3️⃣ 生成签名: ${SIGNATURE:0:20}..."

# 5. 构造curl命令
URL="${BASE_URL}${REQUEST_PATH}"

echo ""
echo "4️⃣ 发送请求..."
echo "   URL: $URL"
echo ""

# 执行curl请求
RESPONSE=$(curl -s -X GET "$URL" \
  -H "ACCESS-KEY: $API_KEY" \
  -H "ACCESS-SIGN: $SIGNATURE" \
  -H "ACCESS-PASSPHRASE: $PASSPHRASE" \
  -H "ACCESS-TIMESTAMP: $TIMESTAMP" \
  -H "Content-Type: application/json" \
  -H "locale: zh-CN" \
  -w "\nHTTP_CODE:%{http_code}")

# 分离响应体和HTTP状态码
HTTP_CODE=$(echo "$RESPONSE" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
BODY=$(echo "$RESPONSE" | sed 's/HTTP_CODE:[0-9]*$//')

echo "============================================================"
echo "📥 响应结果"
echo "============================================================"
echo "HTTP状态码: $HTTP_CODE"
echo ""
echo "响应内容:"
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo ""

# 解析结果（如果响应是JSON）
if command -v python3 &> /dev/null; then
    echo "============================================================"
    echo "📊 解析结果"
    echo "============================================================"
    
    python3 <<EOF
import json
import sys

try:
    response = json.loads("""$BODY""")
    
    if 'code' in response:
        code = response.get('code')
        if code == '0' or code == 0:
            print("✅ API调用成功")
        else:
            print(f"❌ API错误: 错误码 {code}")
            print(f"   原因: {response.get('msg', 'Unknown error')}")
            sys.exit(1)
    
    if 'data' in response and response['data']:
        data = response['data']
        print(f"✅ 找到 {len(data)} 个资产")
        
        # 查找USDT资产
        for asset in data:
            currency = asset.get('currency') or asset.get('coin', '')
            balance = asset.get('balance') or asset.get('total', '0')
            available = asset.get('available') or asset.get('free', '0')
            
            print(f"\n资产: {currency}")
            print(f"  总余额: {balance}")
            print(f"  可用余额: {available}")
            
            if currency == 'USDT':
                print(f"\n💰 USDT资产:")
                print(f"  总权益: {balance} USDT")
                print(f"  可用余额: {available} USDT")
    else:
        print("⚠️  响应中没有data字段或data为空")
        print(f"完整响应: {json.dumps(response, indent=2, ensure_ascii=False)}")
        
except json.JSONDecodeError:
    print("❌ 响应不是有效的JSON格式")
    print(f"原始响应: $BODY")
except Exception as e:
    print(f"❌ 解析失败: {e}")
    print(f"原始响应: $BODY")
EOF
fi

echo ""
echo "============================================================"

