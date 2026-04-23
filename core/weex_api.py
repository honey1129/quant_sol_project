"""
WEEX API 客户端实现
基于 WEEX API 文档: https://www.weex.com/news/detail/ai-wars-weex-alpha-awakens-weex-global-hackathon-api-test-process-guide-264335
"""
import time
import hmac
import hashlib
import base64
import requests
import pandas as pd
from config import config
from utils.utils import log_info, log_error

def convert_symbol_to_weex_format(symbol):
    """
    将标准交易对格式转换为WEEX格式
    
    Args:
        symbol: 标准格式，如 'DOGE-USDT-SWAP' 或 'BTC-USDT-SWAP'
    
    Returns:
        weex_symbol: WEEX格式，如 'cmt_dogeusdt' 或 'cmt_btcusdt'
    """
    # 移除 -SWAP 后缀，转换为小写，用下划线连接
    if symbol.endswith('-SWAP'):
        symbol = symbol[:-5]  # 移除 -SWAP
    parts = symbol.split('-')
    if len(parts) == 2:
        # BTC-USDT -> btcusdt -> cmt_btcusdt
        base, quote = parts[0].lower(), parts[1].lower()
        return f"cmt_{base}{quote}"
    return symbol.lower()

class WEEXClient:
    def __init__(self, api_key=None, api_secret=None, passphrase=None, base_url=None):
        """
        初始化 WEEX API 客户端
        
        Args:
            api_key: API密钥
            api_secret: API密钥Secret
            passphrase: API密钥Passphrase
            base_url: API基础URL，默认使用合约API
        """
        self.api_key = api_key or config.OKX_API_KEY
        self.api_secret = api_secret or config.OKX_SECRET
        self.passphrase = passphrase or config.OKX_PASSWORD
        # WEEX 合约API基础URL
        self.base_url = base_url or "https://api-contract.weex.com"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'locale': 'zh-CN'
        })
    
    def _generate_signature(self, timestamp, method, request_path, body=''):
        """
        生成 WEEX API 签名
        
        Args:
            timestamp: 时间戳
            method: HTTP方法 (GET, POST等)
            request_path: 请求路径
            body: 请求体（JSON字符串）
        
        Returns:
            signature: 签名字符串
        """
        message = str(timestamp) + method + request_path + body
        mac = hmac.new(
            bytes(self.api_secret, encoding='utf8'),
            bytes(message, encoding='utf8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    def _get_headers(self, method, request_path, body=''):
        """
        获取请求头
        
        Args:
            method: HTTP方法
            request_path: 请求路径
            body: 请求体
        
        Returns:
            headers: 请求头字典
        """
        # WEEX要求时间戳为13位毫秒
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, request_path, body)
        
        headers = {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-PASSPHRASE': self.passphrase,
            'ACCESS-TIMESTAMP': timestamp,
            'Content-Type': 'application/json',
            'locale': 'zh-CN'
        }
        return headers
    
    def _request(self, method, endpoint, params=None, data=None):
        """
        发送API请求
        
        Args:
            method: HTTP方法
            endpoint: API端点路径（包含查询参数）
            params: URL参数（备用，如果endpoint已包含参数则不使用）
            data: 请求体数据
        
        Returns:
            response: API响应
        """
        # 分离路径和查询参数
        if '?' in endpoint:
            path, query_string = endpoint.split('?', 1)
            url = f"{self.base_url}{endpoint}"
        else:
            path = endpoint
            query_string = ''
            if params:
                query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                url = f"{self.base_url}{endpoint}?{query_string}"
            else:
                url = f"{self.base_url}{endpoint}"
        
        body = ''
        if data:
            import json
            body = json.dumps(data)
        
        # 签名使用路径部分（不包含查询参数）
        headers = self._get_headers(method, path, body)
        
        try:
            # 设置超时时间（连接超时5秒，读取超时10秒）
            timeout = (5, 10)
            
            if method == 'GET':
                # GET请求的参数应该已经在endpoint中，或者通过params传递
                if params and '?' not in endpoint:
                    response = self.session.get(f"{self.base_url}{endpoint}", headers=headers, params=params, timeout=timeout)
                else:
                    response = self.session.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = self.session.post(url, headers=headers, json=data, params=params, timeout=timeout)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            log_error(f"API请求失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    return error_data
                except:
                    # 如果响应不是JSON，返回原始文本
                    error_text = e.response.text if hasattr(e.response, 'text') else str(e)
                    return {'code': '50000', 'msg': f"{str(e)}: {error_text}", 'data': None}
            return {'code': '50000', 'msg': str(e), 'data': None}
    
    def get_account_balance(self):
        """
        获取账户余额
        
        Returns:
            result: 账户余额信息
        """
        result = self._request('GET', '/capi/v2/account/assets')
        
        # ✅ WEEX实际响应格式：直接返回数组，不是 {code, data} 格式
        # 实际格式: [{"coinName": "USDT", "available": "1000.00", "equity": "1000.00", ...}]
        
        # 如果响应是列表（WEEX实际格式）
        if isinstance(result, list):
            weex_data = result
            total_eq = 0.0
            avail_eq = 0.0
            
            for asset in weex_data:
                # WEEX使用 coinName 字段
                coin_name = asset.get('coinName') or asset.get('currency') or asset.get('coin', '')
                if coin_name == 'USDT':
                    # WEEX使用 equity 表示总权益，available 表示可用余额
                    total_eq = float(asset.get('equity', 0) or asset.get('balance', 0) or asset.get('total', 0) or 0)
                    avail_eq = float(asset.get('available', 0) or asset.get('free', 0) or 0)
                    break
            
            return {
                'data': [{
                    'totalEq': total_eq,
                    'availEq': avail_eq,
                    'details': weex_data
                }]
            }
        
        # ✅ 兼容旧格式：如果响应是字典格式 {code, data}
        if isinstance(result, dict):
            # 检查错误码
            if 'code' in result and result.get('code') != '0' and result.get('code') != 0:
                error_msg = result.get('msg', 'Unknown error')
                error_code = result.get('code', 'Unknown')
                log_error(f"❌ 获取账户余额失败: 错误码 {error_code}, 原因: {error_msg}")
                return {'data': [{'totalEq': 0.0, 'availEq': 0.0}]}
            
            # 如果有data字段
            if 'data' in result and result['data']:
                weex_data = result['data']
                total_eq = 0.0
                avail_eq = 0.0
                
                for asset in weex_data:
                    coin_name = asset.get('coinName') or asset.get('currency') or asset.get('coin', '')
                    if coin_name == 'USDT':
                        total_eq = float(asset.get('equity', 0) or asset.get('balance', 0) or asset.get('total', 0) or 0)
                        avail_eq = float(asset.get('available', 0) or asset.get('free', 0) or 0)
                        break
                
                return {
                    'data': [{
                        'totalEq': total_eq,
                        'availEq': avail_eq,
                        'details': weex_data
                    }]
                }
        
        # 默认返回
        log_error(f"⚠️ 未知的响应格式: {type(result)}, 内容: {result}")
        return {'data': [{'totalEq': 0.0, 'availEq': 0.0}]}
    
    def get_position(self):
        """
        获取持仓信息
        
        Returns:
            (long_position, short_position): 多空持仓信息
        """
        # 尝试多个可能的持仓端点
        possible_endpoints = [
            '/capi/v2/account/accounts',  # 根据文档，可能是这个
            '/capi/v2/account/position',  # 单数形式
            '/capi/v2/account/positions',  # 复数形式
        ]
        
        result = None
        for endpoint in possible_endpoints:
            try:
                log_info(f"🔍 尝试持仓端点: {endpoint}")
                result = self._request('GET', endpoint)
                
                # 检查是否成功
                if isinstance(result, dict):
                    # /capi/v2/account/accounts 成功时返回 {'account': {...}, 'position': [...], ...}
                    if 'position' in result or 'account' in result:
                        log_info(f"✅ 使用端点: {endpoint}")
                        break
                    # 检查错误码
                    code = result.get('code')
                    if code in ['0', 0, 200]:
                        log_info(f"✅ 使用端点: {endpoint}")
                        break
                    # 如果是404，尝试下一个端点
                    if code in ['404', 404]:
                        continue
                elif isinstance(result, list):
                    # 如果直接返回列表，也认为成功
                    log_info(f"✅ 使用端点: {endpoint}")
                    break
            except Exception as e:
                log_error(f"⚠️ 端点 {endpoint} 失败: {e}")
                continue
        
        if result is None:
            log_error("❌ 所有持仓端点都失败")
            result = {'position': []}  # 返回空持仓结构
        
        long_position = {'size': 0.0, 'entry_price': 0.0}
        short_position = {'size': 0.0, 'entry_price': 0.0}
        
        # ✅ 检查API响应是否成功
        if result is None:
            log_error("❌ 获取持仓失败: 所有端点都失败")
            return long_position, short_position
        
        # WEEX /capi/v2/account/accounts 响应格式: {'account': {...}, 'position': [...], ...}
        # 持仓信息在 'position' 字段中（是一个列表）
        if isinstance(result, dict):
            # 如果响应包含 position 字段（WEEX标准格式）
            if 'position' in result:
                positions = result['position']
                if isinstance(positions, list) and len(positions) > 0:
                    log_info(f"✅ 找到 {len(positions)} 个持仓")
                    for pos in positions:
                        # WEEX持仓格式解析
                        pos_side = pos.get('side', '').lower() or pos.get('positionSide', '').lower() or pos.get('direction', '').lower()
                        size_raw = pos.get('size', '0') or pos.get('position', '0') or pos.get('quantity', '0') or pos.get('amount', '0')
                        avgPx_raw = pos.get('avgPrice', '0') or pos.get('entryPrice', '0') or pos.get('avg_price', '0') or pos.get('openPrice', '0') or pos.get('price', '0')
                        
                        size = float(size_raw) if size_raw not in ['', None] else 0.0
                        avg_price = float(avgPx_raw) if avgPx_raw not in ['', None] else 0.0
                        
                        if pos_side in ['long', 'buy', '1']:
                            long_position['size'] = size
                            long_position['entry_price'] = avg_price
                            log_info(f"   多仓: {size} 张, 成本价: {avg_price}")
                        elif pos_side in ['short', 'sell', '2']:
                            short_position['size'] = size
                            short_position['entry_price'] = avg_price
                            log_info(f"   空仓: {size} 张, 成本价: {avg_price}")
                elif isinstance(positions, list):
                    log_info("✅ 当前无持仓")
            
            # 兼容其他格式：如果响应包含 data 字段
            elif 'data' in result and result['data']:
                positions = result['data'] if isinstance(result['data'], list) else [result['data']]
                for pos in positions:
                    pos_side = pos.get('side', '').lower() or pos.get('positionSide', '').lower()
                    size_raw = pos.get('size', '0') or pos.get('position', '0') or pos.get('quantity', '0')
                    avgPx_raw = pos.get('avgPrice', '0') or pos.get('entryPrice', '0') or pos.get('avg_price', '0') or pos.get('openPrice', '0')
                    
                    size = float(size_raw) if size_raw not in ['', None] else 0.0
                    avg_price = float(avgPx_raw) if avgPx_raw not in ['', None] else 0.0
                    
                    if pos_side in ['long', 'buy', '1']:
                        long_position['size'] = size
                        long_position['entry_price'] = avg_price
                    elif pos_side in ['short', 'sell', '2']:
                        short_position['size'] = size
                        short_position['entry_price'] = avg_price
        
        return long_position, short_position
    
    def get_price(self, symbol=None):
        """
        获取当前价格
        
        根据WEEX文档：https://www.weex.com/news/detail/ai-wars-weex-alpha-awakens-weex-global-hackathon-api-test-process-guide-264335
        响应格式是直接的JSON对象，包含 'last' 字段
        
        Args:
            symbol: 交易对符号，默认使用config.SYMBOL
        
        Returns:
            price: 当前价格
        """
        symbol = symbol or config.SYMBOL
        # 转换为WEEX格式：DOGE-USDT-SWAP -> cmt_dogeusdt
        weex_symbol = convert_symbol_to_weex_format(symbol)
        
        # 根据WEEX文档，端点是 /capi/v2/market/ticker
        endpoint = f'/capi/v2/market/ticker?symbol={weex_symbol}'
        
        try:
            log_info(f"🔍 请求价格: {endpoint}")
            result = self._request('GET', endpoint)
            log_info(f"📥 收到响应: {type(result)}, keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            
            # 根据WEEX文档，响应是直接的JSON对象，不是包装在data中的
            # 响应格式：{"last": "86639.8", "symbol": "cmt_btcusdt", ...}
            if isinstance(result, dict):
                # 如果有code字段且不是成功码，说明是错误响应
                if 'code' in result:
                    code = result.get('code')
                    if code not in ['0', 0, 200, None]:
                        error_msg = result.get('msg', 'Unknown error')
                        raise Exception(f"API错误: 错误码 {code}, 原因: {error_msg}")
                
                # 直接提取last字段（根据文档示例）
                price_raw = result.get('last', '0') or result.get('price', '0')
                if price_raw not in ['', None, '0']:
                    return float(price_raw)
                
                # 如果有data字段，尝试从data中提取
                if 'data' in result:
                    data = result['data']
                    if isinstance(data, dict):
                        price_raw = data.get('last', '0') or data.get('price', '0')
                        if price_raw not in ['', None, '0']:
                            return float(price_raw)
                    elif isinstance(data, list) and len(data) > 0:
                        ticker = data[0]
                        price_raw = ticker.get('last', '0') or ticker.get('price', '0')
                        if price_raw not in ['', None, '0']:
                            return float(price_raw)
            
            raise Exception(f"❌ 无法从响应中提取价格。响应: {result}")
            
        except Exception as e:
            log_error(f"获取价格失败: {e}")
            raise
    
    def fetch_ohlcv(self, symbol=None, bar="1H", max_limit=2000, max_retry=3, sleep_sec=1):
        """
        获取K线数据
        
        Args:
            symbol: 交易对符号
            bar: K线周期 (1m, 5m, 15m, 1H, 4H, 1D等)
            max_limit: 最大获取数量
            max_retry: 最大重试次数
            sleep_sec: 重试间隔
        
        Returns:
            df: K线数据DataFrame
        """
        symbol = symbol or config.SYMBOL
        # 转换为WEEX格式：DOGE-USDT-SWAP -> cmt_dogeusdt
        weex_symbol = convert_symbol_to_weex_format(symbol)
        
        # WEEX K线周期映射（granularity参数使用简洁格式：5m, 1h等）
        period_map = {
            '1m': '1m',      # 1分钟
            '5m': '5m',      # 5分钟
            '15m': '15m',    # 15分钟
            '30m': '30m',    # 30分钟
            '1H': '1h',      # 1小时
            '1h': '1h',      # 1小时
            '4H': '4h',      # 4小时
            '4h': '4h',      # 4小时
            '1D': '1d',      # 1天
            '1d': '1d',      # 1天
        }
        # 使用简洁格式（5m而不是5min）
        period = period_map.get(bar, bar)
        
        all_data = []
        limit = min(200, max_limit)  # WEEX单次请求限制
        
        # WEEX K线端点（根据测试，使用granularity参数而不是period）
        # 端点路径: /capi/v2/market/candles
        # 参数: symbol, granularity (而不是period), limit
        endpoint = f'/capi/v2/market/candles?symbol={weex_symbol}&granularity={period}&limit={limit}'
        
        for attempt in range(max_retry):
            try:
                log_info(f"🔍 请求K线数据: {endpoint}")
                response = self._request('GET', endpoint)
                
                # 检查错误
                if 'code' in response:
                    code = response.get('code')
                    if code not in ['0', 0, 200, None]:
                        error_msg = response.get('msg', 'Unknown error')
                        raise Exception(f"API错误: 错误码 {code}, 原因: {error_msg}")
                
                # 提取K线数据
                if 'data' in response and response['data']:
                    batch = response['data']
                    all_data.extend(batch if isinstance(batch, list) else [batch])
                    log_info(f"✅ K线数据获取成功: {len(all_data)}条")
                    break
                elif isinstance(response, list):
                    # 如果响应直接是数组
                    all_data.extend(response)
                    log_info(f"✅ K线数据获取成功: {len(all_data)}条")
                    break
            except Exception as e:
                log_error(f"⚠️ 拉取K线失败，重试中 ({attempt + 1}/{max_retry}): {e}")
                if attempt < max_retry - 1:
                    time.sleep(sleep_sec)
                else:
                    raise
        
        if not all_data:
            log_error("⚠️ 无法拉取K线数据，可能端点路径不正确。返回空DataFrame")
            # 返回空DataFrame而不是抛出异常，避免系统崩溃
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            return pd.DataFrame(columns=columns)
        
        # 转换为DataFrame（WEEX格式可能不同，需要根据实际调整）
        # 假设WEEX返回格式: [timestamp, open, high, low, close, volume]
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame([row[:6] for row in all_data], columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def place_order(self, side, pos_side, size, price=None, order_type='market', reduce_only=False):
        """
        下单
        
        Args:
            side: 买卖方向 ('buy' or 'sell')
            pos_side: 持仓方向 ('long' or 'short')
            size: 下单数量
            price: 价格（限价单需要）
            order_type: 订单类型 ('market' or 'limit')
            reduce_only: 是否只减仓
        
        Returns:
            result: 下单结果
        """
        # WEEX下单端点（需要根据实际API文档调整）
        endpoint = '/capi/v2/order/place'
        
        # 转换为WEEX格式
        weex_symbol = convert_symbol_to_weex_format(config.SYMBOL)
        
        # WEEX下单参数（根据WEEX API文档调整）
        order_data = {
            'symbol': weex_symbol,
            'side': side,  # 'buy' or 'sell'
            'positionSide': pos_side,  # 'long' or 'short'
            'type': order_type,  # 'market' or 'limit'
            'quantity': str(size),
            'reduceOnly': reduce_only
        }
        
        if order_type == 'limit' and price:
            order_data['price'] = str(price)
        
        result = self._request('POST', endpoint, data=order_data)
        return result
    
    def place_order_with_leverage(self, side, posSide, usd_amount, leverage, reduce_only=False, max_retry=3, sleep_sec=1):
        """
        带杠杆下单（兼容OKX接口）
        
        Args:
            side: 买卖方向
            posSide: 持仓方向
            usd_amount: 本金金额
            leverage: 杠杆倍数
            reduce_only: 是否只减仓
            max_retry: 最大重试次数
            sleep_sec: 重试间隔
        """
        for attempt in range(max_retry):
            try:
                market_price = self.get_price()
                
                # ✅ 资金安全校验
                account_info = self.get_account_balance()
                available_usdt = float(account_info['data'][0]['availEq'])
                required_margin = usd_amount
                
                if required_margin > available_usdt:
                    log_error(f"❌ 保证金不足: 需 {required_margin} USDT，可用 {available_usdt} USDT，取消下单")
                    return False
                
                # ✅ 计算下单数量
                lot_size = config.LOT_SIZE
                order_value = usd_amount * leverage
                raw_size = order_value / market_price
                size = (raw_size // lot_size) * lot_size
                size = round(size, 6)
                
                if size < lot_size:
                    if reduce_only:
                        log_info(f"🟡 平仓 size={size} 小于最小下单单位 {lot_size}，自动跳过")
                        return False
                    else:
                        raise Exception(f"⚠ 下单失败: 开仓 size={size} 小于最小下单单位 {lot_size}")
                
                # ✅ 下单
                result = self.place_order(
                    side=side,
                    pos_side=posSide,
                    size=size,
                    order_type='market',
                    reduce_only=reduce_only
                )
                
                if result.get('code') == '0' or result.get('code') == 0:
                    order_id = result.get('data', {}).get('orderId', 'N/A')
                    log_info(f"✅ 下单成功: {side} {posSide} 杠杆: {leverage}x, 本金: {usd_amount} USD, 下单数量: {size} {config.SYMBOL}, 订单ID: {order_id}")
                    return True
                else:
                    error_code = result.get('code', '')
                    error_msg = result.get('msg', '')
                    log_error(f"❌ 下单失败: 错误码 {error_code}, 原因: {error_msg}")
                    time.sleep(sleep_sec)
            except Exception as e:
                log_error(f"⚠ 下单异常({attempt + 1}): {e}")
                time.sleep(sleep_sec)
        
        raise Exception("❌ 超过最大重试次数，下单失败")
    
    # 兼容OKX接口的方法
    def open_long(self, usd_amount, leverage):
        self.place_order_with_leverage("buy", "long", usd_amount, leverage, reduce_only=False)
    
    def close_long(self, usd_amount, leverage):
        long_pos, _ = self.get_position()
        if long_pos['size'] == 0:
            log_info("🟢 无多仓位，跳过平多")
            return
        self.place_order_with_leverage("sell", "long", usd_amount, leverage, reduce_only=True)
    
    def open_short(self, usd_amount, leverage):
        self.place_order_with_leverage("sell", "short", usd_amount, leverage, reduce_only=False)
    
    def close_short(self, usd_amount, leverage):
        _, short_pos = self.get_position()
        if short_pos['size'] == 0:
            log_info("🟢 无空仓位，跳过平空")
            return
        self.place_order_with_leverage("buy", "short", usd_amount, leverage, reduce_only=True)
    
    def fetch_data(self):
        """获取多周期数据（兼容OKX接口）"""
        data_dict = {}
        for interval in config.INTERVALS:
            df = self.fetch_ohlcv(config.SYMBOL, bar=interval, max_limit=config.WINDOWS.get(interval, 2000))
            df.set_index("timestamp", inplace=True)
            data_dict[interval] = df
            time.sleep(0.3)
        return data_dict

