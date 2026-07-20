import asyncio
import base64
import hashlib
import hmac
import json
import threading
import time

import aiohttp

from utils.utils import log_error, log_info


def okx_websocket_urls(simulated):
    host = "wspap.okx.com" if bool(simulated) else "ws.okx.com"
    return (
        f"wss://{host}:8443/ws/v5/public",
        f"wss://{host}:8443/ws/v5/private",
    )


def build_login_payload(api_key, secret_key, passphrase, timestamp=None):
    timestamp = str(int(time.time()) if timestamp is None else timestamp)
    prehash = f"{timestamp}GET/users/self/verify"
    digest = hmac.new(
        str(secret_key).encode(),
        prehash.encode(),
        hashlib.sha256,
    ).digest()
    signature = base64.b64encode(digest).decode()
    return {
        "op": "login",
        "args": [{
            "apiKey": api_key,
            "passphrase": passphrase,
            "timestamp": timestamp,
            "sign": signature,
        }],
    }


class OKXRealtimeStream:
    def __init__(
        self,
        *,
        symbol,
        api_key,
        secret_key,
        passphrase,
        simulated,
        reconnect_max_sec=30.0,
    ):
        self.symbol = str(symbol)
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.public_url, self.private_url = okx_websocket_urls(simulated)
        self.reconnect_max_sec = max(1.0, float(reconnect_max_sec))

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ticker_ready = threading.Event()
        self._position_ready = threading.Event()
        self._thread = None

        self._ticker_connected = False
        self._position_connected = False
        self._last_price = None
        self._last_price_exchange_ts = None
        self._last_price_received_at = None
        self._long_position = {"size": 0.0, "entry_price": 0.0}
        self._short_position = {"size": 0.0, "entry_price": 0.0}
        self._last_position_received_at = None
        self._last_error = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._ticker_ready.clear()
        self._position_ready.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="okx-realtime-ws",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout=5.0):
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))

    def wait_until_ready(self, timeout=10.0):
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            if self._ticker_ready.is_set() and self._position_ready.is_set():
                return True
            time.sleep(0.05)
        return self._ticker_ready.is_set() and self._position_ready.is_set()

    def get_price(self, max_age_sec):
        now = time.monotonic()
        with self._lock:
            if self._last_price_received_at is None:
                return None
            if now - self._last_price_received_at > float(max_age_sec):
                return None
            return self._last_price

    def get_position(self, max_age_sec):
        with self._lock:
            if not self._position_connected or self._last_position_received_at is None:
                return None
            return dict(self._long_position), dict(self._short_position)

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            ticker_age_ms = None
            if self._last_price_received_at is not None:
                ticker_age_ms = max(0.0, now - self._last_price_received_at) * 1000.0
            position_age_ms = None
            if self._last_position_received_at is not None:
                position_age_ms = max(0.0, now - self._last_position_received_at) * 1000.0
            return {
                "ticker_connected": bool(self._ticker_connected),
                "position_connected": bool(self._position_connected),
                "ticker_age_ms": ticker_age_ms,
                "position_age_ms": position_age_ms,
                "last_price": self._last_price,
                "last_price_exchange_ts": self._last_price_exchange_ts,
                "last_error": self._last_error,
            }

    def _set_connection_state(self, channel, connected, error=None):
        with self._lock:
            if channel == "ticker":
                self._ticker_connected = bool(connected)
                if not connected:
                    self._last_price_received_at = None
            else:
                self._position_connected = bool(connected)
                if not connected:
                    self._last_position_received_at = None
            if error:
                self._last_error = str(error)

    def _handle_ticker_message(self, message):
        if (message.get("arg") or {}).get("channel") != "tickers":
            return False
        updated = False
        for row in message.get("data") or []:
            if str(row.get("instId") or "") != self.symbol:
                continue
            try:
                price = float(row.get("last"))
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue
            with self._lock:
                self._last_price = price
                self._last_price_exchange_ts = row.get("ts")
                self._last_price_received_at = time.monotonic()
                self._ticker_connected = True
                self._last_error = None
            self._ticker_ready.set()
            updated = True
        return updated

    def _handle_position_message(self, message):
        if (message.get("arg") or {}).get("channel") != "positions":
            return False
        rows = message.get("data") or []
        with self._lock:
            if not rows:
                self._long_position = {"size": 0.0, "entry_price": 0.0}
                self._short_position = {"size": 0.0, "entry_price": 0.0}
            for row in rows:
                if str(row.get("instId") or "") != self.symbol:
                    continue
                try:
                    size = abs(float(row.get("pos") or 0.0))
                    entry_price = float(row.get("avgPx") or 0.0)
                except (TypeError, ValueError):
                    continue
                pos_side = str(row.get("posSide") or "").lower()
                if pos_side == "long":
                    self._long_position = {"size": size, "entry_price": entry_price}
                elif pos_side == "short":
                    self._short_position = {"size": size, "entry_price": entry_price}
            self._last_position_received_at = time.monotonic()
            self._position_connected = True
            self._last_error = None
        self._position_ready.set()
        return True

    def _thread_main(self):
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self._set_connection_state("ticker", False, exc)
            self._set_connection_state("position", False, exc)
            log_error(f"OKX WebSocket worker stopped: {exc}")

    async def _run(self):
        timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await asyncio.gather(
                self._run_ticker_loop(session),
                self._run_position_loop(session),
            )

    async def _reconnect_sleep(self, delay):
        deadline = time.monotonic() + delay
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _decode_message(message):
        if message.type == aiohttp.WSMsgType.TEXT:
            return json.loads(message.data)
        if message.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.ERROR,
        }:
            raise ConnectionError(f"websocket closed: type={message.type}")
        return None

    async def _receive_message(self, ws):
        try:
            raw_message = await asyncio.wait_for(ws.receive(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
        return self._decode_message(raw_message)

    async def _run_ticker_loop(self, session):
        delay = 1.0
        while not self._stop_event.is_set():
            try:
                async with session.ws_connect(self.public_url, heartbeat=20) as ws:
                    await ws.send_json({
                        "op": "subscribe",
                        "args": [{"channel": "tickers", "instId": self.symbol}],
                    })
                    self._set_connection_state("ticker", True)
                    log_info("OKX ticker WebSocket connected")
                    delay = 1.0
                    while not self._stop_event.is_set():
                        message = await self._receive_message(ws)
                        if not message:
                            continue
                        if message.get("event") == "error":
                            raise RuntimeError(
                                f"ticker subscribe failed: code={message.get('code')} msg={message.get('msg')}"
                            )
                        self._handle_ticker_message(message)
            except Exception as exc:
                self._set_connection_state("ticker", False, exc)
                if not self._stop_event.is_set():
                    log_error(f"OKX ticker WebSocket reconnecting: {exc}")
                    await self._reconnect_sleep(delay)
                    delay = min(self.reconnect_max_sec, delay * 2.0)
        self._set_connection_state("ticker", False)

    async def _run_position_loop(self, session):
        delay = 1.0
        while not self._stop_event.is_set():
            try:
                async with session.ws_connect(self.private_url, heartbeat=20) as ws:
                    await ws.send_json(build_login_payload(
                        self.api_key,
                        self.secret_key,
                        self.passphrase,
                    ))
                    subscribed = False
                    while not self._stop_event.is_set():
                        message = await self._receive_message(ws)
                        if not message:
                            continue
                        event = message.get("event")
                        if event == "error":
                            raise RuntimeError(
                                f"positions subscribe failed: code={message.get('code')} msg={message.get('msg')}"
                            )
                        if event == "login":
                            if str(message.get("code") or "") != "0":
                                raise RuntimeError(
                                    f"positions login failed: code={message.get('code')} msg={message.get('msg')}"
                                )
                            await ws.send_json({
                                "op": "subscribe",
                                "args": [{
                                    "channel": "positions",
                                    "instType": "SWAP",
                                    "instId": self.symbol,
                                    "extraParams": json.dumps({"updateInterval": "2000"}),
                                }],
                            })
                            continue
                        if event == "subscribe":
                            subscribed = True
                            self._set_connection_state("position", True)
                            log_info("OKX positions WebSocket connected")
                            delay = 1.0
                            continue
                        if subscribed:
                            self._handle_position_message(message)
            except Exception as exc:
                self._set_connection_state("position", False, exc)
                if not self._stop_event.is_set():
                    log_error(f"OKX positions WebSocket reconnecting: {exc}")
                    await self._reconnect_sleep(delay)
                    delay = min(self.reconnect_max_sec, delay * 2.0)
        self._set_connection_state("position", False)
