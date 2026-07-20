import base64
import hashlib
import hmac
import unittest
from unittest.mock import patch

from core.okx_ws import OKXRealtimeStream, build_login_payload, okx_websocket_urls


class OKXWebSocketTests(unittest.TestCase):
    def make_stream(self):
        return OKXRealtimeStream(
            symbol="SOL-USDT-SWAP",
            api_key="key",
            secret_key="secret",
            passphrase="passphrase",
            simulated=True,
        )

    def test_demo_and_production_urls(self):
        self.assertEqual(
            okx_websocket_urls(True),
            (
                "wss://wspap.okx.com:8443/ws/v5/public",
                "wss://wspap.okx.com:8443/ws/v5/private",
            ),
        )
        self.assertEqual(
            okx_websocket_urls(False),
            (
                "wss://ws.okx.com:8443/ws/v5/public",
                "wss://ws.okx.com:8443/ws/v5/private",
            ),
        )

    def test_login_payload_uses_okx_signature_format(self):
        payload = build_login_payload("key", "secret", "passphrase", timestamp="123")
        expected = base64.b64encode(
            hmac.new(b"secret", b"123GET/users/self/verify", hashlib.sha256).digest()
        ).decode()

        self.assertEqual(payload["op"], "login")
        self.assertEqual(payload["args"][0]["timestamp"], "123")
        self.assertEqual(payload["args"][0]["sign"], expected)

    def test_ticker_message_updates_fresh_price(self):
        stream = self.make_stream()
        with patch("core.okx_ws.time.monotonic", return_value=100.0):
            updated = stream._handle_ticker_message({
                "arg": {"channel": "tickers"},
                "data": [{"instId": "SOL-USDT-SWAP", "last": "76.85", "ts": "1234"}],
            })

        with patch("core.okx_ws.time.monotonic", return_value=102.0):
            self.assertTrue(updated)
            self.assertEqual(stream.get_price(3.0), 76.85)
        with patch("core.okx_ws.time.monotonic", return_value=104.0):
            self.assertIsNone(stream.get_price(3.0))

    def test_position_message_tracks_both_sides_and_flat_snapshot(self):
        stream = self.make_stream()
        with patch("core.okx_ws.time.monotonic", return_value=200.0):
            stream._handle_position_message({
                "arg": {"channel": "positions"},
                "data": [
                    {"instId": "SOL-USDT-SWAP", "posSide": "long", "pos": "2.5", "avgPx": "75.2"},
                    {"instId": "SOL-USDT-SWAP", "posSide": "short", "pos": "0", "avgPx": ""},
                ],
            })
        with patch("core.okx_ws.time.monotonic", return_value=201.0):
            long_pos, short_pos = stream.get_position(3.0)
            self.assertEqual(long_pos, {"size": 2.5, "entry_price": 75.2})
            self.assertEqual(short_pos, {"size": 0.0, "entry_price": 0.0})

        with patch("core.okx_ws.time.monotonic", return_value=202.0):
            stream._handle_position_message({"arg": {"channel": "positions"}, "data": []})
            long_pos, short_pos = stream.get_position(3.0)
            self.assertEqual(long_pos["size"], 0.0)
            self.assertEqual(short_pos["size"], 0.0)

    def test_position_remains_valid_while_event_stream_is_connected(self):
        stream = self.make_stream()
        with patch("core.okx_ws.time.monotonic", return_value=10.0):
            stream._handle_position_message({"arg": {"channel": "positions"}, "data": []})
        with patch("core.okx_ws.time.monotonic", return_value=16.0):
            self.assertIsNotNone(stream.get_position(5.0))

    def test_disconnected_position_stream_returns_none(self):
        stream = self.make_stream()
        with patch("core.okx_ws.time.monotonic", return_value=10.0):
            stream._handle_position_message({"arg": {"channel": "positions"}, "data": []})
            stream._set_connection_state("position", False)
        with patch("core.okx_ws.time.monotonic", return_value=11.0):
            self.assertIsNone(stream.get_position(5.0))

        stream._set_connection_state("position", True)
        with patch("core.okx_ws.time.monotonic", return_value=12.0):
            self.assertIsNone(stream.get_position(5.0))


if __name__ == "__main__":
    unittest.main()
