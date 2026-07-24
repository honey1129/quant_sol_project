import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.okx_api import OKXClient, OKXResponseError


class OKXReadRetryTests(unittest.TestCase):
    def setUp(self):
        self.client = OKXClient.__new__(OKXClient)

    def test_custom_request_timeout_applies_to_all_api_clients(self):
        client = OKXClient(request_timeout_sec=1.25)
        try:
            for name in (
                "account_api",
                "trade_api",
                "market_api",
                "public_api",
                "trading_data_api",
            ):
                api = getattr(client, name)
                self.assertEqual(api.timeout.connect, 1.25)
                self.assertEqual(api.timeout.read, 1.25)
                self.assertEqual(api.timeout.write, 1.25)
                self.assertEqual(api.timeout.pool, 1.25)
        finally:
            for name in (
                "account_api",
                "trade_api",
                "market_api",
                "public_api",
                "trading_data_api",
            ):
                close = getattr(getattr(client, name), "close", None)
                if callable(close):
                    close()

    def test_retries_business_error_response_until_success(self):
        responses = iter([
            {
                "code": "50001",
                "data": [],
                "msg": "Service temporarily unavailable. Please try again later.",
            },
            {"code": "0", "data": [{"posSide": "long", "pos": "2", "avgPx": "78.3"}]},
        ])
        self.client.account_api = SimpleNamespace(
            get_positions=lambda **_kwargs: next(responses)
        )

        with patch("core.okx_api.config.OKX_API_MAX_RETRY", 3):
            with patch("core.okx_api.config.OKX_API_RETRY_SLEEP_SEC", 0):
                long_pos, short_pos = self.client.get_position()

        self.assertEqual(long_pos, {"size": 2.0, "entry_price": 78.3})
        self.assertEqual(short_pos, {"size": 0.0, "entry_price": 0.0})

    def test_position_read_accepts_one_shot_retry_override(self):
        attempts = 0

        def unavailable(**_kwargs):
            nonlocal attempts
            attempts += 1
            return {"code": "50001", "data": [], "msg": "temporarily unavailable"}

        self.client.account_api = SimpleNamespace(get_positions=unavailable)

        with self.assertRaises(OKXResponseError):
            self.client.get_position(max_retry=1, sleep_sec=0)

        self.assertEqual(attempts, 1)

    def test_retries_response_without_okx_code(self):
        responses = iter([
            {
                "message": "failure to get a peer from the ring-balancer",
                "error_id": "temporary-error",
            },
            {"code": "0", "data": []},
        ])

        with patch("core.okx_api.config.OKX_API_MAX_RETRY", 2):
            with patch("core.okx_api.config.OKX_API_RETRY_SLEEP_SEC", 0):
                result = self.client._call_read_with_retry(
                    "获取仓位",
                    lambda: next(responses),
                )

        self.assertEqual(result, {"code": "0", "data": []})

    def test_empty_data_is_valid_for_position_and_order_lists(self):
        result = self.client._call_read_with_retry(
            "获取挂单列表",
            lambda: {"code": "0", "data": []},
            max_retry=1,
            sleep_sec=0,
        )

        self.assertEqual(result["data"], [])

    def test_required_data_retries_empty_payload(self):
        responses = iter([
            {"code": "0", "data": []},
            {"code": "0", "data": [{"totalEq": "1000"}]},
        ])

        result = self.client._call_read_with_retry(
            "获取账户余额",
            lambda: next(responses),
            require_data=True,
            max_retry=2,
            sleep_sec=0,
        )

        self.assertEqual(result["data"][0]["totalEq"], "1000")

    def test_raises_after_all_business_error_responses_fail(self):
        attempts = 0

        def unavailable():
            nonlocal attempts
            attempts += 1
            return {"code": "50001", "data": [], "msg": "temporarily unavailable"}

        with self.assertRaises(OKXResponseError):
            self.client._call_read_with_retry(
                "获取仓位",
                unavailable,
                max_retry=3,
                sleep_sec=0,
            )

        self.assertEqual(attempts, 3)

    def test_get_price_retries_invalid_last_field(self):
        responses = iter([
            {"code": "0", "data": [{"last": ""}]},
            {"code": "0", "data": [{"last": "78.32"}]},
        ])
        self.client.market_api = SimpleNamespace(
            get_ticker=lambda **_kwargs: next(responses)
        )

        price = self.client.get_price(max_retry=2, sleep_sec=0)

        self.assertEqual(price, 78.32)


if __name__ == "__main__":
    unittest.main()
