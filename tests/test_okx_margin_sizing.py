import unittest
import sys
import types

try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pandas"] = types.ModuleType("pandas")

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    fake_requests = types.ModuleType("requests")
    fake_requests.post = lambda *args, **kwargs: None
    sys.modules["requests"] = fake_requests

try:
    import okx.Account  # noqa: F401
except ModuleNotFoundError:
    fake_okx = types.ModuleType("okx")
    fake_okx.__path__ = []
    sys.modules["okx"] = fake_okx
    for name in ("Account", "Trade", "MarketData", "PublicData", "TradingData"):
        module = types.ModuleType(f"okx.{name}")
        sys.modules[f"okx.{name}"] = module

from core.okx_api import cap_size_by_available_margin, floor_size_to_lot, is_insufficient_margin_error


class OKXMarginSizingTests(unittest.TestCase):
    def test_floor_size_to_lot_rounds_down(self):
        self.assertAlmostEqual(floor_size_to_lot(1.239, 0.01), 1.23)

    def test_floor_size_to_lot_preserves_exact_lot_multiple(self):
        self.assertEqual(floor_size_to_lot(30.83, 0.01), 30.83)

    def test_cap_size_by_available_margin_reduces_oversized_open(self):
        capped_size, required_margin, usable_margin, was_capped = cap_size_by_available_margin(
            size=100.0,
            market_price=100.0,
            leverage=5,
            available_usdt=1000.0,
            lot_size=0.01,
            usage_ratio=0.8,
            min_free_margin_usdt=100.0,
        )

        self.assertTrue(was_capped)
        self.assertAlmostEqual(usable_margin, 720.0)
        self.assertAlmostEqual(capped_size, 36.0)
        self.assertAlmostEqual(required_margin, 720.0)

    def test_cap_size_by_available_margin_keeps_affordable_open(self):
        capped_size, required_margin, usable_margin, was_capped = cap_size_by_available_margin(
            size=10.0,
            market_price=100.0,
            leverage=5,
            available_usdt=1000.0,
            lot_size=0.01,
            usage_ratio=0.8,
            min_free_margin_usdt=100.0,
        )

        self.assertFalse(was_capped)
        self.assertAlmostEqual(capped_size, 10.0)
        self.assertAlmostEqual(required_margin, 200.0)
        self.assertAlmostEqual(usable_margin, 720.0)

    def test_cap_size_returns_zero_when_margin_cannot_cover_one_lot(self):
        capped_size, required_margin, usable_margin, was_capped = cap_size_by_available_margin(
            size=1.0,
            market_price=100.0,
            leverage=5,
            available_usdt=20.0,
            lot_size=0.01,
            usage_ratio=0.8,
            min_free_margin_usdt=30.0,
        )

        self.assertTrue(was_capped)
        self.assertEqual(capped_size, 0.0)
        self.assertEqual(required_margin, 0.0)
        self.assertEqual(usable_margin, 0.0)

    def test_insufficient_margin_error_detection(self):
        result = {
            "code": "1",
            "data": [
                {
                    "sCode": "51008",
                    "sMsg": "Order failed. Insufficient USDT margin in account",
                }
            ],
        }

        self.assertTrue(is_insufficient_margin_error(result))


if __name__ == "__main__":
    unittest.main()
