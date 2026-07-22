import math
import time
import uuid
from decimal import Decimal, ROUND_FLOOR

import pandas as pd
from config import config
import okx.Account as Account
import okx.Trade as Trade
import okx.MarketData as Market
import okx.PublicData as Public
import okx.TradingData as TradingData
from utils.utils import log_info, log_error


def build_client_order_id(symbol, side, pos_side, reduce_only):
    symbol_part = "".join(ch for ch in str(symbol).lower() if ch.isalnum())[:8]
    side_part = str(side).lower()[:1] or "x"
    pos_part = str(pos_side).lower()[:1] or "x"
    reduce_part = "r" if reduce_only else "o"
    unique_part = uuid.uuid4().hex[:16]
    return f"qs{symbol_part}{reduce_part}{side_part}{pos_part}{unique_part}"[:32]


def order_is_acknowledged(order):
    if not order:
        return False
    state = str(order.get("state", "") or "").lower()
    return state in {"live", "partially_filled", "filled"}


def order_is_filled(order):
    if not order:
        return False
    state = str(order.get("state", "") or "").lower()
    return state == "filled"


def order_has_fill(order):
    if not order:
        return False
    if str(order.get("state", "") or "").lower() == "partially_filled":
        return True
    for key in ("accFillSz", "fillSz"):
        try:
            if float(order.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def order_is_terminal(order):
    if not order:
        return False
    state = str(order.get("state", "") or "").lower()
    return state in {"filled", "canceled", "rejected", "failed", "mmp_canceled"}


class OrderStateUnknownError(RuntimeError):
    """Raised when an accepted order cannot be reconciled to a terminal state."""


class OKXResponseError(RuntimeError):
    """Raised when an OKX read endpoint returns an unsuccessful or malformed payload."""


def floor_size_to_lot(size, lot_size):
    size_decimal = Decimal(str(size))
    lot_decimal = Decimal(str(lot_size))
    if lot_decimal <= 0:
        raise ValueError("lot_size must be positive")
    steps = (size_decimal / lot_decimal).to_integral_value(rounding=ROUND_FLOOR)
    return float(steps * lot_decimal)


def cap_size_by_available_margin(
    size,
    market_price,
    leverage,
    available_usdt,
    lot_size,
    *,
    usage_ratio=0.85,
    min_free_margin_usdt=0.0,
):
    size = floor_size_to_lot(size, lot_size)
    market_price = float(market_price)
    leverage = float(leverage)
    available_usdt = float(available_usdt)
    lot_size = float(lot_size)
    usage_ratio = max(0.0, min(float(usage_ratio), 1.0))
    min_free_margin_usdt = max(0.0, float(min_free_margin_usdt))

    if size <= 0 or market_price <= 0 or leverage <= 0:
        return 0.0, 0.0, 0.0, False

    usable_margin = max(0.0, available_usdt - min_free_margin_usdt) * usage_ratio
    required_margin = size * market_price / leverage
    if required_margin <= usable_margin:
        return size, required_margin, usable_margin, False

    capped_size = floor_size_to_lot(usable_margin * leverage / market_price, lot_size)
    capped_required_margin = capped_size * market_price / leverage if capped_size > 0 else 0.0
    return capped_size, capped_required_margin, usable_margin, True


def is_insufficient_margin_error(result):
    if not isinstance(result, dict):
        return False
    for item in result.get("data", []) or []:
        code = str(item.get("sCode", "") or "")
        message = str(item.get("sMsg", "") or "").lower()
        if code == "51008" or "insufficient" in message and "margin" in message:
            return True
    return False


class OKXClient:
    def __init__(self):
        self.account_api = Account.AccountAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.trade_api = Trade.TradeAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.market_api = Market.MarketAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True, flag=config.USE_SERVER)
        self.public_api = Public.PublicAPI(config.OKX_API_KEY, config.OKX_SECRET, config.OKX_PASSWORD, use_server_time=True,flag=config.USE_SERVER)
        # Rubik 交易大数据(OI / taker / 多空比)。仅做只读统计,无需签名,但沿用同一 flag。
        self.trading_data_api = TradingData.TradingDataAPI(flag=config.USE_SERVER, debug=False)

    def _call_with_retry(self, label, func, *, max_retry=None, sleep_sec=None, backoff=None):
        max_retry = max(1, int(max_retry if max_retry is not None else config.OKX_API_MAX_RETRY))
        sleep_sec = max(0.0, float(sleep_sec if sleep_sec is not None else config.OKX_API_RETRY_SLEEP_SEC))
        backoff = max(1.0, float(backoff if backoff is not None else config.OKX_API_RETRY_BACKOFF))
        delay = sleep_sec
        last_error = None

        for attempt in range(max_retry):
            try:
                return func()
            except Exception as exc:
                last_error = exc
                log_error(f"⚠ {label}失败，第{attempt + 1}/{max_retry}次: {exc}")
                if attempt < max_retry - 1 and delay > 0:
                    time.sleep(delay)
                    delay *= backoff

        raise last_error

    @staticmethod
    def _validate_read_response(label, result, *, require_data=False):
        if not isinstance(result, dict):
            raise OKXResponseError(
                f"{label}响应格式异常: expected=dict, actual={type(result).__name__}"
            )

        code = str(result.get("code", "") or "")
        if code != "0":
            message = result.get("msg") or result.get("message") or "unknown error"
            error_id = result.get("error_id")
            error_suffix = f", error_id={error_id}" if error_id else ""
            raise OKXResponseError(
                f"{label}响应失败: code={code or 'missing'}, msg={message}{error_suffix}"
            )

        if "data" not in result or result.get("data") is None:
            raise OKXResponseError(f"{label}响应缺少 data")
        if not isinstance(result["data"], list):
            raise OKXResponseError(
                f"{label}响应 data 格式异常: actual={type(result['data']).__name__}"
            )
        if require_data and not result["data"]:
            raise OKXResponseError(f"{label}响应 data 为空")
        return result

    def _call_read_with_retry(
        self,
        label,
        func,
        *,
        require_data=False,
        transform=None,
        max_retry=None,
        sleep_sec=None,
        backoff=None,
    ):
        def checked_call():
            result = self._validate_read_response(
                label,
                func(),
                require_data=require_data,
            )
            return transform(result) if transform is not None else result

        return self._call_with_retry(
            label,
            checked_call,
            max_retry=max_retry,
            sleep_sec=sleep_sec,
            backoff=backoff,
        )

    def get_account_config(self):
        result = self._call_read_with_retry(
            "获取账户配置",
            self.account_api.get_account_config,
            require_data=True,
        )
        if result.get("code") != "0":
            raise RuntimeError(f"获取账户配置失败: {result}")
        data = result.get("data", [])
        if not data:
            raise RuntimeError("账户配置为空，无法继续")
        return data[0]

    def list_pending_orders(self):
        result = self._call_read_with_retry(
            "获取挂单列表",
            lambda: self.trade_api.get_order_list(
                instType="SWAP",
                instId=config.SYMBOL,
            ),
        )
        if result.get("code") != "0":
            raise RuntimeError(f"获取挂单列表失败: {result}")
        return result.get("data", [])

    def get_order_by_client_id(self, cl_ord_id):
        if not cl_ord_id:
            return None
        try:
            result = self._call_read_with_retry(
                "按 clOrdId 查询订单",
                lambda: self.trade_api.get_order(instId=config.SYMBOL, clOrdId=cl_ord_id),
            )
        except Exception as exc:
            log_error(f"按 clOrdId 查询订单失败: {cl_ord_id} -> {exc}")
            return None

        if result.get("code") != "0":
            return None
        data = result.get("data", [])
        return data[0] if data else None

    def fetch_order_fills(self, ord_id):
        if not ord_id:
            return []
        try:
            result = self._call_read_with_retry(
                "查询订单成交明细",
                lambda: self.trade_api.get_fills(
                    instType="SWAP",
                    instId=config.SYMBOL,
                    ordId=ord_id,
                ),
            )
        except Exception as exc:
            log_error(f"查询订单成交明细失败: ordId={ord_id} -> {exc}")
            return []

        if result.get("code") != "0":
            log_error(f"查询订单成交明细失败: ordId={ord_id}, result={result}")
            return []
        return result.get("data", []) or []

    def _attach_order_fills(self, order):
        if not order:
            return order
        if "_fills" in order:
            return order
        if not hasattr(self, "trade_api"):
            enriched = dict(order)
            enriched["_fills"] = []
            return enriched
        try:
            fills = self.fetch_order_fills(order.get("ordId"))
        except Exception:
            fills = []
        enriched = dict(order)
        enriched["_fills"] = fills
        return enriched

    def _terminal_order_result(self, order):
        if order_is_filled(order):
            return self._attach_order_fills(order)
        if order_has_fill(order):
            enriched = self._attach_order_fills(order)
            enriched["_partial_fill"] = True
            return enriched
        return False

    def wait_until_filled(
        self,
        cl_ord_id,
        timeout_sec=5.0,
        poll_interval_sec=0.3,
        cancel_confirm_timeout_sec=5.0,
    ):
        if not cl_ord_id:
            return None
        deadline = time.monotonic() + float(timeout_sec)
        terminal_states = {"canceled", "rejected", "failed", "mmp_canceled"}
        last_order_with_fill = None

        def terminal_result(order):
            nonlocal last_order_with_fill
            if order_is_filled(order):
                return self._attach_order_fills(order)
            if order_has_fill(order):
                enriched = self._attach_order_fills(order)
                enriched["_partial_fill"] = True
                return enriched
            if last_order_with_fill is not None:
                merged = dict(last_order_with_fill)
                merged.update(order or {})
                enriched = self._attach_order_fills(merged)
                enriched["_partial_fill"] = True
                return enriched
            return None

        while True:
            order = self.get_order_by_client_id(cl_ord_id)
            if order is not None:
                if order_has_fill(order):
                    last_order_with_fill = order
                if order_is_filled(order):
                    return self._attach_order_fills(order)
                state = str(order.get("state", "") or "").lower()
                if state in terminal_states:
                    return terminal_result(order)
            if time.monotonic() >= deadline:
                try:
                    pending = self.get_order_by_client_id(cl_ord_id)
                    if pending is None:
                        raise OrderStateUnknownError(
                            f"订单超时后无法查询状态: clOrdId={cl_ord_id}"
                        )
                    pstate = str(pending.get("state", "") or "").lower()
                    if order_has_fill(pending):
                        last_order_with_fill = pending
                    if pstate == "filled" or pstate in terminal_states:
                        return terminal_result(pending)

                    cancel_result = self.trade_api.cancel_order(
                        instId=config.SYMBOL,
                        clOrdId=cl_ord_id,
                    )
                    cancel_data = cancel_result.get("data", []) or []
                    cancel_item_code = str(cancel_data[0].get("sCode", "") or "") if cancel_data else ""
                    cancel_accepted = (
                        str(cancel_result.get("code", "")) == "0"
                        and cancel_item_code in {"", "0"}
                    )
                    if not cancel_accepted:
                        raise OrderStateUnknownError(
                            f"订单超时且撤单请求失败: clOrdId={cl_ord_id}, result={cancel_result}"
                        )

                    log_error(f"⚠ 下单超时（{timeout_sec}s），撤单请求已受理: clOrdId={cl_ord_id}")
                    cancel_deadline = time.monotonic() + float(cancel_confirm_timeout_sec)
                    while time.monotonic() < cancel_deadline:
                        final_order = self.get_order_by_client_id(cl_ord_id)
                        if final_order is not None:
                            if order_has_fill(final_order):
                                last_order_with_fill = final_order
                            final_state = str(final_order.get("state", "") or "").lower()
                            if final_state == "filled" or final_state in terminal_states:
                                return terminal_result(final_order)
                        time.sleep(poll_interval_sec)
                    raise OrderStateUnknownError(
                        f"撤单请求已受理但未确认终态: clOrdId={cl_ord_id}"
                    )
                except OrderStateUnknownError:
                    raise
                except Exception as exc:
                    raise OrderStateUnknownError(
                        f"超时撤单或终态确认异常: clOrdId={cl_ord_id}, err={exc}"
                    ) from exc
            time.sleep(poll_interval_sec)

    def cancel_pending_orders(self, confirm_timeout_sec=5.0, poll_interval_sec=0.3):
        pending_orders = self.list_pending_orders()
        if not pending_orders:
            return []

        canceled = []
        for order in pending_orders:
            ord_id = order.get("ordId", "")
            cl_ord_id = order.get("clOrdId", "")
            try:
                result = self.trade_api.cancel_order(
                    instId=config.SYMBOL,
                    ordId=ord_id,
                    clOrdId=cl_ord_id,
                )
                data = result.get("data", []) or []
                item_code = str(data[0].get("sCode", "") or "") if data else ""
                if str(result.get("code", "")) == "0" and item_code in {"", "0"}:
                    canceled.append({
                        "ordId": ord_id,
                        "clOrdId": cl_ord_id,
                        "state": order.get("state", ""),
                    })
                else:
                    raise OrderStateUnknownError(
                        f"撤销挂单请求失败: ordId={ord_id}, clOrdId={cl_ord_id}, result={result}"
                    )
            except OrderStateUnknownError:
                raise
            except Exception as exc:
                raise OrderStateUnknownError(
                    f"撤销挂单异常: ordId={ord_id}, clOrdId={cl_ord_id}, err={exc}"
                ) from exc

        if canceled:
            canceled_ids = {str(item.get("ordId", "")) for item in canceled}
            deadline = time.monotonic() + float(confirm_timeout_sec)
            while time.monotonic() < deadline:
                remaining = self.list_pending_orders()
                if not any(str(item.get("ordId", "")) in canceled_ids for item in remaining):
                    log_info(f"已确认清理挂单 {len(canceled)} 笔")
                    return canceled
                time.sleep(poll_interval_sec)
            raise OrderStateUnknownError(
                f"挂单撤销请求已受理但未确认终态: ordIds={sorted(canceled_ids)}"
            )
        return canceled

    def _extract_leverage_by_side(self, leverage_rows):
        leverage_map = {}
        for row in leverage_rows:
            pos_side = str(row.get("posSide", "") or "").lower() or "both"
            try:
                leverage_map[pos_side] = float(row.get("lever", 0) or 0)
            except (TypeError, ValueError):
                continue
        return leverage_map

    def ensure_trading_ready(self):
        if bool(config.LIVE_REQUIRE_SIMULATED_TRADING) and str(config.USE_SERVER) != "1":
            raise RuntimeError("当前配置不是 OKX 模拟盘(USE_SERVER=1)，已阻止启动交易监控")

        if not config.OKX_API_KEY or not config.OKX_SECRET or not config.OKX_PASSWORD:
            raise RuntimeError("OKX API 凭证缺失，无法启动交易监控")

        account_config = self.get_account_config()
        pos_mode = str(account_config.get("posMode", "") or "").lower()

        if pos_mode != "long_short_mode":
            if not bool(config.LIVE_AUTO_SET_POSITION_MODE):
                raise RuntimeError(f"账户持仓模式为 {pos_mode or 'unknown'}，与策略要求的 long_short_mode 不一致")

            long_pos, short_pos = self.get_position()
            if long_pos["size"] > 0 or short_pos["size"] > 0:
                raise RuntimeError("持仓模式不是 long_short_mode，且当前已有持仓，拒绝自动切换")

            result = self._call_with_retry(
                "自动设置持仓模式",
                lambda: self.account_api.set_position_mode("long_short_mode"),
            )
            if result.get("code") != "0":
                raise RuntimeError(f"自动设置持仓模式失败: {result}")
            log_info("已自动切换到账户双向持仓模式 long_short_mode")
            time.sleep(0.5)
            account_config = self.get_account_config()
            pos_mode = str(account_config.get("posMode", "") or "").lower()
            if pos_mode != "long_short_mode":
                raise RuntimeError(f"持仓模式校验失败，当前为 {pos_mode or 'unknown'}")

        if bool(config.LIVE_RECONCILE_PENDING_ORDERS):
            self.cancel_pending_orders()

        leverage_rows = self._call_read_with_retry(
            "获取杠杆配置",
            lambda: self.account_api.get_leverage(mgnMode="cross", instId=config.SYMBOL),
            require_data=True,
        )
        if leverage_rows.get("code") != "0":
            raise RuntimeError(f"获取杠杆配置失败: {leverage_rows}")
        leverage_map = self._extract_leverage_by_side(leverage_rows.get("data", []))
        target_leverage = float(config.LEVERAGE)

        if bool(config.LIVE_AUTO_SET_LEVERAGE):
            for pos_side in ("long", "short"):
                current_leverage = leverage_map.get(pos_side, leverage_map.get("both", 0.0))
                if abs(current_leverage - target_leverage) > 1e-9:
                    result = self._call_with_retry(
                        f"设置 {pos_side} 杠杆",
                        lambda pos_side=pos_side: self.account_api.set_leverage(
                            lever=str(config.LEVERAGE),
                            mgnMode="cross",
                            instId=config.SYMBOL,
                            posSide=pos_side,
                        ),
                    )
                    if result.get("code") != "0":
                        raise RuntimeError(f"设置 {pos_side} 杠杆失败: {result}")
                    log_info(f"已自动设置 {pos_side} 杠杆为 {config.LEVERAGE}x")
        else:
            for pos_side in ("long", "short"):
                current_leverage = leverage_map.get(pos_side, leverage_map.get("both", 0.0))
                if abs(current_leverage - target_leverage) > 1e-9:
                    raise RuntimeError(
                        f"{pos_side} 杠杆当前为 {current_leverage}x，目标为 {target_leverage}x，且已关闭自动设置"
                    )

        log_info(
            f"交易环境校验完成: simulated={str(config.USE_SERVER) == '1'}, "
            f"pos_mode={pos_mode}, leverage_target={config.LEVERAGE}x"
        )
        # 启动时从 OKX 动态读取合约规格，覆盖 .env 里的硬编码默认值
        self._refresh_instrument_specs()

    def _refresh_instrument_specs(self):
        """从 OKX 动态读取合约规格（lotSz、tickSz、ctVal），覆盖 config 默认值。
        确保换品种或 OKX 调整规格后无需手动修改 .env。
        """
        try:
            result = self._call_read_with_retry(
                "获取合约规格",
                lambda: self.public_api.get_instruments(instType="SWAP", instId=config.SYMBOL),
                require_data=True,
            )
            if result.get("code") != "0" or not result.get("data"):
                log_error(f"获取合约规格失败，继续使用 .env 配置值: {result}")
                return
            inst = result["data"][0]
            lot_sz = float(inst.get("lotSz", config.LOT_SIZE) or config.LOT_SIZE)
            tick_sz = float(inst.get("tickSz", config.TICK_SIZE) or config.TICK_SIZE)
            ct_val = float(inst.get("ctVal", 1.0) or 1.0)
            min_sz = float(inst.get("minSz", lot_sz) or lot_sz)
            # 运行时动态覆盖 config 属性（不影响其他进程）
            config.LOT_SIZE = lot_sz
            config.TICK_SIZE = tick_sz
            config.CT_VAL = ct_val
            config.MIN_SZ = min_sz
            log_info(
                f"合约规格已从 OKX 动态加载: {config.SYMBOL} "
                f"lotSz={lot_sz} tickSz={tick_sz} ctVal={ct_val} minSz={min_sz}"
            )
        except Exception as exc:
            log_error(f"动态加载合约规格异常，继续使用 .env 配置值: {exc}")

    # 获取当前账户余额等信息
    def get_account_balance(self):
        result = self._call_read_with_retry(
            "获取账户余额",
            self.account_api.get_account_balance,
            require_data=True,
        )
        if result.get("code") != "0":
            raise RuntimeError(f"获取账户余额失败: {result}")

        total_eq_raw = result['data'][0].get('totalEq', '0')
        total_eq = float(total_eq_raw) if total_eq_raw not in ['', None] else 0.0
        result['data'][0]['totalEq'] = total_eq

        details = result['data'][0].get('details', [])
        usdt_detail = next((d for d in details if d.get('ccy') == 'USDT'), None)

        if usdt_detail:
            avail_eq_raw = usdt_detail.get('availEq', '0')
            avail_eq = float(avail_eq_raw) if avail_eq_raw not in ['', None] else 0.0
            usdt_eq_raw = usdt_detail.get('eq', '0')
            usdt_eq = float(usdt_eq_raw) if usdt_eq_raw not in ['', None] else 0.0
            cash_bal_raw = usdt_detail.get('cashBal', '0')
            cash_bal = float(cash_bal_raw) if cash_bal_raw not in ['', None] else 0.0
        else:
            avail_eq = 0.0
            usdt_eq = 0.0
            cash_bal = 0.0

        result['data'][0]['availEq'] = avail_eq
        result['data'][0]['usdtEq'] = usdt_eq
        result['data'][0]['cashBal'] = cash_bal

        return result

    # 获取SYMBOL当前最新仓位
    def get_position(self):
        result = self._call_read_with_retry(
            "获取仓位",
            lambda: self.account_api.get_positions(instType='SWAP', instId=config.SYMBOL),
        )
        if result.get("code") != "0":
            raise RuntimeError(f"获取仓位失败: {result}")
        positions = result['data']

        long_position = {'size': 0.0, 'entry_price': 0.0}
        short_position = {'size': 0.0, 'entry_price': 0.0}

        for pos in positions:
            pos_side = pos.get('posSide', '')
            size_raw = pos.get('pos', '0')
            avgPx_raw = pos.get('avgPx', '0')

            size = float(size_raw) if size_raw not in ['', None] else 0.0
            avg_price = float(avgPx_raw) if avgPx_raw not in ['', None] else 0.0

            if pos_side == 'long':
                long_position['size'] = size
                long_position['entry_price'] = avg_price

            elif pos_side == 'short':
                short_position['size'] = size
                short_position['entry_price'] = avg_price

        return long_position, short_position

    # 获取SYMBOL当前最新价格(以usdt计价)
    def get_price(self, max_retry=3, sleep_sec=1):
        def parse_price(result):
            price_raw = result["data"][0].get("last", "0")
            if price_raw in ["", None]:
                raise OKXResponseError("获取价格响应 last 字段为空")
            return float(price_raw)

        return self._call_read_with_retry(
            "获取价格",
            lambda: self.market_api.get_ticker(instId=config.SYMBOL),
            require_data=True,
            transform=parse_price,
            max_retry=max_retry,
            sleep_sec=sleep_sec,
            backoff=1.0,
        )

    # 获取最近已平仓交易的真实收益率（计算reward_risk用）
    def fetch_recent_closed_trades(self, limit=50):
        result = self._call_read_with_retry(
            "获取历史平仓",
            lambda: self.account_api.get_positions_history(instType="SWAP", instId=config.SYMBOL, limit=str(limit)),
        )
        if result.get("code") != "0":
            log_error(f"获取历史平仓失败: {result}")
            return []
        trades = []
        for item in result.get("data", []):
            try:
                open_px = float(item.get("openAvgPx", 0))
                close_px = float(item.get("closeAvgPx", 0))
                size = abs(float(item.get("closeTotalPos", 0)))
                realized_pnl = float(item.get("realizedPnl", 0))
                fee = float(item.get("fee", 0))

                if open_px <= 0 or close_px <= 0 or size <= 0:
                    continue

                avg_px = (open_px + close_px) / 2
                notional = size * avg_px
                if notional <= 0:
                    continue
                net_pnl = realized_pnl + fee
                trade_return = net_pnl / notional
                trades.append(trade_return)

            except Exception:
                continue

        return trades

    # OKX 历史K线完整拉取函数：支持自动分页、稳定拉取大规模历史数据
    def fetch_ohlcv(self,symbol=config.SYMBOL, bar="1H", max_limit=2000, max_retry=3, sleep_sec=1):
        all_data = []
        next_after = ''

        while len(all_data) < max_limit:
            remaining = max_limit - len(all_data)
            limit = min(300, remaining)
            batch = None
            for attempt in range(max_retry):
                try:
                    response = self.market_api.get_history_candlesticks(
                        instId=symbol,
                        bar=bar,
                        limit=limit,
                        after=next_after
                    )
                    batch = response['data']
                    break
                except Exception as e:
                    print(f"⚠️ 拉取K线失败，重试中 ({attempt + 1}/{max_retry}): {e}")
                    time.sleep(sleep_sec)
            else:
                print("❌ 超过最大重试次数，跳过当前分页")
                break

            if not batch:
                break

            batch_sorted = sorted(batch, key=lambda x: int(x[0]))  # 时间升序
            all_data.extend(batch_sorted)

            if len(batch) < limit:
                break  # 没有更多了

            # ✅ 翻页核心逻辑：用最早时间戳向前翻页
            next_after = str(batch_sorted[0][0])

            time.sleep(0.2)  # 防止API限速

        if not all_data:
            raise Exception("❌ 无法拉取任何K线数据，请检查API权限/网络")

        # 转换为DataFrame。分页结果可能按“最近批次在前、历史批次在后”拼接，
        # 这里统一按时间正序排序，并去重，避免滚动特征被乱序数据污染。
        normalized_rows = []
        for row in all_data:
            normalized_rows.append({
                "timestamp": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "confirm": row[8] if len(row) > 8 else "1",
            })

        df = pd.DataFrame(normalized_rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms', utc=True)
        df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        df.sort_values('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df['confirm'] = df['confirm'].astype(str)
        df.reset_index(drop=True, inplace=True)
        return df

    # 批量获取多个周期的k线数据
    def fetch_data(self):
        data_dict = {}
        for interval in config.INTERVALS:
            df = self.fetch_ohlcv(config.SYMBOL, bar=interval, max_limit=config.WINDOWS[interval])
            df.set_index("timestamp", inplace=True)
            data_dict[interval] = df
            time.sleep(0.3)
        return data_dict

    def fetch_funding_rate_history(self, symbol=config.SYMBOL, max_records=None, max_retry=3, sleep_sec=1):
        if max_records is None:
            max_records = config.BACKTEST_FUNDING_HISTORY_LIMIT

        all_rows = []
        next_after = ''

        while len(all_rows) < max_records:
            remaining = max_records - len(all_rows)
            limit = min(100, remaining)
            batch = None

            for attempt in range(max_retry):
                try:
                    response = self.public_api.funding_rate_history(
                        instId=symbol,
                        after=next_after,
                        limit=str(limit),
                    )
                    batch = response.get('data', [])
                    break
                except Exception as e:
                    print(f"⚠️ 拉取 funding 历史失败，重试中 ({attempt + 1}/{max_retry}): {e}")
                    time.sleep(sleep_sec)
            else:
                print("❌ 超过最大重试次数，跳过 funding 历史分页")
                break

            if not batch:
                break

            all_rows.extend(batch)

            if len(batch) < limit:
                break

            oldest_time = min(int(item['fundingTime']) for item in batch)
            next_after = str(oldest_time)
            time.sleep(0.2)

        if not all_rows:
            return pd.DataFrame(columns=['funding_time', 'funding_rate'])

        df = pd.DataFrame(all_rows)
        df['funding_time'] = pd.to_datetime(df['fundingTime'].astype(float), unit='ms', utc=True)
        rate_col = 'realizedRate' if 'realizedRate' in df.columns else 'fundingRate'
        df['funding_rate'] = df[rate_col].astype(float)
        df.drop_duplicates(subset=['funding_time'], keep='last', inplace=True)
        df.sort_values('funding_time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df[['funding_time', 'funding_rate']]

    @staticmethod
    def _ccy_from_symbol(symbol):
        # Rubik 统计接口按币种(ccy)而非合约 instId 查询,例如 SOL-USDT-SWAP -> SOL。
        return str(symbol).split('-')[0]

    def _fetch_rubik_series(self, label, fetch_fn, value_cols, *, period='1H', max_retry=3, sleep_sec=1):
        """通用 Rubik 历史拉取。

        Rubik 端点单次返回约 720 行(1H 约 30 天)、时间倒序、每行是数组。
        这里统一成时间正序的 DataFrame,首列为 ts,其余按 value_cols 命名。
        无历史(快照型)或拉取失败时返回空表,调用方需容忍缺列。
        """
        rows = None
        for attempt in range(max_retry):
            try:
                resp = fetch_fn()
                if str(resp.get('code')) != '0':
                    raise ValueError(f"code={resp.get('code')} msg={resp.get('msg')}")
                rows = resp.get('data', [])
                break
            except Exception as e:
                print(f"⚠️ 拉取 {label} 失败，重试中 ({attempt + 1}/{max_retry}): {e}")
                time.sleep(sleep_sec)
        else:
            print(f"❌ 超过最大重试次数，跳过 {label}")
            rows = None

        cols = ['ts'] + list(value_cols)
        if not rows:
            return pd.DataFrame(columns=cols)

        # 防御:实际列数可能与预期不符(OKX 偶尔加列),按最小长度对齐。
        width = min(len(cols), len(rows[0]))
        df = pd.DataFrame([r[:width] for r in rows], columns=cols[:width])
        df['ts'] = pd.to_datetime(df['ts'].astype('int64'), unit='ms', utc=True)
        for col in cols[1:width]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.drop_duplicates(subset=['ts'], keep='last', inplace=True)
        df.sort_values('ts', inplace=True)  # Rubik 默认倒序,这里转正序
        df.reset_index(drop=True, inplace=True)
        return df

    def fetch_open_interest_history(self, symbol=config.SYMBOL, period='1H', max_retry=3, sleep_sec=1):
        """合约持仓量(OI)+ 成交量历史。行格式 [ts, oi, volume]。"""
        ccy = self._ccy_from_symbol(symbol)
        return self._fetch_rubik_series(
            "OI 历史",
            lambda: self.trading_data_api.get_contracts_interest_volume(ccy=ccy, period=period),
            ['open_interest', 'oi_volume'],
            period=period, max_retry=max_retry, sleep_sec=sleep_sec,
        )

    def fetch_taker_volume_history(self, symbol=config.SYMBOL, period='1H', max_retry=3, sleep_sec=1):
        """主动买卖(taker)成交量历史。OKX 行格式 [ts, sellVol, buyVol]。"""
        ccy = self._ccy_from_symbol(symbol)
        return self._fetch_rubik_series(
            "taker 成交量历史",
            lambda: self.trading_data_api.get_taker_volume(ccy=ccy, instType='CONTRACTS', period=period),
            ['taker_sell_vol', 'taker_buy_vol'],
            period=period, max_retry=max_retry, sleep_sec=sleep_sec,
        )

    def fetch_long_short_ratio_history(self, symbol=config.SYMBOL, period='1H', max_retry=3, sleep_sec=1):
        """多空账户比历史。行格式 [ts, ratio]。"""
        ccy = self._ccy_from_symbol(symbol)
        return self._fetch_rubik_series(
            "多空比历史",
            lambda: self.trading_data_api.get_long_short_ratio(ccy=ccy, period=period),
            ['long_short_ratio'],
            period=period, max_retry=max_retry, sleep_sec=sleep_sec,
        )

    def fetch_rubik_data(self, symbol=config.SYMBOL, period='1H'):
        """一次性拉取 OI / taker / 多空比,返回 add_rubik_features 期望的 dict。"""
        return {
            "open_interest": self.fetch_open_interest_history(symbol=symbol, period=period),
            "taker_volume": self.fetch_taker_volume_history(symbol=symbol, period=period),
            "long_short_ratio": self.fetch_long_short_ratio_history(symbol=symbol, period=period),
        }

    ### 封装开仓/平仓逻辑(按usdt开仓)
    def place_order_with_leverage(self, side, posSide, usd_amount, leverage, reduce_only=False, max_retry=3, sleep_sec=1, fill_timeout_sec=5.0):
        if not isinstance(usd_amount, (int, float)):
            try:
                usd_amount = float(usd_amount)
            except Exception:
                raise Exception(f"❌ usd_amount 类型异常: 传入了无法转换的值 '{usd_amount}'")
        cl_ord_id = build_client_order_id(config.SYMBOL, side, posSide, reduce_only)
        cached_size = None
        for attempt in range(max_retry):
            # 重试时先查既有订单：避免重复下单（幂等保护）
            existing_order = self.get_order_by_client_id(cl_ord_id)
            if order_is_filled(existing_order):
                enriched = self._attach_order_fills(existing_order)
                log_info(f"✅ 订单已成交（查询确认）: clOrdId={cl_ord_id}, state={existing_order.get('state')}")
                return enriched
            if order_is_terminal(existing_order):
                return self._terminal_order_result(existing_order)
            if order_is_acknowledged(existing_order):
                final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                if final_order is not None:
                    log_info(f"✅ 订单已受理后成交: clOrdId={cl_ord_id}")
                    return final_order
                return False

            try:
                # 使用缓存 size 保证幂等（重试时 size 与首次相同）
                if cached_size is None:
                    market_price = self.get_price()

                    # ✅ 资金安全校验 (账户可用保证金检查)
                    if not reduce_only:
                        account_info = self.get_account_balance()
                        available_usdt = float(account_info['data'][0]['availEq'])
                        usable_margin = max(
                            0.0,
                            available_usdt - float(config.LIVE_MIN_FREE_MARGIN_USDT),
                        ) * max(0.0, min(float(config.LIVE_MARGIN_USAGE_RATIO), 1.0))

                        if usd_amount > usable_margin:
                            log_info(
                                f"保证金约束: requested_margin={usd_amount:.2f} USDT, "
                                f"capped_margin={usable_margin:.2f} USDT, avail={available_usdt:.2f} USDT"
                            )
                            usd_amount = usable_margin

                        if usd_amount <= 0:
                            log_error(f"❌ 保证金不足: 可用 {available_usdt:.2f} USDT，取消下单")
                            return False

                    # ✅ 直接读取写死的合约参数
                    lot_size = config.LOT_SIZE
                    tick_size = config.TICK_SIZE

                    # ✅ 合法计算下单数量（注意保险性精度控制）
                    order_value = usd_amount * leverage
                    raw_size = order_value / market_price
                    size = math.floor(raw_size / lot_size) * lot_size
                    size = round(size, 6)

                    if size < lot_size:
                        if reduce_only:
                            log_info(f"🟡 平仓 size={size} 小于最小下单单位 {lot_size}，自动跳过")
                            return False
                        else:
                            raise Exception(f"⚠ 下单失败: 开仓 size={size} 小于最小下单单位 {lot_size}")

                    cached_size = size
                else:
                    size = cached_size

                # ✅ 发单
                result = self.trade_api.place_order(
                    instId=config.SYMBOL,
                    tdMode="cross",
                    side=side,
                    posSide=posSide,
                    ordType="market",
                    sz=str(size),
                    reduceOnly=reduce_only,
                    clOrdId=cl_ord_id,
                )

                if result['code'] == "0":
                    order_id = result['data'][0]['ordId']
                    log_info(
                        f"✅ 下单已提交: {side} {posSide} 杠杆: {leverage}x, 本金: {usd_amount} USD, 下单数量: {size} {config.SYMBOL}, 订单ID: {order_id}, clOrdId: {cl_ord_id}")
                    final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                    if final_order is not None:
                        log_info(f"✅ 下单已成交: ordId={order_id}, state={final_order.get('state')}")
                        return final_order
                    log_error(f"⚠ 下单未成交且已确认终态: clOrdId={cl_ord_id}")
                    return False
                else:
                    existing_order = self.get_order_by_client_id(cl_ord_id)
                    if order_is_filled(existing_order):
                        log_info(f"✅ 下单响应异常但订单已成交: clOrdId={cl_ord_id}")
                        return self._attach_order_fills(existing_order)
                    if order_is_terminal(existing_order):
                        return self._terminal_order_result(existing_order)
                    if order_is_acknowledged(existing_order):
                        final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                        if final_order is not None:
                            log_info(f"✅ 下单响应异常但订单已成交: clOrdId={cl_ord_id}")
                            return final_order
                        return False
                    # ✅ 保险：防止无 data 崩溃
                    error_data = result.get('data', [{}])[0]
                    error_code = error_data.get('sCode', '')
                    error_msg = error_data.get('sMsg', '')
                    log_error(f"❌ 下单失败: 错误码 {error_code}, 原因: {error_msg}")
                    if is_insufficient_margin_error(result):
                        return False
                    time.sleep(sleep_sec)

            except OrderStateUnknownError:
                raise
            except Exception as e:
                existing_order = self.get_order_by_client_id(cl_ord_id)
                if order_is_filled(existing_order):
                    log_info(f"✅ 下单异常但订单已成交: clOrdId={cl_ord_id}")
                    return self._attach_order_fills(existing_order)
                if order_is_terminal(existing_order):
                    return self._terminal_order_result(existing_order)
                if order_is_acknowledged(existing_order):
                    final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                    if final_order is not None:
                        log_info(f"✅ 下单异常但订单已成交: clOrdId={cl_ord_id}")
                        return final_order
                    return False
                log_error(f"⚠ 下单异常({attempt + 1}): {e}")
                time.sleep(sleep_sec)

        # 超过重试次数后失败
        raise Exception("❌ 超过最大重试次数，下单失败")

    # 开多仓(按usdt)
    def open_long(self, usd_amount, leverage):
        return self.place_order_with_leverage("buy", "long", usd_amount, leverage, reduce_only=False)

    # 平多仓(按usdt)
    def close_long(self, usd_amount, leverage):
        long_pos, _ = self.get_position()
        if long_pos['size'] == 0:
            log_info("🟢 无多仓位，跳过平多")
            return False
        return self.place_order_with_leverage("sell", "long", usd_amount, leverage, reduce_only=True)

    # 开空仓(按usdt)
    def open_short(self, usd_amount, leverage):
        return self.place_order_with_leverage("sell", "short", usd_amount, leverage, reduce_only=False)

    # 平空仓(按usdt)
    def close_short(self, usd_amount, leverage):
        _, short_pos = self.get_position()
        if short_pos['size'] == 0:
            log_info("🟢 无空仓位，跳过平空")
            return False
        return self.place_order_with_leverage("buy", "short", usd_amount, leverage, reduce_only=True)

    ### 封装开仓/平仓逻辑(按size开仓)
    def place_order_with_size(self, side, posSide, size, leverage, reduce_only=False, max_retry=3, sleep_sec=1, fill_timeout_sec=5.0):
        """
        按"sz=size"直接下单，避免 usd_amount->size 二次floor，确保与回测 delta_qty 精确对齐。
        """
        if not isinstance(size, (int, float)):
            try:
                size = float(size)
            except Exception:
                raise Exception(f"❌ size 类型异常: '{size}'")

        lot_size = float(config.LOT_SIZE)
        size = floor_size_to_lot(size, lot_size)
        cl_ord_id = build_client_order_id(config.SYMBOL, side, posSide, reduce_only)

        if size < lot_size:
            if reduce_only:
                log_info(f"🟡 reduceOnly 平仓 size={size} 小于最小下单单位 {lot_size}，自动跳过")
                return False
            else:
                raise Exception(f"⚠ 下单失败: 开仓 size={size} 小于最小下单单位 {lot_size}")

        for attempt in range(max_retry):
            # 重试时先查既有订单：避免重复下单（幂等保护）
            if attempt > 0:
                existing_order = self.get_order_by_client_id(cl_ord_id)
                if order_is_filled(existing_order):
                    log_info(f"✅ 订单已成交（查询确认，sz模式）: clOrdId={cl_ord_id}, state={existing_order.get('state')}")
                    return self._attach_order_fills(existing_order)
                if order_is_terminal(existing_order):
                    return self._terminal_order_result(existing_order)
                if order_is_acknowledged(existing_order):
                    final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                    if final_order is not None:
                        log_info(f"✅ 订单已受理后成交(sz模式): clOrdId={cl_ord_id}")
                        return final_order
                    return False

            try:
                if not reduce_only:
                    market_price = self.get_price()

                    # 保证金检查：估算 required_margin = 名义价值 / leverage = size*price/leverage。
                    # 若目标 size 超过可用保证金，按可用额度截断，避免同一类信号反复失败重试。
                    account_info = self.get_account_balance()
                    available_usdt = float(account_info['data'][0]['availEq'])
                    capped_size, required_margin, usable_margin, was_capped = cap_size_by_available_margin(
                        size,
                        market_price,
                        leverage,
                        available_usdt,
                        lot_size,
                        usage_ratio=config.LIVE_MARGIN_USAGE_RATIO,
                        min_free_margin_usdt=config.LIVE_MIN_FREE_MARGIN_USDT,
                    )

                    if was_capped:
                        log_info(
                            f"保证金约束: requested_sz={size}, capped_sz={capped_size}, "
                            f"usable_margin={usable_margin:.2f} USDT, avail={available_usdt:.2f} USDT"
                        )
                        size = capped_size

                    if size < lot_size:
                        log_error(
                            f"❌ 保证金不足: 截断后 size={size} 小于最小下单单位 {lot_size}, "
                            f"usable_margin={usable_margin:.2f} USDT, 可用 {available_usdt:.2f} USDT，取消下单"
                        )
                        return False

                result = self.trade_api.place_order(
                    instId=config.SYMBOL,
                    tdMode="cross",
                    side=side,
                    posSide=posSide,
                    ordType="market",
                    sz=str(size),
                    reduceOnly=reduce_only,
                    clOrdId=cl_ord_id,
                )

                if result['code'] == "0":
                    order_id = result['data'][0]['ordId']
                    log_info(
                        f"✅ 下单已提交(sz模式): {side} {posSide} {leverage}x, sz={size}, reduceOnly={reduce_only}, ordId={order_id}, clOrdId={cl_ord_id}")
                    final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                    if final_order is not None:
                        log_info(f"✅ 下单已成交(sz模式): ordId={order_id}, state={final_order.get('state')}")
                        return final_order
                    log_error(f"⚠ 下单未成交且已确认终态(sz模式): clOrdId={cl_ord_id}")
                    return False
                else:
                    existing_order = self.get_order_by_client_id(cl_ord_id)
                    if order_is_filled(existing_order):
                        log_info(f"✅ 下单响应异常但订单已成交(sz模式): clOrdId={cl_ord_id}")
                        return self._attach_order_fills(existing_order)
                    if order_is_terminal(existing_order):
                        return self._terminal_order_result(existing_order)
                    if order_is_acknowledged(existing_order):
                        final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                        if final_order is not None:
                            log_info(f"✅ 下单响应异常但订单已成交(sz模式): clOrdId={cl_ord_id}")
                            return final_order
                        return False
                    error_data = result.get('data', [{}])[0]
                    error_code = error_data.get('sCode', '')
                    error_msg = error_data.get('sMsg', '')
                    log_error(f"❌ 下单失败(sz模式): 错误码 {error_code}, 原因: {error_msg}")
                    if is_insufficient_margin_error(result):
                        return False
                    time.sleep(sleep_sec)

            except OrderStateUnknownError:
                raise
            except Exception as e:
                existing_order = self.get_order_by_client_id(cl_ord_id)
                if order_is_filled(existing_order):
                    log_info(f"✅ 下单异常但订单已成交(sz模式): clOrdId={cl_ord_id}")
                    return self._attach_order_fills(existing_order)
                if order_is_terminal(existing_order):
                    return self._terminal_order_result(existing_order)
                if order_is_acknowledged(existing_order):
                    final_order = self.wait_until_filled(cl_ord_id, fill_timeout_sec)
                    if final_order is not None:
                        log_info(f"✅ 下单异常但订单已成交(sz模式): clOrdId={cl_ord_id}")
                        return final_order
                    return False
                log_error(f"⚠ 下单异常(sz模式)({attempt + 1}): {e}")
                time.sleep(sleep_sec)

        raise Exception("❌ 超过最大重试次数，下单失败(sz模式)")

    def open_long_sz(self, sz, leverage):
        return self.place_order_with_size("buy", "long", sz, leverage, reduce_only=False)

    def close_long_sz(self, sz, leverage, known_position_size=None):
        if known_position_size is None:
            long_pos, _ = self.get_position()
            position_size = float(long_pos['size'])
        else:
            position_size = max(0.0, float(known_position_size))
        if position_size <= 0:
            log_info("🟢 无多仓位，跳过平多")
            return False
        close_size = min(float(sz), position_size)
        return self.place_order_with_size("sell", "long", close_size, leverage, reduce_only=True)

    def open_short_sz(self, sz, leverage):
        return self.place_order_with_size("sell", "short", sz, leverage, reduce_only=False)

    def close_short_sz(self, sz, leverage, known_position_size=None):
        if known_position_size is None:
            _, short_pos = self.get_position()
            position_size = float(short_pos['size'])
        else:
            position_size = max(0.0, float(known_position_size))
        if position_size <= 0:
            log_info("🟢 无空仓位，跳过平空")
            return False
        close_size = min(float(sz), position_size)
        return self.place_order_with_size("buy", "short", close_size, leverage, reduce_only=True)

    # ─────────────────────────────────────────────────────────────
    # 交易所端 TP/SL 算法单（P0修复：进程崩溃时交易所端止损仍有效）
    # ─────────────────────────────────────────────────────────────

    def place_tpsl_algo_order(
        self,
        pos_side: str,
        sz: float,
        entry_price: float,
        take_profit_ratio: float,
        stop_loss_ratio: float,
    ):
        """在 OKX 下 OCO 算法止盈止损单（tpsl 类型）。

        持仓存续期间即使进程崩溃，止损也不会失效。
        平仓或手动撤销前，该单会持续有效。

        Args:
            pos_side: "long" 或 "short"
            sz:        持仓张数
            entry_price:  实际成交均价
            take_profit_ratio: 止盈距离比例（如 0.026 = 2.6%）
            stop_loss_ratio:   止损距离比例（如 0.012 = 1.2%）

        Returns:
            algo order ID（str）或 None（下单失败）
        """
        if pos_side == "long":
            tp_price = round(entry_price * (1 + take_profit_ratio), 6)
            sl_price = round(entry_price * (1 - stop_loss_ratio), 6)
            side = "sell"
        elif pos_side == "short":
            tp_price = round(entry_price * (1 - take_profit_ratio), 6)
            sl_price = round(entry_price * (1 + stop_loss_ratio), 6)
            side = "buy"
        else:
            log_error(f"place_tpsl_algo_order: 未知 pos_side={pos_side}")
            return None

        lot_size = float(config.LOT_SIZE)
        sz_floors = floor_size_to_lot(sz, lot_size)
        if sz_floors <= 0:
            log_error(f"place_tpsl_algo_order: 下单数量为0，跳过")
            return None

        try:
            # mark price 比 last price 更稳定：不会被闪崩/价格操纵触发止损
            trigger_px_type = str(getattr(config, "TPSL_TRIGGER_PX_TYPE", "mark")).lower()
            result = self._call_with_retry(
                "下 TP/SL 算法单",
                lambda: self.trade_api.place_algo_order(
                    instId=config.SYMBOL,
                    tdMode="cross",
                    side=side,
                    posSide=pos_side,
                    ordType="oco",
                    sz=str(sz_floors),
                    tpTriggerPx=str(tp_price),
                    tpOrdPx="-1",         # 市价触发
                    tpTriggerPxType=trigger_px_type,
                    slTriggerPx=str(sl_price),
                    slOrdPx="-1",         # 市价触发
                    slTriggerPxType=trigger_px_type,
                    reduceOnly=True,
                ),
            )
            if result.get("code") == "0" and result.get("data"):
                algo_id = result["data"][0].get("algoId", "")
                log_info(
                    f"✅ 交易所端 TP/SL 已下单: {pos_side} algoId={algo_id}"
                    f" TP触发={tp_price} SL触发={sl_price} sz={sz_floors}"
                    f" 触发价格类型={trigger_px_type}"
                )
                return algo_id
            else:
                log_error(f"⚠ 交易所端 TP/SL 下单失败: {result}")
                return None
        except Exception as exc:
            log_error(f"⚠ 交易所端 TP/SL 下单异常: {exc}")
            return None

    def list_pending_tpsl_algo_orders(self, pos_side=None):
        result = self._call_read_with_retry(
            "查询待触发 TP/SL 算法单",
            lambda: self.trade_api.order_algos_list(
                ordType="oco",
                instType="SWAP",
                instId=config.SYMBOL,
                limit="100",
            ),
        )
        if str(result.get("code", "")) != "0":
            raise RuntimeError(f"查询待触发 TP/SL 算法单失败: {result}")
        orders = list(result.get("data", []) or [])
        if pos_side is not None:
            orders = [item for item in orders if str(item.get("posSide", "")) == str(pos_side)]
        return orders

    def get_algo_order_details(self, algo_id):
        if not algo_id:
            return None
        result = self._call_read_with_retry(
            "查询 TP/SL 算法单详情",
            lambda: self.trade_api.get_algo_order_details(algoId=str(algo_id)),
        )
        if str(result.get("code", "")) != "0":
            raise RuntimeError(f"查询 TP/SL 算法单详情失败: {result}")
        data = result.get("data", []) or []
        return data[0] if data else None

    def get_order_by_id(self, ord_id):
        if not ord_id:
            return None
        result = self._call_read_with_retry(
            "按 ordId 查询订单",
            lambda: self.trade_api.get_order(instId=config.SYMBOL, ordId=str(ord_id)),
        )
        if str(result.get("code", "")) != "0":
            raise RuntimeError(f"按 ordId 查询订单失败: {result}")
        data = result.get("data", []) or []
        return self._attach_order_fills(data[0]) if data else None

    def fetch_algo_child_orders(self, algo_id):
        detail = self.get_algo_order_details(algo_id)
        if not detail:
            return None, []

        order_ids = detail.get("ordIdList") or []
        if isinstance(order_ids, str):
            order_ids = [order_ids] if order_ids else []
        fallback_order_id = str(detail.get("ordId", "") or "")
        if fallback_order_id:
            order_ids = list(order_ids) + [fallback_order_id]

        child_orders = []
        seen = set()
        for order_id in order_ids:
            order_id = str(order_id or "")
            if not order_id or order_id in seen:
                continue
            seen.add(order_id)
            order = self.get_order_by_id(order_id)
            if order:
                child_orders.append(order)
        return detail, child_orders

    def cancel_algo_order(
        self,
        algo_id: str,
        *,
        confirm_timeout_sec=5.0,
        poll_interval_sec=0.3,
    ) -> bool:
        """撤销指定 algo 算法单（平仓前调用，防止残留单反向开仓）。

        Args:
            algo_id: place_tpsl_algo_order 返回的 algoId

        Returns:
            True 表示成功撤销或已不存在，False 表示撤销失败
        """
        if not algo_id:
            return True
        try:
            result = self._call_with_retry(
                "撤销 TP/SL 算法单",
                lambda: self.trade_api.cancel_algo_order(
                    [{"instId": config.SYMBOL, "algoId": algo_id}]
                ),
            )
            code = str(result.get("code", ""))
            response_data = result.get("data", []) or []
            item_code = str(response_data[0].get("sCode", "") or "") if response_data else ""
            if code == "0" and item_code in {"", "0"}:
                deadline = time.monotonic() + float(confirm_timeout_sec)
                while time.monotonic() < deadline:
                    pending = self.list_pending_tpsl_algo_orders()
                    if not any(str(item.get("algoId", "")) == str(algo_id) for item in pending):
                        log_info(f"✅ 交易所端 TP/SL 算法单已确认撤销: algoId={algo_id}")
                        return True
                    time.sleep(poll_interval_sec)
                log_error(f"⚠ TP/SL 撤单请求已受理但未确认终态: algoId={algo_id}")
                return False
            # 51400/51603/51609: 算法单已触发、已撤销或不存在，都已是终态。
            data = result.get("data", [{}])
            s_code = data[0].get("sCode", "") if data else ""
            if s_code in ("51400", "51603", "51609"):
                log_info(f"交易所端 TP/SL 算法单不存在（已触发/已撤），algoId={algo_id}")
                return True
            log_error(f"⚠ 撤销 TP/SL 算法单失败: {result}")
            return False
        except Exception as exc:
            log_error(f"⚠ 撤销 TP/SL 算法单异常: algoId={algo_id}, err={exc}")
            return False



if __name__ == '__main__':
    client = OKXClient()
    result = client.fetch_data()
    print(result)
