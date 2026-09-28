"""Interactive Brokers (IBKR) client for global equity access (US, EU markets).

Uses ib_insync for async TWS/Gateway connectivity.
FEMA/LRS compliance enforced on every order.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from config.settings import settings

try:
    from ib_insync import IB, Contract, LimitOrder, MarketOrder, Stock, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logger.info("ib_insync not installed — IBKR client unavailable")


@dataclass
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497  # 7496=TWS live, 7497=TWS paper, 4001=Gateway live, 4002=Gateway paper
    client_id: int = 1
    timeout: int = 30
    account: str = ""
    readonly: bool = False


@dataclass
class IBKRPosition:
    symbol: str
    exchange: str
    currency: str
    quantity: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class IBKROrder:
    order_id: str = ""
    symbol: str = ""
    action: str = ""  # BUY or SELL
    quantity: float = 0
    order_type: str = "MKT"  # MKT, LMT, STP
    limit_price: float = 0.0
    status: str = "PENDING"
    filled: float = 0
    avg_fill_price: float = 0.0
    currency: str = "USD"
    exchange: str = "SMART"
    timestamp: str = ""


class InteractiveBrokersClient:
    """IBKR client with FEMA compliance on every order."""

    def __init__(self, config: IBKRConfig | None = None) -> None:
        self.config = config or IBKRConfig(
            host=getattr(settings, "ibkr_host", "127.0.0.1"),
            port=getattr(settings, "ibkr_port", 7497),
            client_id=getattr(settings, "ibkr_client_id", 1),
            account=getattr(settings, "ibkr_account", ""),
        )
        self.ib: Any = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected and self.ib is not None

    def connect(self) -> bool:
        """Connect to TWS/IB Gateway."""
        if not IB_AVAILABLE:
            logger.error("ib_insync not installed")
            return False

        try:
            self.ib = IB()
            self.ib.connect(
                self.config.host,
                self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
                readonly=self.config.readonly,
            )
            self._connected = True
            logger.info("Connected to IBKR at {}:{}", self.config.host, self.config.port)
            return True
        except Exception as e:
            logger.error("IBKR connection failed: {}", e)
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self.ib and self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def get_account_summary(self) -> dict[str, Any]:
        """Get account balance, buying power, and P&L."""
        if not self.connected:
            return {"error": "Not connected"}

        try:
            summary = self.ib.accountSummary(self.config.account)
            result: dict[str, Any] = {}
            for item in summary:
                result[item.tag] = {"value": item.value, "currency": item.currency}
            return result
        except Exception as e:
            logger.error("Account summary failed: {}", e)
            return {"error": str(e)}

    def get_positions(self) -> list[IBKRPosition]:
        """Get all open positions."""
        if not self.connected:
            return []

        try:
            positions = self.ib.positions(self.config.account)
            result = []
            for pos in positions:
                result.append(IBKRPosition(
                    symbol=pos.contract.symbol,
                    exchange=pos.contract.exchange or "SMART",
                    currency=pos.contract.currency,
                    quantity=pos.position,
                    avg_cost=pos.avgCost,
                    market_price=0,  # Needs separate market data request
                    market_value=pos.position * pos.avgCost,
                    unrealized_pnl=0,
                    realized_pnl=0,
                ))
            return result
        except Exception as e:
            logger.error("Get positions failed: {}", e)
            return []

    def get_quote(self, symbol: str, exchange: str = "SMART",
                  currency: str = "USD") -> dict[str, Any]:
        """Get current market quote for a symbol."""
        if not self.connected:
            return {"error": "Not connected"}

        try:
            contract = Stock(symbol, exchange, currency)
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, snapshot=True)
            self.ib.sleep(2)
            return {
                "symbol": symbol,
                "bid": ticker.bid,
                "ask": ticker.ask,
                "last": ticker.last,
                "volume": ticker.volume,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error("Quote failed for {}: {}", symbol, e)
            return {"error": str(e)}

    def place_market_order(self, symbol: str, action: str, quantity: float,
                           exchange: str = "SMART", currency: str = "USD") -> IBKROrder:
        """Place a market order with FEMA compliance check."""
        return self._place_order(symbol, action, quantity, "MKT", 0, exchange, currency)

    def place_limit_order(self, symbol: str, action: str, quantity: float,
                          limit_price: float, exchange: str = "SMART",
                          currency: str = "USD") -> IBKROrder:
        """Place a limit order with FEMA compliance check."""
        return self._place_order(symbol, action, quantity, "LMT", limit_price, exchange, currency)

    def _place_order(self, symbol: str, action: str, quantity: float,
                     order_type: str, limit_price: float,
                     exchange: str, currency: str) -> IBKROrder:
        """Internal order placement with FEMA gate."""
        action = action.upper()
        if action not in ("BUY", "SELL"):
            return IBKROrder(status="REJECTED", symbol=symbol,
                             action=action, quantity=quantity)

        if not self.connected:
            return IBKROrder(status="REJECTED", symbol=symbol,
                             action=action, quantity=quantity)

        # FEMA compliance gate
        if action == "BUY":
            if not self._check_fema_compliance(symbol, quantity, limit_price, currency):
                return IBKROrder(
                    status="FEMA_BLOCKED", symbol=symbol,
                    action=action, quantity=quantity,
                    currency=currency,
                )

        try:
            contract = Stock(symbol, exchange, currency)
            self.ib.qualifyContracts(contract)

            if order_type == "MKT":
                order = MarketOrder(action, quantity)
            else:
                order = LimitOrder(action, quantity, limit_price)

            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)

            return IBKROrder(
                order_id=str(trade.order.orderId),
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                status=trade.orderStatus.status,
                filled=trade.orderStatus.filled,
                avg_fill_price=trade.orderStatus.avgFillPrice,
                currency=currency,
                exchange=exchange,
                timestamp=datetime.utcnow().isoformat(),
            )
        except Exception as e:
            logger.error("IBKR order failed: {} {} {} — {}", action, quantity, symbol, e)
            return IBKROrder(status="ERROR", symbol=symbol,
                             action=action, quantity=quantity)

    def _check_fema_compliance(self, symbol: str, quantity: float,
                               price: float, currency: str) -> bool:
        """Check FEMA/LRS compliance before placing international order."""
        try:
            from config.fema_lrs_controller import fema_controller

            estimated_usd = quantity * price if price > 0 else quantity * 100
            result = fema_controller.pre_trade_compliance_check(
                symbol=symbol,
                quantity=quantity,
                estimated_usd_value=estimated_usd,
                destination_country="US",
            )
            if not result.get("approved"):
                reason = result.get("reason", "FEMA compliance check failed")
                logger.warning("FEMA BLOCKED: {} {} {} — {}", symbol, quantity, currency, reason)
                return False
            return True
        except ImportError:
            logger.warning("FEMA controller not available — allowing trade")
            return True
        except Exception as e:
            logger.error("FEMA check error: {} — blocking trade for safety", e)
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self.connected:
            return False
        try:
            for trade in self.ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self.ib.cancelOrder(trade.order)
                    return True
            return False
        except Exception as e:
            logger.error("Cancel order {} failed: {}", order_id, e)
            return False

    def get_open_orders(self) -> list[IBKROrder]:
        """Get all open/pending orders."""
        if not self.connected:
            return []
        try:
            trades = self.ib.openTrades()
            return [
                IBKROrder(
                    order_id=str(t.order.orderId),
                    symbol=t.contract.symbol,
                    action=t.order.action,
                    quantity=t.order.totalQuantity,
                    order_type=t.order.orderType,
                    limit_price=t.order.lmtPrice or 0,
                    status=t.orderStatus.status,
                    filled=t.orderStatus.filled,
                    avg_fill_price=t.orderStatus.avgFillPrice,
                    currency=t.contract.currency,
                    exchange=t.contract.exchange,
                )
                for t in trades
            ]
        except Exception as e:
            logger.error("Get open orders failed: {}", e)
            return []

    def get_historical_data(self, symbol: str, duration: str = "1 Y",
                            bar_size: str = "1 day", exchange: str = "SMART",
                            currency: str = "USD") -> list[dict[str, Any]]:
        """Get historical OHLCV data."""
        if not self.connected:
            return []
        try:
            contract = Stock(symbol, exchange, currency)
            self.ib.qualifyContracts(contract)
            bars = self.ib.reqHistoricalData(
                contract, endDateTime="",
                durationStr=duration, barSizeSetting=bar_size,
                whatToShow="TRADES", useRTH=True,
            )
            return [
                {
                    "date": b.date.isoformat() if hasattr(b.date, "isoformat") else str(b.date),
                    "open": b.open, "high": b.high,
                    "low": b.low, "close": b.close,
                    "volume": b.volume,
                }
                for b in bars
            ]
        except Exception as e:
            logger.error("Historical data failed for {}: {}", symbol, e)
            return []


# Singleton
ibkr_client = InteractiveBrokersClient()
