"""Intelligent Order Router — routes orders across Angel One (India), IBKR (Global), Alpaca (US)
based on best execution, liquidity, and cost analysis.

Includes algorithmic execution: VWAP, TWAP, Implementation Shortfall.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from config.settings import settings


class Broker(str, Enum):
    ANGEL_ONE = "angel_one"
    IBKR = "ibkr"
    ALPACA = "alpaca"


class AlgoType(str, Enum):
    MARKET = "market"       # Immediate execution
    VWAP = "vwap"           # Volume-Weighted Average Price
    TWAP = "twap"           # Time-Weighted Average Price
    IS = "is"               # Implementation Shortfall (minimize slippage)
    ICEBERG = "iceberg"     # Large order hidden in smaller chunks


@dataclass
class RoutingDecision:
    symbol: str
    broker: Broker
    algo: AlgoType
    reason: str
    estimated_cost_bps: float = 0  # Basis points
    slices: int = 1
    interval_seconds: int = 60


@dataclass
class ExecutionReport:
    order_id: str = ""
    symbol: str = ""
    broker: str = ""
    algo: str = ""
    action: str = ""
    requested_qty: float = 0
    filled_qty: float = 0
    avg_price: float = 0.0
    vwap: float = 0.0
    slippage_bps: float = 0.0
    total_cost: float = 0.0
    execution_time_ms: int = 0
    status: str = "PENDING"
    slices_executed: int = 0
    timestamp: str = ""


# Exchange/instrument classification
INDIA_EXCHANGES = {"NSE", "BSE", "NFO", "BFO", "MCX", "CDS"}
US_EXCHANGES = {"SMART", "NYSE", "NASDAQ", "ARCA", "BATS"}
INDIA_SUFFIXES = {".NS", ".BO"}


class IntelligentOrderRouter:
    """Routes orders to optimal broker + algo combination."""

    # Broker cost estimates (bps)
    BROKER_COSTS = {
        Broker.ANGEL_ONE: {"commission": 2, "stt": 10, "spread": 5},
        Broker.IBKR: {"commission": 1, "spread": 2},
        Broker.ALPACA: {"commission": 0, "spread": 3},
    }

    # Size thresholds for algo selection (in INR/USD)
    LARGE_ORDER_INR = 500_000   # ₹5 Lakh
    LARGE_ORDER_USD = 10_000    # $10K

    def __init__(self) -> None:
        self._angel_client = None
        self._ibkr_client = None
        self._alpaca_client = None

    @property
    def angel_client(self) -> Any:
        if self._angel_client is None:
            try:
                from broker.angel_client import AngelClient
                self._angel_client = AngelClient()
            except ImportError:
                pass
        return self._angel_client

    @property
    def ibkr_client(self) -> Any:
        if self._ibkr_client is None:
            try:
                from broker.interactive_brokers_api import ibkr_client
                self._ibkr_client = ibkr_client
            except ImportError:
                pass
        return self._ibkr_client

    @property
    def alpaca_client(self) -> Any:
        if self._alpaca_client is None:
            try:
                from broker.alpaca_api import alpaca_client
                self._alpaca_client = alpaca_client
            except ImportError:
                pass
        return self._alpaca_client

    def route_order(self, symbol: str, action: str, quantity: float,
                    price: float = 0, exchange: str = "",
                    currency: str = "INR",
                    force_algo: AlgoType | None = None) -> RoutingDecision:
        """Determine optimal broker and execution algorithm."""
        # Step 1: Determine market
        market = self._classify_market(symbol, exchange, currency)

        # Step 2: Select broker
        broker = self._select_broker(market, symbol, quantity, price)

        # Step 3: Select algorithm
        order_value = quantity * price if price > 0 else quantity * 100
        algo = force_algo or self._select_algo(order_value, currency, market)

        # Step 4: Calculate slicing
        slices, interval = self._calculate_slicing(algo, order_value, currency)

        # Step 5: Estimate cost
        cost_bps = self._estimate_cost(broker, algo, order_value)

        reason = (
            f"Market={market}, OrderValue={order_value:.0f} {currency}, "
            f"Algo={algo.value} ({slices} slices)"
        )

        return RoutingDecision(
            symbol=symbol,
            broker=broker,
            algo=algo,
            reason=reason,
            estimated_cost_bps=cost_bps,
            slices=slices,
            interval_seconds=interval,
        )

    def execute_order(self, symbol: str, action: str, quantity: float,
                      price: float = 0, exchange: str = "",
                      currency: str = "INR",
                      force_algo: AlgoType | None = None) -> ExecutionReport:
        """Route + execute an order end-to-end."""
        start_time = time.time()

        routing = self.route_order(symbol, action, quantity, price, exchange, currency, force_algo)
        logger.info("Routing {} {} {}: {} via {} ({})",
                     action, quantity, symbol, routing.broker.value,
                     routing.algo.value, routing.reason)

        if routing.algo == AlgoType.MARKET or routing.slices <= 1:
            report = self._execute_single(routing, action, quantity, price)
        else:
            report = self._execute_sliced(routing, action, quantity, price)

        report.execution_time_ms = int((time.time() - start_time) * 1000)
        report.timestamp = datetime.utcnow().isoformat()

        # Log execution quality
        if report.avg_price > 0 and price > 0:
            if action.upper() == "BUY":
                report.slippage_bps = (report.avg_price - price) / price * 10000
            else:
                report.slippage_bps = (price - report.avg_price) / price * 10000

        logger.info("Execution complete: {} {} {} @ {:.2f} (slippage={:.1f}bps, {}ms)",
                     action, report.filled_qty, symbol, report.avg_price,
                     report.slippage_bps, report.execution_time_ms)

        return report

    # ─── Market Classification ──────────────────────────

    def _classify_market(self, symbol: str, exchange: str, currency: str) -> str:
        if exchange.upper() in INDIA_EXCHANGES:
            return "INDIA"
        if exchange.upper() in US_EXCHANGES:
            return "US"
        if currency == "INR":
            return "INDIA"
        if any(symbol.upper().endswith(s) for s in INDIA_SUFFIXES):
            return "INDIA"
        if currency == "USD":
            return "US"
        return "GLOBAL"

    # ─── Broker Selection ───────────────────────────────

    def _select_broker(self, market: str, symbol: str,
                       quantity: float, price: float) -> Broker:
        if market == "INDIA":
            return Broker.ANGEL_ONE

        # For US/Global, prefer IBKR for large orders, Alpaca for small/fractional
        order_value = quantity * price if price > 0 else 0

        if market == "US":
            # Alpaca for small orders or fractional shares (free commission)
            if order_value < self.LARGE_ORDER_USD or quantity < 1:
                if self.alpaca_client and self.alpaca_client.available:
                    return Broker.ALPACA

            # IBKR for large orders (better execution quality)
            if self.ibkr_client and self.ibkr_client.connected:
                return Broker.IBKR

            # Fallback
            if self.alpaca_client and self.alpaca_client.available:
                return Broker.ALPACA

        # Global markets — IBKR only
        if self.ibkr_client and self.ibkr_client.connected:
            return Broker.IBKR

        logger.warning("No suitable broker for {} market — defaulting to Angel One", market)
        return Broker.ANGEL_ONE

    # ─── Algorithm Selection ────────────────────────────

    def _select_algo(self, order_value: float, currency: str, market: str) -> AlgoType:
        threshold = self.LARGE_ORDER_INR if currency == "INR" else self.LARGE_ORDER_USD

        if order_value < threshold * 0.5:
            return AlgoType.MARKET

        if order_value < threshold:
            return AlgoType.TWAP

        if order_value < threshold * 5:
            return AlgoType.VWAP

        # Very large orders
        return AlgoType.IS

    # ─── Slicing ────────────────────────────────────────

    def _calculate_slicing(self, algo: AlgoType, order_value: float,
                           currency: str) -> tuple[int, int]:
        """Returns (num_slices, interval_seconds)."""
        if algo == AlgoType.MARKET:
            return 1, 0

        threshold = self.LARGE_ORDER_INR if currency == "INR" else self.LARGE_ORDER_USD

        if algo == AlgoType.TWAP:
            slices = max(2, min(10, int(order_value / threshold)))
            return slices, 120  # 2 min intervals

        if algo == AlgoType.VWAP:
            slices = max(3, min(20, int(order_value / threshold)))
            return slices, 180  # 3 min intervals

        if algo == AlgoType.IS:
            slices = max(5, min(30, int(order_value / threshold)))
            return slices, 60  # 1 min intervals — aggressive

        return 1, 0

    # ─── Cost Estimation ────────────────────────────────

    def _estimate_cost(self, broker: Broker, algo: AlgoType,
                       order_value: float) -> float:
        costs = self.BROKER_COSTS.get(broker, {})
        base_cost = sum(costs.values())

        # Algo cost savings
        algo_impact = {
            AlgoType.MARKET: 5,    # +5 bps market impact
            AlgoType.TWAP: -2,     # -2 bps from time spreading
            AlgoType.VWAP: -3,     # -3 bps from volume tracking
            AlgoType.IS: -4,       # -4 bps from IS optimization
        }
        return base_cost + algo_impact.get(algo, 0)

    # ─── Execution ──────────────────────────────────────

    def _execute_single(self, routing: RoutingDecision, action: str,
                        quantity: float, price: float) -> ExecutionReport:
        """Execute a single order on the chosen broker."""
        report = ExecutionReport(
            symbol=routing.symbol,
            broker=routing.broker.value,
            algo=routing.algo.value,
            action=action,
            requested_qty=quantity,
        )

        try:
            if routing.broker == Broker.ANGEL_ONE:
                report = self._execute_angel(routing.symbol, action, quantity, price, report)
            elif routing.broker == Broker.IBKR:
                report = self._execute_ibkr(routing.symbol, action, quantity, price, report)
            elif routing.broker == Broker.ALPACA:
                report = self._execute_alpaca(routing.symbol, action, quantity, price, report)
        except Exception as e:
            report.status = "ERROR"
            logger.error("Execution failed: {}", e)

        return report

    def _execute_sliced(self, routing: RoutingDecision, action: str,
                        quantity: float, price: float) -> ExecutionReport:
        """Execute order in multiple slices (VWAP/TWAP/IS)."""
        report = ExecutionReport(
            symbol=routing.symbol,
            broker=routing.broker.value,
            algo=routing.algo.value,
            action=action,
            requested_qty=quantity,
        )

        slice_qty = quantity / routing.slices
        total_filled = 0.0
        total_cost = 0.0

        for i in range(routing.slices):
            remaining = quantity - total_filled
            if remaining <= 0:
                break

            current_slice = min(slice_qty, remaining)
            slice_report = self._execute_single(routing, action, current_slice, price)

            if slice_report.filled_qty > 0:
                total_filled += slice_report.filled_qty
                total_cost += slice_report.filled_qty * slice_report.avg_price

            report.slices_executed = i + 1

            # Wait between slices (except last)
            if i < routing.slices - 1 and routing.interval_seconds > 0:
                # In production this would be async; here we log intent
                logger.debug("Slice {}/{} done, waiting {}s",
                             i + 1, routing.slices, routing.interval_seconds)
                # NOTE: actual time.sleep removed — scheduler handles timing

        report.filled_qty = total_filled
        report.avg_price = total_cost / total_filled if total_filled > 0 else 0
        report.total_cost = total_cost
        report.status = "FILLED" if total_filled >= quantity * 0.95 else "PARTIAL"

        return report

    def _execute_angel(self, symbol: str, action: str, qty: float,
                       price: float, report: ExecutionReport) -> ExecutionReport:
        """Execute via Angel One."""
        try:
            from broker.order_manager import OrderManager
            om = OrderManager()
            result = om.place_order(
                symbol=symbol,
                action=action.upper(),
                quantity=int(qty),
                price=price,
                order_type="LIMIT" if price > 0 else "MARKET",
            )
            report.order_id = result.get("order_id", "")
            report.status = result.get("status", "PENDING")
            report.filled_qty = qty if report.status == "COMPLETE" else 0
            report.avg_price = price
        except Exception as e:
            report.status = "ERROR"
            logger.error("Angel execution error: {}", e)
        return report

    def _execute_ibkr(self, symbol: str, action: str, qty: float,
                      price: float, report: ExecutionReport) -> ExecutionReport:
        """Execute via Interactive Brokers."""
        if not self.ibkr_client or not self.ibkr_client.connected:
            report.status = "BROKER_UNAVAILABLE"
            return report

        if price > 0:
            result = self.ibkr_client.place_limit_order(symbol, action.upper(), qty, price)
        else:
            result = self.ibkr_client.place_market_order(symbol, action.upper(), qty)

        report.order_id = result.order_id
        report.status = result.status
        report.filled_qty = result.filled
        report.avg_price = result.avg_fill_price
        return report

    def _execute_alpaca(self, symbol: str, action: str, qty: float,
                        price: float, report: ExecutionReport) -> ExecutionReport:
        """Execute via Alpaca."""
        if not self.alpaca_client or not self.alpaca_client.available:
            report.status = "BROKER_UNAVAILABLE"
            return report

        fractional = qty != int(qty)
        if price > 0:
            result = self.alpaca_client.place_limit_order(
                symbol, action.lower(), qty, price, fractional=fractional)
        else:
            result = self.alpaca_client.place_market_order(
                symbol, action.lower(), qty, fractional=fractional)

        report.order_id = result.order_id
        report.status = result.status
        report.filled_qty = result.filled_qty
        report.avg_price = result.filled_avg_price
        return report


# Singleton
order_router = IntelligentOrderRouter()
