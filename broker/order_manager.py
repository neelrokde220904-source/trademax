"""Order Manager — places, modifies, and cancels orders with paper-trading support."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytz
from loguru import logger

from broker.angel_client import angel_client
from config.settings import settings
from database.db import get_db
from database.models import Trade

IST = pytz.timezone("Asia/Kolkata")


def _is_market_hours() -> bool:
    """Check if current time is within NSE market hours (9:15 – 15:30 IST, Mon-Fri)."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    today_str = now.strftime("%Y-%m-%d")
    if today_str in settings.nse_holidays:
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def place_order(
    symbol: str,
    token: str,
    action: str,
    quantity: int,
    order_type: str = "LIMIT",
    price: float = 0.0,
    trigger_price: float = 0.0,
    stop_loss: float | None = None,
    target: float | None = None,
    exchange: str = "NSE",
    product_type: str = "INTRADAY",
    strategy: str = "",
    confidence: float = 0.0,
    reasoning: str = "",
) -> dict[str, Any] | None:
    """Place an order — paper or live depending on config.

    Returns:
        Dict with order details or None on failure.
    """
    # Market-hours guard for live trading
    if not settings.paper_trading and not _is_market_hours():
        logger.warning("Blocked live order for {} — outside market hours.", symbol)
        return None

    order_params = {
        "variety": "NORMAL",
        "tradingsymbol": symbol,
        "symboltoken": token,
        "transactiontype": action.upper(),
        "exchange": exchange,
        "ordertype": order_type,
        "producttype": product_type,
        "duration": "DAY",
        "price": str(price),
        "squareoff": "0",
        "stoploss": "0",
        "quantity": str(quantity),
        "triggerprice": str(trigger_price),
    }

    if settings.paper_trading:
        return _paper_trade(order_params, symbol, action, quantity, price, stop_loss, target, exchange, strategy, confidence, reasoning)
    else:
        return _live_trade(order_params, symbol, action, quantity, price, stop_loss, target, exchange, strategy, confidence, reasoning)


def _paper_trade(
    order_params: dict,
    symbol: str,
    action: str,
    quantity: int,
    price: float,
    stop_loss: float | None,
    target: float | None,
    exchange: str,
    strategy: str,
    confidence: float,
    reasoning: str,
) -> dict[str, Any]:
    """Simulate order execution with slippage."""
    # Get LTP for realistic fill
    ltp = (
        angel_client.get_ltp(exchange, symbol, order_params["symboltoken"])
        if angel_client.has_credentials
        else None
    )
    if ltp is None:
        ltp = price  # fallback to requested price

    # Apply slippage
    if action.upper() == "BUY":
        fill_price = ltp * (1 + settings.slippage_pct)
    else:
        fill_price = ltp * (1 - settings.slippage_pct)

    # Log to database
    with get_db() as db:
        trade = Trade(
            timestamp=datetime.now(IST),
            symbol=symbol,
            exchange=exchange,
            action=action.upper(),
            quantity=quantity,
            order_type=order_params["ordertype"],
            price=round(fill_price, 2),
            stop_loss=stop_loss,
            target=target,
            holding_period="INTRADAY" if order_params.get("producttype") == "INTRADAY" else "DELIVERY",
            status="EXECUTED",
            order_id=f"PAPER-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            is_paper=True,
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
        )
        db.add(trade)

    result = {
        "order_id": trade.order_id,
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "fill_price": round(fill_price, 2),
        "price": round(fill_price, 2),
        "stop_loss": stop_loss,
        "target": target,
        "holding_period": "INTRADAY" if order_params.get("producttype") == "INTRADAY" else "DELIVERY",
        "strategy": strategy,
        "is_paper": True,
        "status": "EXECUTED",
    }

    logger.info("[PAPER] {} {}x {} @ ₹{:.2f}", action, quantity, symbol, fill_price)
    return result


def _live_trade(
    order_params: dict,
    symbol: str,
    action: str,
    quantity: int,
    price: float,
    stop_loss: float | None,
    target: float | None,
    exchange: str,
    strategy: str,
    confidence: float,
    reasoning: str,
) -> dict[str, Any] | None:
    """Execute a live order via Angel One."""
    try:
        resp = angel_client.place_order(order_params)
        order_id = resp if isinstance(resp, str) else (resp.get("data", {}).get("orderid") if isinstance(resp, dict) else None)

        with get_db() as db:
            trade = Trade(
                timestamp=datetime.now(IST),
                symbol=symbol,
                exchange=exchange,
                action=action.upper(),
                quantity=quantity,
                order_type=order_params["ordertype"],
                price=price,
                stop_loss=stop_loss,
                target=target,
                holding_period="INTRADAY" if order_params.get("producttype") == "INTRADAY" else "DELIVERY",
                status="PENDING",
                order_id=str(order_id) if order_id else None,
                is_paper=False,
                strategy=strategy,
                confidence=confidence,
                reasoning=reasoning,
            )
            db.add(trade)

        logger.info("[LIVE] {} {}x {} @ ₹{:.2f} — order_id: {}", action, quantity, symbol, price, order_id)
        return {
            "order_id": order_id,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "stop_loss": stop_loss,
            "target": target,
            "holding_period": "INTRADAY" if order_params.get("producttype") == "INTRADAY" else "DELIVERY",
            "strategy": strategy,
            "is_paper": False,
            "status": "PENDING",
        }

    except Exception as exc:
        logger.exception("Live order failed for {}: {}", symbol, exc)
        return None


def cancel_order(order_id: str, variety: str = "NORMAL") -> bool:
    """Cancel an order by ID."""
    if settings.paper_trading:
        logger.info("[PAPER] Cancel order {}", order_id)
        with get_db() as db:
            trade = db.query(Trade).filter(Trade.order_id == order_id).first()
            if trade:
                trade.status = "CANCELLED"
        return True
    else:
        resp = angel_client.cancel_order(order_id, variety)
        return resp is not None


def get_open_orders() -> list[dict[str, Any]]:
    """Get all pending/open orders."""
    if settings.paper_trading:
        with get_db() as db:
            trades = db.query(Trade).filter(Trade.status == "PENDING", Trade.is_paper == True).all()
            return [{"order_id": t.order_id, "symbol": t.symbol, "action": t.action, "quantity": t.quantity, "price": t.price} for t in trades]
    else:
        return angel_client.get_order_book()
