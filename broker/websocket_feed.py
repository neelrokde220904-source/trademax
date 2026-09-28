"""SmartWebSocketV2 — real-time tick data feed from Angel One."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime
from typing import Any, Callable

import pytz
from loguru import logger
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

from broker.angel_client import angel_client
from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")

# WebSocket mode constants
MODE_LTP = 1
MODE_QUOTE = 2
MODE_FULL = 3  # LTP + OHLC + Volume + OI + depth


class WebSocketFeed:
    """Manages SmartWebSocketV2 connections for real-time market data."""

    def __init__(self) -> None:
        self._ws: SmartWebSocketV2 | None = None
        self._subscriptions: dict[str, list[str]] = {}  # exchange → [tokens]
        self._tick_data: dict[str, dict[str, Any]] = {}  # token → latest tick
        self._callbacks: list[Callable] = []
        self._connected = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Establish WebSocket connection."""
        if self._connected:
            logger.warning("WebSocket already connected.")
            return

        angel_client.ensure_session()

        try:
            self._ws = SmartWebSocketV2(
                angel_client.jwt_token,
                settings.angel_api_key,
                settings.angel_client_id,
                angel_client.feed_token,
            )

            self._ws.on_open = self._on_open
            self._ws.on_data = self._on_data
            self._ws.on_error = self._on_error
            self._ws.on_close = self._on_close

            self._thread = threading.Thread(target=self._ws.connect, daemon=True)
            self._thread.start()

            logger.info("WebSocket connection initiated.")

        except Exception as exc:
            logger.exception("WebSocket connection failed: {}", exc)

    def _on_open(self, wsapp: Any) -> None:
        """Called when WebSocket opens."""
        self._connected = True
        logger.info("WebSocket connected successfully.")

        # Resubscribe to all tokens
        if self._subscriptions:
            self._do_subscribe()

    def _on_data(self, wsapp: Any, message: dict) -> None:
        """Called on each tick — store and forward to callbacks."""
        try:
            token = str(message.get("token", ""))
            with self._lock:
                self._tick_data[token] = {
                    "token": token,
                    "ltp": message.get("last_traded_price", 0) / 100.0,
                    "open": message.get("open_price_of_the_day", 0) / 100.0,
                    "high": message.get("high_price_of_the_day", 0) / 100.0,
                    "low": message.get("low_price_of_the_day", 0) / 100.0,
                    "close": message.get("closed_price", 0) / 100.0,
                    "volume": message.get("volume_trade_for_the_day", 0),
                    "oi": message.get("open_interest", 0),
                    "best_bid": message.get("best_5_buy_data", []),
                    "best_ask": message.get("best_5_sell_data", []),
                    "upper_circuit": message.get("upper_circuit_limit", 0) / 100.0,
                    "lower_circuit": message.get("lower_circuit_limit", 0) / 100.0,
                    "timestamp": datetime.now(IST).isoformat(),
                }

            # Fire callbacks
            for cb in self._callbacks:
                try:
                    cb(self._tick_data[token])
                except Exception as e:
                    logger.error("Tick callback error: {}", e)

        except Exception as exc:
            logger.error("Tick processing error: {}", exc)

    def _on_error(self, wsapp: Any, error: Any) -> None:
        """Called on WebSocket error."""
        logger.error("WebSocket error: {}", error)
        self._connected = False

    def _on_close(self, wsapp: Any) -> None:
        """Called when WebSocket closes — attempt reconnect."""
        logger.warning("WebSocket closed. Reconnecting in 5 seconds...")
        self._connected = False
        time.sleep(5)
        self.connect()

    def subscribe(self, exchange: str, tokens: list[str], mode: int = MODE_FULL) -> None:
        """Subscribe to tokens for real-time data.

        Args:
            exchange: 'nse_cm', 'nse_fo', 'bse_cm', etc.
            tokens: List of Angel One tokens (strings).
            mode: 1=LTP, 2=Quote, 3=Full (default).
        """
        if exchange not in self._subscriptions:
            self._subscriptions[exchange] = []
        self._subscriptions[exchange].extend(tokens)
        # Deduplicate
        self._subscriptions[exchange] = list(set(self._subscriptions[exchange]))

        if self._connected:
            self._do_subscribe(mode=mode)

    def _do_subscribe(self, mode: int = MODE_FULL) -> None:
        """Send subscribe request to WebSocket."""
        if not self._ws:
            return

        token_list = []
        for exchange, tokens in self._subscriptions.items():
            for token in tokens:
                token_list.append({
                    "exchangeType": self._exchange_type_code(exchange),
                    "tokens": [token],
                })

        if token_list:
            try:
                correlation_id = "india_trader_ws"
                self._ws.subscribe(correlation_id, mode, token_list)
                logger.info("Subscribed to {} tokens in mode {}.", sum(len(t["tokens"]) for t in token_list), mode)
            except Exception as exc:
                logger.error("Subscribe error: {}", exc)

    def _exchange_type_code(self, exchange: str) -> int:
        """Map exchange string to Angel One exchange type code."""
        mapping = {
            "nse_cm": 1,
            "nse_fo": 2,
            "bse_cm": 3,
            "bse_fo": 4,
            "mcx_fo": 5,
            "ncx_fo": 7,
            "cde_fo": 13,
        }
        return mapping.get(exchange.lower(), 1)

    def get_tick(self, token: str) -> dict[str, Any] | None:
        """Get the latest tick data for a token."""
        with self._lock:
            return self._tick_data.get(token)

    def get_all_ticks(self) -> dict[str, dict[str, Any]]:
        """Get all latest tick data."""
        with self._lock:
            return dict(self._tick_data)

    def register_callback(self, callback: Callable) -> None:
        """Register a function to call on every tick."""
        self._callbacks.append(callback)

    def unsubscribe(self, exchange: str, tokens: list[str]) -> None:
        """Unsubscribe from tokens."""
        if exchange in self._subscriptions:
            for t in tokens:
                if t in self._subscriptions[exchange]:
                    self._subscriptions[exchange].remove(t)

        if self._ws and self._connected:
            try:
                token_list = [{"exchangeType": self._exchange_type_code(exchange), "tokens": tokens}]
                self._ws.unsubscribe("india_trader_ws", MODE_FULL, token_list)
            except Exception as exc:
                logger.error("Unsubscribe error: {}", exc)

    def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._ws:
            try:
                self._ws.close_connection()
            except Exception:
                pass
        self._connected = False
        logger.info("WebSocket disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected


# Module-level singleton
ws_feed = WebSocketFeed()
