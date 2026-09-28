"""Angel One SmartAPI wrapper — authentication, session management, and order placement."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Any

import pyotp
from loguru import logger
from SmartApi import SmartConnect
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings


class AngelClient:
    """Thread-safe Angel One SmartAPI client with auto-TOTP and session refresh."""

    def __init__(self) -> None:
        self._api: SmartConnect | None = None
        self._jwt_token: str = ""
        self._refresh_token: str = ""
        self._feed_token: str = ""
        self._last_auth: datetime | None = None
        self._lock = threading.Lock()
        self._authenticated = False

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def authenticate(self) -> bool:
        """Authenticate with Angel One using API key, client ID, MPIN, and auto-generated TOTP."""
        with self._lock:
            try:
                self._api = SmartConnect(api_key=settings.angel_api_key)
                totp = pyotp.TOTP(settings.angel_totp_secret).now()
                data = self._api.generateSession(
                    settings.angel_client_id,
                    settings.angel_mpin,
                    totp,
                )

                if not data or data.get("status") is False:
                    msg = data.get("message", "Unknown auth error") if data else "No response"
                    logger.error("Angel One auth failed: {}", msg)
                    return False

                self._jwt_token = data["data"]["jwtToken"]
                self._refresh_token = data["data"]["refreshToken"]
                self._feed_token = self._api.getfeedToken()
                self._last_auth = datetime.now()
                self._authenticated = True

                logger.info(
                    "Angel One authenticated for client {}. Feed token obtained.",
                    settings.angel_client_id,
                )
                return True

            except Exception as exc:
                logger.exception("Angel One authentication error: {}", exc)
                self._authenticated = False
                raise

    def refresh_session(self) -> bool:
        """Refresh the JWT token using the refresh token. Call every ~6 hours."""
        with self._lock:
            if not self._api or not self._refresh_token:
                logger.warning("Cannot refresh — no active session. Re-authenticating...")
                return self.authenticate()

            try:
                token_data = self._api.generateToken(self._refresh_token)
                if token_data and token_data.get("status"):
                    self._jwt_token = token_data["data"]["jwtToken"]
                    self._refresh_token = token_data["data"]["refreshToken"]
                    self._last_auth = datetime.now()
                    logger.info("Angel One session refreshed.")
                    return True
                else:
                    logger.warning("Token refresh failed, re-authenticating...")
                    return self.authenticate()
            except Exception as exc:
                logger.exception("Session refresh error: {}", exc)
                return self.authenticate()

    def ensure_session(self) -> None:
        """Ensure we have a valid session; refresh if >6 hours old."""
        if not self._authenticated:
            self.authenticate()
            return
        if self._last_auth and (datetime.now() - self._last_auth) > timedelta(hours=6):
            self.refresh_session()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def api(self) -> SmartConnect:
        """Return the underlying SmartConnect instance."""
        self.ensure_session()
        assert self._api is not None, "Client not authenticated"
        return self._api

    @property
    def feed_token(self) -> str:
        self.ensure_session()
        return self._feed_token

    @property
    def jwt_token(self) -> str:
        self.ensure_session()
        return self._jwt_token

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    @property
    def has_credentials(self) -> bool:
        """Whether a complete non-placeholder Angel One login is configured.

        Paper trading can run without this integration, using public OHLCV data
        and simulated requested-price fills.  Callers use this guard to avoid
        slow, repeated authentication attempts when no broker is configured.
        """
        required = (
            settings.angel_api_key,
            settings.angel_client_id,
            settings.angel_mpin,
            settings.angel_totp_secret,
        )
        return all(value and not value.startswith("your_") for value in required)

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def get_candle_data(
        self,
        exchange: str,
        symbol_token: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> list[list[Any]]:
        """Fetch historical OHLCV candle data.

        Args:
            exchange: 'NSE' or 'NFO'
            symbol_token: Angel One token from instrument master
            interval: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, ONE_DAY, etc.
            from_date: 'YYYY-MM-DD HH:MM' format
            to_date: 'YYYY-MM-DD HH:MM' format

        Returns:
            List of candle data lists.
        """
        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date,
        }
        resp = self.api.getCandleData(params)
        if resp and resp.get("status"):
            return resp.get("data", [])
        logger.warning("getCandleData failed: {}", resp)
        return []

    def get_ltp(self, exchange: str, symbol: str, token: str) -> float | None:
        """Get Last Traded Price for a symbol."""
        try:
            resp = self.api.ltpData(exchange, symbol, token)
            if resp and resp.get("status"):
                return float(resp["data"]["ltp"])
        except Exception as exc:
            logger.error("LTP fetch error for {}: {}", symbol, exc)
        return None

    def get_quote(self, exchange: str, symbol: str, token: str) -> dict[str, Any] | None:
        """Get full quote data for a symbol."""
        try:
            resp = self.api.ltpData(exchange, symbol, token)
            if resp and resp.get("status"):
                return resp["data"]
        except Exception as exc:
            logger.error("Quote fetch error for {}: {}", symbol, exc)
        return None

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=3))
    def place_order(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """Place an order via Angel One.

        Args:
            params: Order parameters dict with keys:
                variety, tradingsymbol, symboltoken, transactiontype,
                exchange, ordertype, producttype, duration, price,
                squareoff, stoploss, quantity, triggerprice
        """
        try:
            resp = self.api.placeOrder(params)
            logger.info("Order placed: {} → {}", params.get("tradingsymbol"), resp)
            return resp
        except Exception as exc:
            logger.exception("Order placement failed for {}: {}", params.get("tradingsymbol"), exc)
            raise

    def modify_order(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """Modify an existing order."""
        try:
            resp = self.api.modifyOrder(params)
            logger.info("Order modified: {}", resp)
            return resp
        except Exception as exc:
            logger.exception("Order modification failed: {}", exc)
            return None

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> dict[str, Any] | None:
        """Cancel an order by order ID."""
        try:
            resp = self.api.cancelOrder(order_id, variety)
            logger.info("Order cancelled: {} → {}", order_id, resp)
            return resp
        except Exception as exc:
            logger.exception("Order cancellation failed for {}: {}", order_id, exc)
            return None

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def get_holdings(self) -> list[dict[str, Any]]:
        """Fetch current holdings."""
        try:
            resp = self.api.holding()
            if resp and resp.get("status"):
                return resp.get("data", [])
        except Exception as exc:
            logger.error("Holdings fetch error: {}", exc)
        return []

    def get_positions(self) -> list[dict[str, Any]]:
        """Fetch current open positions."""
        try:
            resp = self.api.position()
            if resp and resp.get("status"):
                return resp.get("data", [])
        except Exception as exc:
            logger.error("Positions fetch error: {}", exc)
        return []

    def get_order_book(self) -> list[dict[str, Any]]:
        """Fetch the order book."""
        try:
            resp = self.api.orderBook()
            if resp and resp.get("status"):
                return resp.get("data", []) or []
        except Exception as exc:
            logger.error("Order book fetch error: {}", exc)
        return []

    # ------------------------------------------------------------------
    # Margin
    # ------------------------------------------------------------------

    def get_rms_limits(self) -> dict[str, Any] | None:
        """Fetch RMS (risk management) margin limits."""
        try:
            resp = self.api.rmsLimit()
            if resp and resp.get("status"):
                return resp.get("data")
        except Exception as exc:
            logger.error("RMS limits fetch error: {}", exc)
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def logout(self) -> None:
        """Logout and invalidate session."""
        if self._api:
            try:
                self._api.terminateSession(settings.angel_client_id)
                logger.info("Angel One session terminated.")
            except Exception as exc:
                logger.warning("Logout error: {}", exc)
        self._authenticated = False
        self._api = None


# Module-level singleton
angel_client = AngelClient()
