"""Market Data Agent — fetches and preprocesses all market data for the pipeline."""

from __future__ import annotations

from typing import Any

from loguru import logger

from agents.base_agent import BaseAgent, TradingState
from broker.angel_client import angel_client
from broker.portfolio_tracker import portfolio_tracker
from broker.websocket_feed import ws_feed
from config.instruments import get_token
from config.settings import settings
from data.market_data import fetch_multiple, fetch_index_data


class MarketDataAgent(BaseAgent):
    """Fetches historical OHLCV, live quotes, VIX, and portfolio state."""

    def __init__(self) -> None:
        super().__init__("MarketDataAgent")

    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []

        watchlist = state["watchlist"]
        logs.append(self.log("Fetching market data for {} symbols.", len(watchlist)))

        # ----- Historical OHLCV -----
        market_data = {}
        try:
            market_data = fetch_multiple(watchlist, interval="15m", days=30)
            logs.append(self.log("Fetched historical data for {} symbols.", len(market_data)))
        except Exception as exc:
            err = self.log("Historical data fetch failed: {}", exc)
            errors.append(err)

        # ----- Live Quotes from WebSocket -----
        live_quotes = {}
        for sym in watchlist:
            token = get_token(sym)
            if token:
                tick = ws_feed.get_tick(token)
                if tick:
                    live_quotes[sym] = tick
                elif angel_client.has_credentials:
                    # Fallback to REST API LTP
                    ltp = angel_client.get_ltp("NSE", sym, token)
                    if ltp:
                        live_quotes[sym] = {"ltp": ltp, "token": token}

        logs.append(self.log("Got live quotes for {} symbols.", len(live_quotes)))

        # ----- India VIX -----
        india_vix = 0.0
        try:
            vix_data = fetch_index_data("99926004", interval="1d", days=5)  # India VIX token
            if not vix_data.empty:
                india_vix = float(vix_data.iloc[-1]["close"])
            logs.append(self.log("India VIX: {:.2f}", india_vix))
        except Exception as exc:
            logs.append(self.log("VIX fetch skipped: {}", exc))

        # ----- Portfolio Sync -----
        try:
            portfolio_tracker.sync_from_broker()
            portfolio_tracker.update_unrealised_pnl()
        except Exception as exc:
            errors.append(self.log("Portfolio sync error: {}", exc))

        return {
            "market_data": market_data,
            "live_quotes": live_quotes,
            "india_vix": india_vix,
            "portfolio_value": portfolio_tracker.total_value,
            "available_capital": portfolio_tracker.available_capital,
            "current_positions": portfolio_tracker.positions,
            "agent_logs": logs,
            "errors": errors,
        }


def market_data_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return MarketDataAgent().run(state)
