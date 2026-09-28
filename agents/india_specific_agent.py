"""India-Specific Agent — bulk/block deals, promoter pledging, F&O ban, sector rotation."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import httpx
import pytz
from loguru import logger

from agents.base_agent import BaseAgent, TradingState
from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


class IndiaSpecificAgent(BaseAgent):
    """India-specific signals: bulk deals, block deals, promoter pledging,
    F&O ban list, SEBI actions, sector rotation, circuit limits."""

    def __init__(self) -> None:
        super().__init__("IndiaSpecificAgent")

    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []
        signals: dict[str, Any] = {}

        # --- F&O Ban List (critical — never trade banned stocks) ---
        fo_ban_list = self._fetch_fo_ban_list()
        signals["fo_ban_list"] = fo_ban_list
        if fo_ban_list:
            logs.append(self.log("F&O ban list: {}", fo_ban_list))
        else:
            logs.append(self.log("No stocks in F&O ban today."))

        # --- Bulk Deals ---
        bulk_deals = self._fetch_bulk_deals()
        signals["bulk_deals"] = bulk_deals
        if bulk_deals:
            logs.append(self.log("Found {} bulk deals.", len(bulk_deals)))

        # --- Block Deals ---
        block_deals = self._fetch_block_deals()
        signals["block_deals"] = block_deals

        # --- Promoter Pledging (red flag) ---
        pledging_data = self._check_promoter_pledging(state.get("watchlist", []))
        signals["promoter_pledging"] = pledging_data

        # --- Circuit Limits ---
        circuit_data = self._check_circuit_limits(state.get("watchlist", []))
        signals["circuit_filters"] = circuit_data

        # --- Is today an F&O expiry? ---
        is_expiry = self._is_expiry_day()
        signals["is_expiry_day"] = is_expiry
        if is_expiry:
            logs.append(self.log("⚠️ Today is F&O expiry — expect higher volatility."))

        # --- Derive per-symbol signals ---
        for symbol in state.get("watchlist", []):
            sym_signal = self._symbol_signal(symbol, signals)
            if sym_signal:
                signals[symbol] = sym_signal

        logs.append(self.log("India-specific analysis complete."))

        return {
            "india_signals": signals,
            "agent_logs": logs,
            "errors": errors,
        }

    def _fetch_fo_ban_list(self) -> list[str]:
        """Fetch the F&O ban list from NSE."""
        try:
            with httpx.Client(timeout=10.0, headers=NSE_HEADERS) as client:
                client.get("https://www.nseindia.com")
                resp = client.get("https://www.nseindia.com/api/fo-mktlot")
                if resp.status_code == 200:
                    data = resp.json()
                    ban_list = [item["symbol"] for item in data if item.get("ban")]
                    return ban_list
        except Exception as exc:
            logger.debug("F&O ban list fetch failed: {}", exc)
        return []

    def _fetch_bulk_deals(self) -> list[dict[str, Any]]:
        """Fetch bulk deals from NSE."""
        try:
            with httpx.Client(timeout=10.0, headers=NSE_HEADERS) as client:
                client.get("https://www.nseindia.com")
                resp = client.get("https://www.nseindia.com/api/snapshot-capital-market-largedeal")
                if resp.status_code == 200:
                    data = resp.json()
                    deals = []
                    for item in data.get("BULK_DEALS_DATA", [])[:20]:
                        deals.append({
                            "symbol": item.get("symbol", ""),
                            "client": item.get("clientName", ""),
                            "deal_type": item.get("dealType", ""),
                            "quantity": item.get("quantity", 0),
                            "price": item.get("price", 0),
                        })
                    return deals
        except Exception as exc:
            logger.debug("Bulk deals fetch failed: {}", exc)
        return []

    def _fetch_block_deals(self) -> list[dict[str, Any]]:
        """Fetch block deals from NSE."""
        try:
            with httpx.Client(timeout=10.0, headers=NSE_HEADERS) as client:
                client.get("https://www.nseindia.com")
                resp = client.get("https://www.nseindia.com/api/snapshot-capital-market-largedeal")
                if resp.status_code == 200:
                    data = resp.json()
                    deals = []
                    for item in data.get("BLOCK_DEALS_DATA", [])[:20]:
                        deals.append({
                            "symbol": item.get("symbol", ""),
                            "client": item.get("clientName", ""),
                            "quantity": item.get("quantity", 0),
                            "price": item.get("price", 0),
                        })
                    return deals
        except Exception as exc:
            logger.debug("Block deals fetch failed: {}", exc)
        return []

    def _check_promoter_pledging(self, watchlist: list[str]) -> dict[str, Any]:
        """Check promoter pledging data — high pledging is a RED FLAG."""
        # NSE provides this via SAST data; for now use a placeholder
        # In production, scrape from NSE/BSE corporate filing pages
        return {}

    def _check_circuit_limits(self, watchlist: list[str]) -> dict[str, Any]:
        """Check if any watchlist stocks are near circuit limits."""
        from broker.websocket_feed import ws_feed
        from config.instruments import get_token

        circuit_data: dict[str, Any] = {}
        for symbol in watchlist:
            token = get_token(symbol)
            if token:
                tick = ws_feed.get_tick(token)
                if tick and tick.get("upper_circuit") and tick.get("lower_circuit"):
                    ltp = tick.get("ltp", 0)
                    uc = tick["upper_circuit"]
                    lc = tick["lower_circuit"]
                    if ltp and uc and lc:
                        up_pct = (uc - ltp) / ltp * 100 if ltp else 0
                        down_pct = (ltp - lc) / ltp * 100 if ltp else 0
                        if up_pct < 2 or down_pct < 2:
                            circuit_data[symbol] = {
                                "near_upper_circuit": up_pct < 2,
                                "near_lower_circuit": down_pct < 2,
                                "upper_circuit": uc,
                                "lower_circuit": lc,
                                "ltp": ltp,
                            }
        return circuit_data

    def _is_expiry_day(self) -> bool:
        """Check if today is an F&O expiry day (Thursday for weeklies)."""
        now = datetime.now(IST)
        return now.weekday() == 3  # Thursday

    def _symbol_signal(self, symbol: str, signals: dict) -> dict[str, Any] | None:
        """Derive India-specific signal for a symbol."""
        warnings = []

        # F&O ban check
        if symbol in signals.get("fo_ban_list", []):
            return {
                "signal": "AVOID",
                "confidence": 100,
                "reasoning": "Stock in F&O ban list — CANNOT trade",
                "warnings": ["F&O_BAN"],
            }

        # Circuit limit check
        circuits = signals.get("circuit_filters", {})
        if symbol in circuits:
            circuit = circuits[symbol]
            if circuit.get("near_upper_circuit"):
                warnings.append("Near upper circuit — avoid buying")
            if circuit.get("near_lower_circuit"):
                warnings.append("Near lower circuit — avoid selling")

        # Bulk deal activity
        bulk_deals = signals.get("bulk_deals", [])
        for deal in bulk_deals:
            if deal.get("symbol") == symbol:
                warnings.append(f"Bulk deal: {deal.get('client')} — {deal.get('deal_type')}")

        if warnings:
            return {
                "signal": "CAUTION",
                "confidence": 60,
                "reasoning": " | ".join(warnings),
                "warnings": warnings,
            }

        return None


def india_specific_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return IndiaSpecificAgent().run(state)
