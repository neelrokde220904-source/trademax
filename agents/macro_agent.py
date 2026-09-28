"""Macro Agent — GIFT Nifty, global cues, USD/INR, crude, VIX, FII/DII flows."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import httpx
import pytz
import yfinance as yf
from loguru import logger

from agents.base_agent import BaseAgent, TradingState
from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")


class MacroAgent(BaseAgent):
    """Analyses macro factors: global cues, FII/DII, crude oil, VIX, etc."""

    def __init__(self) -> None:
        super().__init__("MacroAgent")

    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []

        india_vix = state.get("india_vix", 0)

        # Fetch macro data points
        macro_data: dict[str, Any] = {}

        # --- Global Market Cues ---
        global_cues = self._fetch_global_cues()
        macro_data["global_cues"] = global_cues

        # --- USD/INR ---
        usdinr = self._fetch_usdinr()
        macro_data["usdinr"] = usdinr

        # --- Crude Oil ---
        crude = self._fetch_crude_oil()
        macro_data["crude_oil"] = crude

        # --- India VIX assessment ---
        vix_assessment = self._assess_vix(india_vix)
        macro_data["vix_assessment"] = vix_assessment

        # --- FII/DII Data ---
        fii_dii = self._fetch_fii_dii()
        macro_data["fii_dii"] = fii_dii

        # Derive overall macro signal
        signal, confidence, reasoning = self._derive_macro_signal(macro_data)

        signals = {
            "overall": {
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning,
            },
            "data": macro_data,
        }

        logs.append(self.log("Macro signal: {} (confidence: {}%)", signal, confidence))
        logs.append(self.log("VIX: {:.2f} — {}", india_vix, vix_assessment.get("assessment", "N/A")))

        if global_cues:
            logs.append(self.log("Global cues: {}", json.dumps(global_cues, default=str)[:200]))

        return {
            "macro_signals": signals,
            "fii_dii_data": fii_dii,
            "agent_logs": logs,
            "errors": errors,
        }

    def _fetch_global_cues(self) -> dict[str, Any]:
        """Fetch overnight global market data."""
        cues: dict[str, Any] = {}
        tickers = {
            "sp500": "^GSPC",
            "dow_jones": "^DJI",
            "nasdaq": "^IXIC",
            "nikkei": "^N225",
            "hang_seng": "^HSI",
        }

        for name, ticker_sym in tickers.items():
            try:
                ticker = yf.Ticker(ticker_sym)
                hist = ticker.history(period="2d")
                if len(hist) >= 2:
                    prev_close = float(hist.iloc[-2]["Close"])
                    last_close = float(hist.iloc[-1]["Close"])
                    change_pct = (last_close - prev_close) / prev_close * 100
                    cues[name] = {"close": round(last_close, 2), "change_pct": round(change_pct, 2)}
            except Exception as exc:
                logger.debug("Failed to fetch {}: {}", name, exc)

        return cues

    def _fetch_usdinr(self) -> dict[str, Any] | None:
        """Fetch USD/INR exchange rate."""
        try:
            ticker = yf.Ticker("INR=X")
            hist = ticker.history(period="2d")
            if not hist.empty:
                rate = float(hist.iloc[-1]["Close"])
                return {"rate": round(rate, 4)}
        except Exception as exc:
            logger.debug("USD/INR fetch failed: {}", exc)
        return None

    def _fetch_crude_oil(self) -> dict[str, Any] | None:
        """Fetch crude oil price (India is oil importer — crude up = bearish)."""
        try:
            ticker = yf.Ticker("CL=F")  # WTI Crude
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                prev = float(hist.iloc[-2]["Close"])
                current = float(hist.iloc[-1]["Close"])
                change_pct = (current - prev) / prev * 100
                return {"price": round(current, 2), "change_pct": round(change_pct, 2)}
        except Exception as exc:
            logger.debug("Crude oil fetch failed: {}", exc)
        return None

    def _assess_vix(self, vix: float) -> dict[str, Any]:
        """Assess India VIX level."""
        if vix == 0:
            return {"level": "UNKNOWN", "assessment": "VIX data unavailable"}
        elif vix < 14:
            return {"level": "LOW", "assessment": "Complacency — trend continuation likely"}
        elif vix < 18:
            return {"level": "NORMAL", "assessment": "Normal volatility — full position sizing OK"}
        elif vix < 22:
            return {"level": "ELEVATED", "assessment": "Elevated volatility — reduce position size"}
        elif vix < 28:
            return {"level": "HIGH", "assessment": "High fear — reduce position size by 50%"}
        else:
            return {"level": "EXTREME", "assessment": "Extreme fear — HALT new trades, protect capital"}

    def _fetch_fii_dii(self) -> dict[str, Any]:
        """Fetch FII/DII provisional data from NSE."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }
            # NSE FII/DII data endpoint
            import httpx as httpx_sync
            with httpx_sync.Client(timeout=10.0, headers=headers) as client:
                # Get cookies first
                client.get("https://www.nseindia.com")
                resp = client.get("https://www.nseindia.com/api/fiidiiTradeReact")
                if resp.status_code == 200:
                    data = resp.json()
                    return data
        except Exception as exc:
            logger.debug("FII/DII data fetch failed: {}", exc)
        return {}

    def _derive_macro_signal(self, data: dict) -> tuple[str, int, str]:
        """Derive overall macro signal from all data points."""
        bullish = 0
        bearish = 0
        reasons: list[str] = []

        # Global cues
        global_cues = data.get("global_cues", {})
        global_positive = sum(1 for v in global_cues.values() if isinstance(v, dict) and v.get("change_pct", 0) > 0)
        global_negative = sum(1 for v in global_cues.values() if isinstance(v, dict) and v.get("change_pct", 0) < 0)
        global_total = global_positive + global_negative

        if global_total > 0:
            if global_positive > global_negative:
                bullish += 1
                reasons.append(f"Global cues positive ({global_positive}/{global_total})")
            else:
                bearish += 1
                reasons.append(f"Global cues negative ({global_negative}/{global_total})")

        # Crude oil (India = importer, crude up = bearish)
        crude = data.get("crude_oil")
        if crude and isinstance(crude, dict):
            change = crude.get("change_pct", 0)
            if change > 2:
                bearish += 1
                reasons.append(f"Crude up {change:.1f}% (bearish for India)")
            elif change < -2:
                bullish += 1
                reasons.append(f"Crude down {change:.1f}% (bullish for India)")

        # VIX
        vix_data = data.get("vix_assessment", {})
        vix_level = vix_data.get("level", "UNKNOWN")
        if vix_level in ("HIGH", "EXTREME"):
            bearish += 2
            reasons.append(f"VIX {vix_level} — high fear")
        elif vix_level == "ELEVATED":
            bearish += 1
            reasons.append(f"VIX elevated")
        elif vix_level == "LOW":
            bullish += 1
            reasons.append("VIX low — complacency")

        # Derive signal
        if bullish >= 3 and bearish <= 1:
            return "BUY", min(80, 50 + bullish * 10), " | ".join(reasons)
        elif bearish >= 3 and bullish <= 1:
            return "SELL", min(80, 50 + bearish * 10), " | ".join(reasons)
        else:
            return "HOLD", 40, " | ".join(reasons) if reasons else "Mixed macro signals"


def macro_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return MacroAgent().run(state)
