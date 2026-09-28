"""NIFTY Options Seller — sells OTM options on NIFTY/BANKNIFTY for premium income."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytz
from loguru import logger

from config.settings import settings
from data.instrument_master import find_nfo_options
from data.options_data import calculate_pcr, fetch_options_chain

IST = pytz.timezone("Asia/Kolkata")


class NiftyOptionsSeller:
    """Core strategy: sell OTM options on NIFTY/BANKNIFTY when VIX is low.

    Rules:
    - Only sell when India VIX < 18
    - 7-14 DTE optimal theta decay
    - Sell OTM Call (spot + 2-3%)
    - Sell OTM Put (spot - 2-3%)
    - Stop loss: 2× premium received
    - Target: 50% of premium (close early)
    - Execute Mon/Tue for weekly expiry (Thu)
    - Execute 10-12 DTE for monthly expiry
    """

    def __init__(self) -> None:
        self.name = "NiftyOptionsSeller"

    def scan(
        self,
        underlying: str = "NIFTY",
        spot_price: float = 0.0,
        india_vix: float = 0.0,
        expiry_contains: str = "",
    ) -> list[dict[str, Any]]:
        """Scan for option selling opportunities.

        Returns:
            List of trade recommendations.
        """
        trades: list[dict[str, Any]] = []

        # VIX filter — only sell when low volatility
        if india_vix > 18:
            logger.info("[{}] VIX {:.1f} > 18 — not ideal for selling. Skipping.", self.name, india_vix)
            return trades

        if spot_price <= 0:
            logger.warning("[{}] No spot price available.", self.name)
            return trades

        # Day of week filter: prefer Mon/Tue for weekly expiry
        now = datetime.now(IST)
        day = now.weekday()
        if day > 2:  # Wed/Thu/Fri — weekly expiry too close or past
            logger.info("[{}] Day {} — prefer Mon/Tue for weekly entry.", self.name, day)

        # Calculate OTM strikes (2-3% away from spot)
        call_strike_min = spot_price * 1.02
        call_strike_max = spot_price * 1.03
        put_strike_min = spot_price * 0.97
        put_strike_max = spot_price * 0.98

        # Fetch options chain
        chain = fetch_options_chain(underlying, expiry_contains=expiry_contains)
        if chain.empty:
            return trades

        # Find best OTM Call to sell
        ce_candidates = chain[
            (chain["option_type"] == "CE")
            & (chain["strike"] >= call_strike_min)
            & (chain["strike"] <= call_strike_max)
            & (chain["ltp"] > 0)
        ]

        if not ce_candidates.empty:
            best_ce = ce_candidates.sort_values("ltp", ascending=False).iloc[0]
            premium = float(best_ce["ltp"])
            trades.append({
                "symbol": best_ce["symbol"],
                "token": best_ce["token"],
                "action": "SELL",
                "strike": float(best_ce["strike"]),
                "option_type": "CE",
                "premium": premium,
                "lot_size": int(best_ce["lot_size"]),
                "quantity": int(best_ce["lot_size"]),
                "stop_loss_premium": round(premium * 2, 2),   # 2× premium
                "target_premium": round(premium * 0.5, 2),     # 50% of premium
                "order_type": "LIMIT",
                "price": premium,
                "holding_period": "EXPIRY",
                "strategy": self.name,
                "reasoning": f"Sell OTM CE at {best_ce['strike']} — VIX {india_vix:.1f}, premium ₹{premium}",
            })

        # Find best OTM Put to sell
        pe_candidates = chain[
            (chain["option_type"] == "PE")
            & (chain["strike"] >= put_strike_min)
            & (chain["strike"] <= put_strike_max)
            & (chain["ltp"] > 0)
        ]

        if not pe_candidates.empty:
            best_pe = pe_candidates.sort_values("ltp", ascending=False).iloc[0]
            premium = float(best_pe["ltp"])
            trades.append({
                "symbol": best_pe["symbol"],
                "token": best_pe["token"],
                "action": "SELL",
                "strike": float(best_pe["strike"]),
                "option_type": "PE",
                "premium": premium,
                "lot_size": int(best_pe["lot_size"]),
                "quantity": int(best_pe["lot_size"]),
                "stop_loss_premium": round(premium * 2, 2),
                "target_premium": round(premium * 0.5, 2),
                "order_type": "LIMIT",
                "price": premium,
                "holding_period": "EXPIRY",
                "strategy": self.name,
                "reasoning": f"Sell OTM PE at {best_pe['strike']} — VIX {india_vix:.1f}, premium ₹{premium}",
            })

        logger.info("[{}] Generated {} option selling trades for {}.", self.name, len(trades), underlying)
        return trades
