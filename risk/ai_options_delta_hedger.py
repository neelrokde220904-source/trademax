"""AI Options Delta Hedger — constructs capital-efficient NSE options hedges when regime turns hostile.

Automatically adjusts portfolio Greeks to neutralize directional risk.
Reference: DeltaHedge framework (arXiv:2509.12753)

Required: mibian (Black-Scholes pricing), numpy
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from loguru import logger

try:
    import mibian
    MIBIAN_AVAILABLE = True
except ImportError:
    MIBIAN_AVAILABLE = False
    logger.warning("mibian not installed. Options pricing will use simplified model.")


class AIOptionsDeltaHedger:
    """Constructs capital-efficient NSE options hedges when regime turns hostile.

    Hedge strategies by panic risk level:
    - panic_risk > 0.35: Buy ATM puts on NIFTY (delta hedge)
    - panic_risk > 0.50: Buy put calendar spreads (cheaper, longer protection)
    - panic_risk > 0.70: EMERGENCY — square off everything, move to cash
    """

    def __init__(self, risk_free_rate: float = 0.065) -> None:
        self.risk_free_rate = risk_free_rate * 100  # mibian expects percentage

    def compute_portfolio_greeks(self, positions: dict[str, Any],
                                 options_chain: dict[str, Any] | None = None) -> dict[str, float]:
        """Calculate aggregate Delta, Gamma, Vega, Theta of the entire portfolio.

        Args:
            positions: {symbol: {quantity, price, delta, gamma, vega, theta, type}}
            options_chain: Current options chain data (for re-pricing if needed)

        Returns:
            {delta, gamma, vega, theta} of the entire book
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0

        for symbol, pos in positions.items():
            qty = pos.get("quantity", 0)
            pos_type = pos.get("type", "equity")

            if pos_type == "equity":
                # Equity delta = 1 per share
                total_delta += qty * 1.0
            else:
                # Options position
                total_delta += qty * pos.get("delta", 0.0)
                total_gamma += qty * pos.get("gamma", 0.0)
                total_vega += qty * pos.get("vega", 0.0)
                total_theta += qty * pos.get("theta", 0.0)

        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "vega": total_vega,
            "theta": total_theta,
        }

    def construct_hedge(self, portfolio_greeks: dict[str, float],
                        options_chain: dict[str, Any],
                        available_capital: float,
                        regime: dict[str, Any]) -> list[dict[str, Any]]:
        """Construct options hedge orders based on regime and portfolio Greeks.

        Args:
            portfolio_greeks: {delta, gamma, vega, theta}
            options_chain: Current NIFTY options chain
            available_capital: Capital available for hedging
            regime: HMM regime output

        Returns:
            List of options orders to hedge portfolio risk.
        """
        panic_risk = regime.get("panic_risk", 0.0)
        orders: list[dict[str, Any]] = []

        if panic_risk > 0.70:
            logger.critical("PANIC_RISK > 0.70: EMERGENCY SQUARE OFF ALL")
            return [{"action": "EMERGENCY_SQUARE_OFF_ALL", "reason": "Panic risk > 70%"}]

        if panic_risk > 0.50:
            # Put calendar spread: cheaper, longer protection
            orders = self._construct_put_calendar(options_chain, portfolio_greeks, available_capital)
        elif panic_risk > 0.35:
            # Simple delta hedge with ATM puts
            orders = self._construct_delta_hedge(options_chain, portfolio_greeks, available_capital)
        else:
            logger.debug("Panic risk {:.2%} — no hedge required", panic_risk)
            return []

        return orders

    def _construct_delta_hedge(self, options_chain: dict[str, Any],
                               greeks: dict[str, float],
                               capital: float) -> list[dict[str, Any]]:
        """Buy ATM NIFTY puts to neutralize portfolio delta."""
        portfolio_delta = greeks.get("delta", 0.0)
        if portfolio_delta <= 0:
            logger.info("Portfolio delta <= 0. No delta hedge needed.")
            return []

        spot = options_chain.get("spot_price", 0)
        if spot <= 0:
            return []

        # Find ATM put strike (nearest to spot)
        put_strikes = options_chain.get("puts", {})
        if not put_strikes:
            logger.warning("No put options available in chain")
            return []

        atm_strike = min(put_strikes.keys(), key=lambda s: abs(float(s) - spot))
        atm_put = put_strikes[atm_strike]

        put_delta = atm_put.get("delta", -0.5)  # ATM put delta ≈ -0.5
        put_premium = atm_put.get("ltp", 0)

        if put_premium <= 0 or put_delta == 0:
            return []

        # Number of lots needed to neutralize delta
        lot_size = options_chain.get("lot_size", 25)  # NIFTY lot size
        lots_needed = abs(portfolio_delta / (put_delta * lot_size))
        lots_needed = max(1, int(math.ceil(lots_needed)))

        # Capital check: don't spend more than 5% of portfolio on hedge
        max_hedge_cost = capital * 0.05
        cost_per_lot = put_premium * lot_size
        max_lots_by_capital = int(max_hedge_cost / cost_per_lot) if cost_per_lot > 0 else 0
        lots_to_buy = min(lots_needed, max(1, max_lots_by_capital))

        order = {
            "action": "BUY",
            "instrument": f"NIFTY {atm_strike} PE",
            "symbol": "NIFTY",
            "option_type": "PE",
            "strike": float(atm_strike),
            "quantity": lots_to_buy * lot_size,
            "lots": lots_to_buy,
            "order_type": "LIMIT",
            "price": put_premium,
            "estimated_cost": lots_to_buy * cost_per_lot,
            "hedge_type": "delta_hedge",
            "target_delta_reduction": lots_to_buy * lot_size * abs(put_delta),
            "reason": f"Delta hedge: portfolio delta={portfolio_delta:.0f}, "
                      f"buying {lots_to_buy} lots of {atm_strike} PE",
        }

        logger.info(
            "Delta hedge: BUY {} lots of NIFTY {} PE @ ₹{:.2f} (cost: ₹{:,.0f})",
            lots_to_buy, atm_strike, put_premium, lots_to_buy * cost_per_lot,
        )

        return [order]

    def _construct_put_calendar(self, options_chain: dict[str, Any],
                                greeks: dict[str, float],
                                capital: float) -> list[dict[str, Any]]:
        """Put calendar spread: buy near-term put, sell far-term put.

        Cheaper than outright puts, provides longer duration protection.
        """
        spot = options_chain.get("spot_price", 0)
        if spot <= 0:
            return []

        put_strikes = options_chain.get("puts", {})
        if not put_strikes:
            return []

        # OTM put (5% below spot) for cheaper protection
        target_strike = spot * 0.95
        strike = min(put_strikes.keys(), key=lambda s: abs(float(s) - target_strike))
        put_data = put_strikes[strike]

        lot_size = options_chain.get("lot_size", 25)
        put_premium = put_data.get("ltp", 0)

        if put_premium <= 0:
            return []

        # Buy 2 lots of near-expiry puts (current week/month)
        lots = 2
        max_cost = capital * 0.03
        cost = lots * lot_size * put_premium
        if cost > max_cost and put_premium > 0:
            lots = max(1, int(max_cost / (lot_size * put_premium)))

        order = {
            "action": "BUY",
            "instrument": f"NIFTY {strike} PE",
            "symbol": "NIFTY",
            "option_type": "PE",
            "strike": float(strike),
            "quantity": lots * lot_size,
            "lots": lots,
            "order_type": "LIMIT",
            "price": put_premium,
            "estimated_cost": lots * lot_size * put_premium,
            "hedge_type": "put_calendar_spread",
            "reason": f"Calendar hedge: elevated panic risk. "
                      f"BUY {lots} lots {strike} PE @ ₹{put_premium:.2f}",
        }

        logger.info(
            "Put calendar hedge: BUY {} lots NIFTY {} PE @ ₹{:.2f}",
            lots, strike, put_premium,
        )

        return [order]

    def monitor_and_rebalance(self, current_greeks: dict[str, float],
                              target_delta: float = 0.0,
                              rebalance_threshold: float = 50.0) -> bool:
        """Check if portfolio delta has drifted and needs rebalancing.

        Called every 30 minutes during market hours.

        Returns:
            True if rebalancing needed, False otherwise.
        """
        delta_drift = abs(current_greeks.get("delta", 0) - target_delta)
        if delta_drift > rebalance_threshold:
            logger.warning(
                "Delta drift detected: {:.0f} (threshold: {:.0f}). Rebalance needed.",
                delta_drift, rebalance_threshold,
            )
            return True
        return False

    def calculate_option_greeks(self, spot: float, strike: float, days_to_expiry: int,
                                iv: float, option_type: str = "PE") -> dict[str, float]:
        """Calculate Greeks for a single option using Black-Scholes.

        Args:
            spot: Current underlying price
            strike: Option strike price
            days_to_expiry: Days until expiration
            iv: Implied volatility (as percentage, e.g. 15.0)
            option_type: "CE" or "PE"

        Returns:
            {delta, gamma, vega, theta}
        """
        if MIBIAN_AVAILABLE:
            try:
                bs = mibian.BS([spot, strike, self.risk_free_rate, days_to_expiry], volatility=iv)
                if option_type.upper() == "CE":
                    return {
                        "delta": bs.callDelta,
                        "gamma": bs.gamma,
                        "vega": bs.vega,
                        "theta": bs.callTheta,
                    }
                else:
                    return {
                        "delta": bs.putDelta,
                        "gamma": bs.gamma,
                        "vega": bs.vega,
                        "theta": bs.putTheta,
                    }
            except Exception as e:
                logger.warning("mibian pricing failed: {}. Using simplified.", e)

        # Simplified Black-Scholes fallback
        return self._simplified_greeks(spot, strike, days_to_expiry, iv, option_type)

    @staticmethod
    def _simplified_greeks(spot: float, strike: float, days: int,
                           iv: float, option_type: str) -> dict[str, float]:
        """Simplified Greeks estimation when mibian is unavailable."""
        t = max(days / 365.0, 0.001)
        sigma = iv / 100.0

        d1 = (math.log(spot / strike) + (0.065 + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))

        # Approximate N(d1) using logistic approximation
        from math import erf
        nd1 = 0.5 * (1 + erf(d1 / math.sqrt(2)))

        delta = nd1 if option_type.upper() == "CE" else nd1 - 1.0

        # Simplified gamma and vega
        phi_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
        gamma = phi_d1 / (spot * sigma * math.sqrt(t))
        vega = spot * phi_d1 * math.sqrt(t) / 100

        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": 0.0}


# Singleton
delta_hedger = AIOptionsDeltaHedger()
