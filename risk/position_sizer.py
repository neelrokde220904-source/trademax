"""Position Sizer — Kelly Criterion + VaR-based position sizing for Indian markets."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from loguru import logger

from broker.portfolio_tracker import portfolio_tracker
from config.settings import settings


class PositionSizer:
    """Calculate position size using fractional Kelly Criterion + VaR constraints.

    Rules:
    - Max 10% of capital per trade
    - Kelly fraction capped at 0.5× Kelly (half-Kelly for safety)
    - VaR (95%) check — reject if position VaR > 2% of portfolio
    - Minimum lot of 1 share
    """

    def __init__(self) -> None:
        self.kelly_fraction = 0.5  # half-Kelly
        self.max_portfolio_risk_pct = 0.02  # max 2% VaR per position
        self.confidence_level = 0.95

    def calculate(
        self,
        signal: dict[str, Any],
        win_rate: float = 0.55,
        avg_win_loss_ratio: float = 1.5,
        daily_returns: list[float] | None = None,
    ) -> int:
        """Calculate optimal position size for a signal.

        Args:
            signal: Trade signal with price, stop_loss, target.
            win_rate: Historical win rate (0-1).
            avg_win_loss_ratio: Average win / average loss.
            daily_returns: Recent daily return series for VaR.

        Returns:
            Number of shares to trade (0 if rejected).
        """
        price = signal.get("price", 0)
        stop_loss = signal.get("stop_loss", 0)

        if price <= 0 or stop_loss <= 0:
            return 0

        available = portfolio_tracker.available_capital
        max_trade_capital = settings.initial_capital * settings.max_capital_per_trade_pct

        # ----- Kelly Criterion -----
        kelly_pct = self._kelly(win_rate, avg_win_loss_ratio)
        kelly_capital = available * kelly_pct

        # ----- Risk-based sizing (% of capital at risk) -----
        risk_per_share = abs(price - stop_loss)
        max_risk = available * self.max_portfolio_risk_pct  # risk 2% of capital
        risk_qty = int(max_risk / risk_per_share) if risk_per_share > 0 else 0

        # ----- Kelly qty -----
        kelly_qty = int(kelly_capital / price) if price > 0 else 0

        # ----- Capital cap -----
        max_qty = int(max_trade_capital / price) if price > 0 else 0

        # Take the minimum of all constraints
        qty = max(1, min(kelly_qty, risk_qty, max_qty))

        # ----- VaR check -----
        if daily_returns and len(daily_returns) > 20:
            position_value = qty * price
            var = self._calculate_var(daily_returns, position_value)
            var_limit = available * self.max_portfolio_risk_pct
            if var > var_limit:
                # Reduce position to fit VaR
                ratio = var_limit / var if var > 0 else 1.0
                qty = max(1, int(qty * ratio))
                logger.info("VaR constraint reduced qty to {} for {}", qty, signal.get("symbol"))

        logger.info(
            "[PositionSizer] {} — Kelly:{} Risk:{} Cap:{} → Final:{}",
            signal.get("symbol", "?"),
            kelly_qty,
            risk_qty,
            max_qty,
            qty,
        )
        return qty

    def _kelly(self, win_rate: float, win_loss_ratio: float) -> float:
        """Half-Kelly fraction: f* = (p*b - q) / b × fraction."""
        p = max(0.01, min(0.99, win_rate))
        q = 1 - p
        b = max(0.01, win_loss_ratio)
        kelly = (p * b - q) / b
        # Clamp to [0, 0.25] — never risk > 25% of capital
        return max(0, min(0.25, kelly * self.kelly_fraction))

    def _calculate_var(
        self, daily_returns: list[float], position_value: float
    ) -> float:
        """Calculate parametric VaR at 95% confidence."""
        returns = np.array(daily_returns)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))
        if sigma == 0:
            return 0.0
        # z-score for 95%
        z = 1.645
        var = position_value * abs(mu - z * sigma)
        return var


# Singleton
position_sizer = PositionSizer()
