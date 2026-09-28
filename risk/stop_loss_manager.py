"""Stop-Loss Manager — ATR trailing stops and bracket target management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytz
from loguru import logger

from broker.angel_client import angel_client
from broker.portfolio_tracker import portfolio_tracker
from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")


class StopLossManager:
    """Manages stop-losses and targets for open positions.

    Features:
    - Initial SL from signal (ATR-based)
    - Trailing stop at 2× ATR from highest price
    - Time-based exit: close intraday positions by 3:15 PM IST
    - Target exit at specified target price
    """

    def __init__(self) -> None:
        # tracking_data: {symbol: {stop_loss, target, highest_since_entry, trailing_atr, entry_time, holding_period}}
        self._tracking: dict[str, dict[str, Any]] = {}

    def register_trade(self, trade: dict[str, Any]) -> None:
        """Register a new trade for stop-loss tracking."""
        symbol = trade["symbol"]
        self._tracking[symbol] = {
            "stop_loss": trade.get("stop_loss", 0),
            "target": trade.get("target", 0),
            "highest_since_entry": trade.get("price", 0),
            "lowest_since_entry": trade.get("price", 0),
            "trailing_atr": trade.get("atr", 0),
            "entry_time": datetime.now(IST),
            "holding_period": trade.get("holding_period", "INTRADAY"),
            "action": trade.get("action", "BUY"),
        }
        logger.info(
            "[SL] Registered {} — SL:₹{:.2f} Target:₹{:.2f}",
            symbol, trade.get("stop_loss", 0), trade.get("target", 0),
        )

    def unregister_trade(self, symbol: str) -> None:
        """Remove tracking for a closed position."""
        self._tracking.pop(symbol, None)

    def check_all_positions(self) -> list[dict[str, Any]]:
        """Check all tracked positions for exit conditions.

        Returns:
            List of exit signals: [{symbol, reason, exit_type}]
        """
        exits: list[dict[str, Any]] = []

        for symbol, data in list(self._tracking.items()):
            pos = portfolio_tracker.positions.get(symbol)
            if not pos:
                self.unregister_trade(symbol)
                continue

            ltp = pos.get("ltp", 0)
            if ltp <= 0:
                continue

            action = data["action"]

            # Update trailing data
            if action == "BUY":
                data["highest_since_entry"] = max(data["highest_since_entry"], ltp)
            else:
                data["lowest_since_entry"] = min(data["lowest_since_entry"], ltp)

            # Check exit conditions
            exit_info = self._check_exit(symbol, ltp, data)
            if exit_info:
                exits.append(exit_info)

        return exits

    def _check_exit(self, symbol: str, ltp: float, data: dict) -> dict[str, Any] | None:
        action = data["action"]
        stop_loss = data["stop_loss"]
        target = data["target"]
        atr = data.get("trailing_atr", 0)

        # --- TRAILING STOP (for longs) ---
        if action == "BUY" and atr > 0:
            trail_sl = data["highest_since_entry"] - settings.trailing_stop_atr_multiplier * atr
            if trail_sl > stop_loss:
                data["stop_loss"] = trail_sl
                stop_loss = trail_sl

        # --- TRAILING STOP (for shorts) ---
        if action == "SELL" and atr > 0:
            trail_sl = data["lowest_since_entry"] + settings.trailing_stop_atr_multiplier * atr
            if trail_sl < stop_loss:
                data["stop_loss"] = trail_sl
                stop_loss = trail_sl

        # --- STOP-LOSS HIT ---
        if action == "BUY" and ltp <= stop_loss:
            return {
                "symbol": symbol,
                "reason": f"Stop-loss hit at ₹{ltp:.2f} (SL: ₹{stop_loss:.2f})",
                "exit_type": "STOP_LOSS",
                "exit_action": "SELL",
            }
        if action == "SELL" and ltp >= stop_loss:
            return {
                "symbol": symbol,
                "reason": f"Stop-loss hit at ₹{ltp:.2f} (SL: ₹{stop_loss:.2f})",
                "exit_type": "STOP_LOSS",
                "exit_action": "BUY",
            }

        # --- TARGET HIT ---
        if action == "BUY" and target > 0 and ltp >= target:
            return {
                "symbol": symbol,
                "reason": f"Target hit at ₹{ltp:.2f} (Target: ₹{target:.2f})",
                "exit_type": "TARGET",
                "exit_action": "SELL",
            }
        if action == "SELL" and target > 0 and ltp <= target:
            return {
                "symbol": symbol,
                "reason": f"Target hit at ₹{ltp:.2f} (Target: ₹{target:.2f})",
                "exit_type": "TARGET",
                "exit_action": "BUY",
            }

        # --- TIME-BASED EXIT (intraday) ---
        if data["holding_period"] == "INTRADAY":
            now = datetime.now(IST)
            if now.hour == 15 and now.minute >= 15:
                return {
                    "symbol": symbol,
                    "reason": f"Intraday time exit at 3:15 PM (LTP: ₹{ltp:.2f})",
                    "exit_type": "TIME_EXIT",
                    "exit_action": "SELL" if action == "BUY" else "BUY",
                }

        return None


# Singleton
stop_loss_manager = StopLossManager()
