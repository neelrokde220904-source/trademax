"""L2 Order Flow Imbalance Analyzer — detects institutional order flow from Angel One depth data.

Uses SmartWebSocketV2 FULL mode (best 5 buy/sell) to calculate Order Flow Imbalance (OFI).
OFI is a leading indicator — institutions accumulate before price moves.

Research basis: Kyle (1985) — informed traders leave footprints in order flow.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from loguru import logger


class L2OrderFlowAnalyzer:
    """Calculates Order Flow Imbalance (OFI) from Angel One depth data.

    depth_data structure from SmartWebSocketV2 FULL mode:
    {
        "best_5_buy_data": [{"price": x, "quantity": y, "orders": z}, ...],
        "best_5_sell_data": [{"price": x, "quantity": y, "orders": z}, ...]
    }
    """

    def __init__(self, rolling_window: int = 20) -> None:
        self.rolling_window = rolling_window
        # Per-symbol OFI history for rolling calculations
        self._ofi_history: dict[str, list[float]] = defaultdict(list)
        # Previous depth snapshot for absorption detection
        self._prev_depth: dict[str, dict[str, Any]] = {}
        # Per-symbol tick history for absorption
        self._tick_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def calculate_ofi(self, depth_data: dict[str, Any]) -> float:
        """Calculate Order Flow Imbalance from L2 depth data.

        OFI = (Buy Volume at Bid) - (Sell Volume at Ask) / Total
        Range: -1.0 (heavy selling) to +1.0 (heavy buying)

        Focus on top 3 levels for more relevant signal.
        """
        buy_data = depth_data.get("best_5_buy_data", [])
        sell_data = depth_data.get("best_5_sell_data", [])

        if not buy_data or not sell_data:
            return 0.0

        # Weight by proximity: top levels matter more
        buy_pressure = sum(
            level.get("quantity", 0) * level.get("orders", 1)
            for level in buy_data[:3]
        )
        sell_pressure = sum(
            level.get("quantity", 0) * level.get("orders", 1)
            for level in sell_data[:3]
        )

        total = buy_pressure + sell_pressure
        if total == 0:
            return 0.0

        ofi = (buy_pressure - sell_pressure) / total
        return float(ofi)

    def update_and_signal(self, symbol: str, depth_data: dict[str, Any],
                          tick_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Update OFI history for a symbol and return current signal.

        Args:
            symbol: Stock symbol
            depth_data: L2 depth from SmartWebSocketV2
            tick_data: Optional LTP tick data for absorption detection

        Returns:
            {
                "ofi": float,          # Current OFI value
                "signal": str,         # INSTITUTIONAL_BUYING / SELLING / NEUTRAL
                "zscore": float,       # Z-score of current OFI vs history
                "absorption": bool,    # Absorption pattern detected
                "interpretation": str, # Human-readable
            }
        """
        ofi = self.calculate_ofi(depth_data)
        self._ofi_history[symbol].append(ofi)

        # Keep bounded history
        max_history = self.rolling_window * 5
        if len(self._ofi_history[symbol]) > max_history:
            self._ofi_history[symbol] = self._ofi_history[symbol][-max_history:]

        # Update tick history
        if tick_data:
            self._tick_history[symbol].append(tick_data)
            if len(self._tick_history[symbol]) > 100:
                self._tick_history[symbol] = self._tick_history[symbol][-100:]

        # Calculate rolling signal
        signal_data = self.rolling_ofi_signal(symbol)

        # Check for absorption pattern
        absorption = self.detect_absorption(symbol, depth_data)
        signal_data["absorption"] = absorption

        if absorption:
            signal_data["interpretation"] += " [ABSORPTION DETECTED: institutional accumulation]"

        # Store current depth for next absorption check
        self._prev_depth[symbol] = depth_data

        return signal_data

    def rolling_ofi_signal(self, symbol: str) -> dict[str, Any]:
        """Normalize OFI over rolling window for cross-asset comparison.

        Signal interpretation:
          OFI_zscore > 2.0:  Strong institutional buying → BUY signal
          OFI_zscore < -2.0: Strong institutional selling → SELL signal
        """
        history = self._ofi_history.get(symbol, [])

        if len(history) < self.rolling_window:
            return {
                "ofi": history[-1] if history else 0.0,
                "signal": "INSUFFICIENT_DATA",
                "zscore": 0.0,
                "interpretation": f"Need {self.rolling_window - len(history)} more data points",
            }

        recent = history[-self.rolling_window:]
        mean_ofi = np.mean(recent)
        std_ofi = np.std(recent)

        if std_ofi < 1e-8:
            return {
                "ofi": history[-1],
                "signal": "NEUTRAL",
                "zscore": 0.0,
                "interpretation": "No significant order flow variation",
            }

        current_zscore = (history[-1] - mean_ofi) / std_ofi

        if current_zscore > 2.0:
            signal = "INSTITUTIONAL_BUYING"
            interp = "Smart money entering — strong buy-side imbalance"
        elif current_zscore < -2.0:
            signal = "INSTITUTIONAL_SELLING"
            interp = "Smart money exiting — strong sell-side imbalance"
        else:
            signal = "NEUTRAL"
            interp = "No unusual order flow"

        return {
            "ofi": float(history[-1]),
            "signal": signal,
            "zscore": float(current_zscore),
            "interpretation": interp,
        }

    def detect_absorption(self, symbol: str, current_depth: dict[str, Any]) -> bool:
        """Detect absorption pattern: price drops but sell orders are being absorbed.

        Classic institutional accumulation pattern.
        Returns True if absorption detected.
        """
        prev_depth = self._prev_depth.get(symbol)
        ticks = self._tick_history.get(symbol, [])

        if not prev_depth or len(ticks) < 5:
            return False

        # Price has been declining
        try:
            price_down = ticks[-1].get("ltp", 0) < ticks[-5].get("ltp", 0)
        except (IndexError, TypeError):
            return False

        if not price_down:
            return False

        # But sell-side size is reducing (orders being absorbed)
        current_sell = sum(
            l.get("quantity", 0) for l in current_depth.get("best_5_sell_data", [])
        )
        prev_sell = sum(
            l.get("quantity", 0) for l in prev_depth.get("best_5_sell_data", [])
        )

        sell_reducing = prev_sell > 0 and current_sell < prev_sell * 0.85

        if sell_reducing:
            logger.info(
                "[{}] Absorption detected: price down but sell pressure reducing "
                "(prev_sell={}, current_sell={})",
                symbol, prev_sell, current_sell,
            )

        return sell_reducing

    def get_all_signals(self) -> dict[str, dict[str, Any]]:
        """Get OFI signals for all tracked symbols."""
        signals = {}
        for symbol in self._ofi_history:
            signals[symbol] = self.rolling_ofi_signal(symbol)
            signals[symbol]["absorption"] = False  # Would need current depth to check
        return signals

    def reset(self, symbol: str | None = None) -> None:
        """Reset history for a symbol or all symbols."""
        if symbol:
            self._ofi_history.pop(symbol, None)
            self._prev_depth.pop(symbol, None)
            self._tick_history.pop(symbol, None)
        else:
            self._ofi_history.clear()
            self._prev_depth.clear()
            self._tick_history.clear()


# Singleton
l2_analyzer = L2OrderFlowAnalyzer()
