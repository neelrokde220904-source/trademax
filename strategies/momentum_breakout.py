"""Momentum Breakout strategy — 52-week high breakout with volume confirmation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".numba_cache"))

import pandas as pd
import pandas_ta as ta
from loguru import logger


class MomentumBreakout:
    """52-week high breakout strategy for NSE stocks.

    ENTRY: Price breaks 52-week high with volume > 2× 20-day average, RSI 55-70
    EXIT:  Trailing stop at 3× ATR or target 2× risk
    TIMEFRAME: Daily candles
    UNIVERSE: Nifty 200 (broad liquid set)
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.name = "MomentumBreakout"
        p = params or {}
        self.volume_multiplier = p.get("volume_multiplier", 2.0)
        self.rsi_low = p.get("rsi_low", 55)
        self.rsi_high = p.get("rsi_high", 70)
        self.rsi_length = p.get("rsi_length", 14)
        self.lookback_52w = p.get("lookback_52w", 252)
        self.trailing_atr_mult = p.get("trailing_atr_mult", 3.0)
        self.rr_ratio = p.get("rr_ratio", 2.0)

    def scan(self, market_data: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        """Scan all symbols for momentum breakout signals.

        Args:
            market_data: {symbol: DataFrame of daily OHLCV with >= 252 rows}

        Returns:
            List of trade recommendations.
        """
        trades: list[dict[str, Any]] = []

        for symbol, df in market_data.items():
            try:
                signal = self._analyse(symbol, df)
                if signal:
                    trades.append(signal)
            except Exception as exc:
                logger.error("[{}] {} analysis failed: {}", self.name, symbol, exc)

        logger.info("[{}] Scan complete: {} signals from {} symbols.", self.name, len(trades), len(market_data))
        return trades

    def _analyse(self, symbol: str, df: pd.DataFrame) -> dict[str, Any] | None:
        """Analyse a single symbol for 52-week high breakout."""
        if df.empty or len(df) < self.lookback_52w:
            return None

        df = df.copy()

        # Compute indicators
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_length)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["vol_sma_20"] = df["volume"].rolling(20).mean()
        df["high_52w"] = df["high"].rolling(self.lookback_52w).max()
        df["prev_high_52w"] = df["high_52w"].shift(1)

        latest = df.iloc[-1]
        close = float(latest["close"])
        high = float(latest["high"])
        rsi = latest.get("rsi")
        atr = latest.get("atr")
        volume = float(latest["volume"])
        vol_avg = latest.get("vol_sma_20")
        prev_52w_high = latest.get("prev_high_52w")

        if any(pd.isna(v) for v in [rsi, atr, vol_avg, prev_52w_high]):
            return None

        rsi = float(rsi)
        atr = float(atr)
        vol_avg = float(vol_avg)
        prev_52w_high = float(prev_52w_high)

        # === BREAKOUT CONDITIONS ===
        # 1. Price broke above previous 52-week high
        is_breakout = high > prev_52w_high

        # 2. Volume confirmation > 2× average
        is_volume_confirm = vol_avg > 0 and volume > self.volume_multiplier * vol_avg

        # 3. RSI in sweet spot (not overbought, not weak)
        is_rsi_ok = self.rsi_low <= rsi <= self.rsi_high

        if not (is_breakout and is_volume_confirm and is_rsi_ok):
            return None

        # Stop-loss at 3× ATR below entry
        stop_loss = close - self.trailing_atr_mult * atr
        risk = close - stop_loss
        target = close + risk * self.rr_ratio

        # Confidence based on volume surge and RSI position
        vol_surge = volume / vol_avg if vol_avg else 1
        confidence = min(90, int(55 + vol_surge * 5 + (rsi - 55)))

        return {
            "symbol": symbol,
            "action": "BUY",
            "price": close,
            "stop_loss": round(stop_loss, 2),
            "target": round(target, 2),
            "order_type": "LIMIT",
            "holding_period": "SWING",
            "strategy": self.name,
            "confidence": confidence,
            "reasoning": (
                f"52-week high breakout! Close={close:.2f} > prev 52W high={prev_52w_high:.2f}. "
                f"Volume {vol_surge:.1f}× avg. RSI={rsi:.1f}. "
                f"Trailing SL at 3×ATR: ₹{stop_loss:.2f}, Target: ₹{target:.2f}"
            ),
        }
