"""Mean Reversion strategy — Bollinger Band squeeze/bounce on Nifty 50 stocks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".numba_cache"))

import pandas as pd
import pandas_ta as ta
from loguru import logger


class MeanReversion:
    """Bollinger Band mean-reversion strategy for NSE stocks.

    ENTRY LONG:  Price touches/breaks lower BB + RSI < 30 + volume spike
    ENTRY SHORT: Price touches/breaks upper BB + RSI > 70 + volume spike
    EXIT:  Middle band (SMA 20) or stop-loss
    TIMEFRAME: Daily candles
    UNIVERSE: Nifty 50 (high-liquidity large caps for mean reversion)
    STOP LOSS: 1.5× ATR beyond entry
    TARGET: Middle Bollinger Band (SMA 20)
    """

    def __init__(self) -> None:
        self.name = "MeanReversion"
        self.bb_length = 20
        self.bb_std = 2.0
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.rsi_length = 14
        self.vol_multiplier = 1.5  # volume > 1.5× average
        self.sl_atr_mult = 1.5

    def scan(self, market_data: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        """Scan all symbols for mean-reversion signals.

        Args:
            market_data: {symbol: DataFrame of daily OHLCV}

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
        """Analyse a single symbol for Bollinger Band mean reversion."""
        if df.empty or len(df) < 30:
            return None

        df = df.copy()

        # Compute Bollinger Bands
        bbands = ta.bbands(df["close"], length=self.bb_length, std=self.bb_std)
        if bbands is None:
            return None
        df = pd.concat([df, bbands], axis=1)

        # Compute RSI and ATR
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_length)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["vol_sma_20"] = df["volume"].rolling(20).mean()

        latest = df.iloc[-1]
        close = float(latest["close"])
        rsi = latest.get("rsi")
        atr = latest.get("atr")
        volume = float(latest["volume"])
        vol_avg = latest.get("vol_sma_20")

        # Bollinger Band columns
        bbl_col = f"BBL_{self.bb_length}_{self.bb_std}"
        bbm_col = f"BBM_{self.bb_length}_{self.bb_std}"
        bbu_col = f"BBU_{self.bb_length}_{self.bb_std}"

        bb_lower = latest.get(bbl_col)
        bb_middle = latest.get(bbm_col)
        bb_upper = latest.get(bbu_col)

        if any(pd.isna(v) for v in [rsi, atr, vol_avg, bb_lower, bb_middle, bb_upper]):
            return None

        rsi = float(rsi)
        atr = float(atr)
        vol_avg = float(vol_avg)
        bb_lower = float(bb_lower)
        bb_middle = float(bb_middle)
        bb_upper = float(bb_upper)

        has_volume = vol_avg > 0 and volume > self.vol_multiplier * vol_avg

        # === LONG: Price at/below lower BB + RSI oversold + volume ===
        if close <= bb_lower and rsi < self.rsi_oversold and has_volume:
            stop_loss = close - self.sl_atr_mult * atr
            target = bb_middle  # mean reversion to middle band
            risk = close - stop_loss
            reward = target - close

            if risk <= 0 or reward <= 0:
                return None

            return {
                "symbol": symbol,
                "action": "BUY",
                "price": close,
                "stop_loss": round(stop_loss, 2),
                "target": round(target, 2),
                "order_type": "LIMIT",
                "holding_period": "SWING",
                "strategy": self.name,
                "confidence": min(85, int(50 + (self.rsi_oversold - rsi) * 2)),
                "reasoning": (
                    f"Mean reversion BUY: Close ₹{close:.2f} at/below lower BB ₹{bb_lower:.2f}. "
                    f"RSI={rsi:.1f} (oversold). Target: middle BB ₹{bb_middle:.2f}. "
                    f"RR={reward / risk:.1f}:1"
                ),
            }

        # === SHORT: Price at/above upper BB + RSI overbought + volume ===
        if close >= bb_upper and rsi > self.rsi_overbought and has_volume:
            stop_loss = close + self.sl_atr_mult * atr
            target = bb_middle  # mean reversion to middle band
            risk = stop_loss - close
            reward = close - target

            if risk <= 0 or reward <= 0:
                return None

            return {
                "symbol": symbol,
                "action": "SELL",
                "price": close,
                "stop_loss": round(stop_loss, 2),
                "target": round(target, 2),
                "order_type": "LIMIT",
                "holding_period": "SWING",
                "strategy": self.name,
                "confidence": min(85, int(50 + (rsi - self.rsi_overbought) * 2)),
                "reasoning": (
                    f"Mean reversion SELL: Close ₹{close:.2f} at/above upper BB ₹{bb_upper:.2f}. "
                    f"RSI={rsi:.1f} (overbought). Target: middle BB ₹{bb_middle:.2f}. "
                    f"RR={reward / risk:.1f}:1"
                ),
            }

        return None
