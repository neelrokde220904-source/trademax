"""Supertrend + ADX strategy — directional trend-following on 15-min candles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".numba_cache"))

import pandas as pd
import pandas_ta as ta
from loguru import logger

from config.settings import settings


class SupertrendADX:
    """Supertrend + ADX strategy for NSE stocks.

    ENTRY: Supertrend flips to BUY + ADX > 25
    EXIT:  Supertrend flips to SELL OR stop-loss OR target
    TIMEFRAME: 15-minute candles
    UNIVERSE: Nifty 50 + Nifty Next 50 (liquid stocks)
    STOP LOSS: Below Supertrend line OR 1.5× ATR
    TARGET: 1.5× risk (min 1:1.5 RR)
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.name = "SupertrendADX"
        p = params or {}
        self.supertrend_length = p.get("supertrend_length", 7)
        self.supertrend_multiplier = p.get("supertrend_multiplier", 3.0)
        self.adx_length = p.get("adx_length", 14)
        self.adx_threshold = p.get("adx_threshold", 25)
        self.rr_ratio = p.get("rr_ratio", 1.5)

    def scan(self, market_data: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        """Scan all symbols for Supertrend + ADX signals.

        Args:
            market_data: {symbol: DataFrame of 15-min OHLCV}

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
        """Analyse a single symbol for Supertrend + ADX entry."""
        if df.empty or len(df) < 30:
            return None

        df = df.copy()

        # Compute Supertrend
        st = ta.supertrend(
            df["high"], df["low"], df["close"],
            length=self.supertrend_length,
            multiplier=self.supertrend_multiplier,
        )
        if st is None:
            return None
        df = pd.concat([df, st], axis=1)

        # Compute ADX
        adx = ta.adx(df["high"], df["low"], df["close"], length=self.adx_length)
        if adx is None:
            return None
        df = pd.concat([df, adx], axis=1)

        # Compute ATR for stop-loss
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # Get latest and previous values
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest

        # Find Supertrend direction columns
        st_dir_cols = [c for c in df.columns if c.startswith("SUPERTd_")]
        st_val_cols = [c for c in df.columns if c.startswith("SUPERT_") and not c.startswith("SUPERTd_") and not c.startswith("SUPERTl_") and not c.startswith("SUPERTs_")]

        if not st_dir_cols:
            return None

        st_dir_col = st_dir_cols[0]
        current_dir = latest.get(st_dir_col)
        prev_dir = prev.get(st_dir_col)
        adx_value = latest.get("ADX_14", 0)
        close = float(latest["close"])
        atr = float(latest.get("atr", 0))

        if pd.isna(current_dir) or pd.isna(adx_value):
            return None

        # Supertrend FLIP + ADX strong trend
        if current_dir == 1 and prev_dir == -1 and adx_value > self.adx_threshold:
            # BUY signal — Supertrend flipped bullish + strong trend
            stop_loss = close - max(atr * settings.stop_loss_atr_multiplier, 0)
            risk = close - stop_loss
            target = close + risk * self.rr_ratio

            # Use Supertrend value as alternative SL
            st_value = latest.get(st_val_cols[0], stop_loss) if st_val_cols else stop_loss
            if pd.notna(st_value):
                stop_loss = min(stop_loss, float(st_value))

            return {
                "symbol": symbol,
                "action": "BUY",
                "price": close,
                "stop_loss": round(stop_loss, 2),
                "target": round(target, 2),
                "order_type": "LIMIT",
                "holding_period": "INTRADAY",
                "strategy": self.name,
                "confidence": min(85, int(50 + adx_value)),
                "reasoning": (
                    f"Supertrend flipped BULLISH + ADX={adx_value:.1f} (strong trend). "
                    f"Risk:₹{risk:.2f} Reward:₹{risk * self.rr_ratio:.2f}"
                ),
            }

        elif current_dir == -1 and prev_dir == 1 and adx_value > self.adx_threshold:
            # SELL signal — Supertrend flipped bearish + strong trend
            stop_loss = close + max(atr * settings.stop_loss_atr_multiplier, 0)
            risk = stop_loss - close
            target = close - risk * self.rr_ratio

            return {
                "symbol": symbol,
                "action": "SELL",
                "price": close,
                "stop_loss": round(stop_loss, 2),
                "target": round(target, 2),
                "order_type": "LIMIT",
                "holding_period": "INTRADAY",
                "strategy": self.name,
                "confidence": min(85, int(50 + adx_value)),
                "reasoning": (
                    f"Supertrend flipped BEARISH + ADX={adx_value:.1f} (strong trend). "
                    f"Risk:₹{risk:.2f} Reward:₹{risk * self.rr_ratio:.2f}"
                ),
            }

        return None
