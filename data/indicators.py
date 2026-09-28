"""Technical indicators using pandas-ta."""

from __future__ import annotations

import os
from pathlib import Path

# pandas-ta uses Numba's on-disk cache.  Python 3.13 cannot infer a cache
# location for some installed pandas-ta builds, so provide a writable project
# location before importing it.  This affects compilation cache only, not data.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".numba_cache"))

import pandas as pd
import pandas_ta as ta
from loguru import logger


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all required technical indicators on an OHLCV DataFrame.

    Expects columns: datetime, open, high, low, close, volume.
    Returns the DataFrame with indicator columns appended.
    """
    if df.empty or len(df) < 30:
        logger.warning("Insufficient data ({} rows) for indicator calculation.", len(df))
        return df

    df = df.copy()

    # RSI (14)
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    # Backward-compatible aliases keep indicator consumers and historical
    # notebooks stable while the canonical names remain parameterised.
    df["rsi"] = df["rsi_14"]

    # MACD (12, 26, 9)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        df["macd"] = df["MACD_12_26_9"]

    # EMA (9, 21, 50, 200)
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)

    # Bollinger Bands (20, 2)
    bbands = ta.bbands(df["close"], length=20, std=2.0)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)

    # Supertrend (7, 3.0)
    st = ta.supertrend(df["high"], df["low"], df["close"], length=7, multiplier=3.0)
    if st is not None:
        df = pd.concat([df, st], axis=1)

    # ADX (14)
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx is not None:
        df = pd.concat([df, adx], axis=1)

    # VWAP (intraday)
    try:
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    except Exception:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3  # Simple approximation

    # ATR (14)
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr"] = df["atr_14"]

    # OBV (On Balance Volume)
    df["obv"] = ta.obv(df["close"], df["volume"])

    # Stochastic RSI
    stoch_rsi = ta.stochrsi(df["close"], length=14)
    if stoch_rsi is not None:
        df = pd.concat([df, stoch_rsi], axis=1)

    return df


def get_latest_signals(df: pd.DataFrame) -> dict:
    """Extract the latest indicator values and generate composite signals.

    Returns:
        Dict with latest values and a composite signal assessment.
    """
    if df.empty:
        return {"signal": "HOLD", "confidence": 0, "indicators": {}}

    latest = df.iloc[-1]

    indicators = {}

    # RSI
    rsi = latest.get("rsi_14")
    if pd.notna(rsi):
        indicators["rsi_14"] = round(float(rsi), 2)

    # MACD
    macd_val = latest.get("MACD_12_26_9")
    macd_signal = latest.get("MACDs_12_26_9")
    macd_hist = latest.get("MACDh_12_26_9")
    if pd.notna(macd_val):
        indicators["macd"] = round(float(macd_val), 2)
        indicators["macd_signal"] = round(float(macd_signal), 2) if pd.notna(macd_signal) else None
        indicators["macd_histogram"] = round(float(macd_hist), 2) if pd.notna(macd_hist) else None

    # EMAs
    for ema_col in ["ema_9", "ema_21", "ema_50", "ema_200"]:
        val = latest.get(ema_col)
        if pd.notna(val):
            indicators[ema_col] = round(float(val), 2)

    # Supertrend
    st_col = [c for c in df.columns if c.startswith("SUPERT_")]
    st_dir_col = [c for c in df.columns if c.startswith("SUPERTd_")]
    if st_col:
        indicators["supertrend"] = round(float(latest[st_col[0]]), 2)
    if st_dir_col:
        indicators["supertrend_direction"] = int(latest[st_dir_col[0]]) if pd.notna(latest[st_dir_col[0]]) else 0

    # ADX
    adx_val = latest.get("ADX_14")
    if pd.notna(adx_val):
        indicators["adx_14"] = round(float(adx_val), 2)

    # VWAP
    vwap = latest.get("vwap")
    if pd.notna(vwap):
        indicators["vwap"] = round(float(vwap), 2)

    # ATR
    atr = latest.get("atr_14")
    if pd.notna(atr):
        indicators["atr_14"] = round(float(atr), 2)

    # Bollinger Bands
    for bb_col in ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0"]:
        val = latest.get(bb_col)
        if pd.notna(val):
            indicators[bb_col.lower()] = round(float(val), 2)

    # Current price
    close = float(latest["close"])
    indicators["close"] = close

    # Compute composite signal
    signal, confidence, reasoning = _compute_composite_signal(indicators, close)

    return {
        "signal": signal,
        "composite_signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
        "indicators": indicators,
        # Compatibility fields are intentionally duplicated at the top level
        # for lightweight consumers; authoritative values remain in indicators.
        "rsi": indicators.get("rsi_14"),
        "macd": indicators.get("macd"),
    }


def _compute_composite_signal(indicators: dict, close: float) -> tuple[str, int, str]:
    """Generate a composite BUY/SELL/HOLD signal from indicator values."""
    bullish_count = 0
    bearish_count = 0
    reasons: list[str] = []

    rsi = indicators.get("rsi_14")
    if rsi is not None:
        if rsi < 35:
            bullish_count += 1
            reasons.append(f"RSI oversold ({rsi})")
        elif rsi > 70:
            bearish_count += 1
            reasons.append(f"RSI overbought ({rsi})")

    # VWAP
    vwap = indicators.get("vwap")
    if vwap is not None:
        if close > vwap:
            bullish_count += 1
            reasons.append("Price above VWAP")
        else:
            bearish_count += 1
            reasons.append("Price below VWAP")

    # MACD crossover
    macd_hist = indicators.get("macd_histogram")
    if macd_hist is not None:
        if macd_hist > 0:
            bullish_count += 1
            reasons.append("MACD bullish")
        else:
            bearish_count += 1
            reasons.append("MACD bearish")

    # Supertrend direction
    st_dir = indicators.get("supertrend_direction")
    if st_dir is not None:
        if st_dir == 1:
            bullish_count += 1
            reasons.append("Supertrend bullish")
        elif st_dir == -1:
            bearish_count += 1
            reasons.append("Supertrend bearish")

    # ADX strength
    adx = indicators.get("adx_14")
    if adx is not None and adx > 25:
        reasons.append(f"Strong trend (ADX={adx})")

    # EMA alignment
    ema9 = indicators.get("ema_9")
    ema21 = indicators.get("ema_21")
    if ema9 is not None and ema21 is not None:
        if ema9 > ema21:
            bullish_count += 1
            reasons.append("EMA9 > EMA21 (bullish)")
        else:
            bearish_count += 1
            reasons.append("EMA9 < EMA21 (bearish)")

    # Determine signal
    total = bullish_count + bearish_count
    if total == 0:
        return "HOLD", 0, "Insufficient indicator data"

    if bullish_count >= 4:
        signal = "STRONG_BUY"
        confidence = min(95, 60 + bullish_count * 8)
    elif bullish_count >= 3 and bearish_count <= 1:
        signal = "BUY"
        confidence = min(85, 50 + bullish_count * 7)
    elif bearish_count >= 4:
        signal = "STRONG_SELL"
        confidence = min(95, 60 + bearish_count * 8)
    elif bearish_count >= 3 and bullish_count <= 1:
        signal = "SELL"
        confidence = min(85, 50 + bearish_count * 7)
    else:
        signal = "HOLD"
        confidence = 30

    reasoning = " | ".join(reasons)
    return signal, confidence, reasoning
