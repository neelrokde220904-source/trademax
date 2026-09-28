"""Tests for data/indicators.py."""

import pandas as pd
import numpy as np

from data.indicators import compute_all_indicators, get_latest_signals


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.2,
        "high": close + abs(np.random.randn(n) * 0.5),
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, n),
    })


def test_compute_all_indicators():
    """compute_all_indicators should add expected columns."""
    df = _make_ohlcv(200)
    result = compute_all_indicators(df)

    assert "rsi" in result.columns
    assert "macd" in result.columns
    assert "ema_9" in result.columns
    assert "ema_21" in result.columns
    assert "atr" in result.columns

    # Should not have all NaN
    assert result["rsi"].dropna().shape[0] > 0


def test_get_latest_signals():
    """get_latest_signals should return a valid signal dict."""
    df = _make_ohlcv(200)
    df = compute_all_indicators(df)
    signals = get_latest_signals(df)

    assert "composite_signal" in signals
    assert signals["composite_signal"] in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    assert "rsi" in signals
    assert "macd" in signals
