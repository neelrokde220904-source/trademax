"""Historical OHLCV fetcher — Angel One getCandleData + yfinance fallback."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytz
import yfinance as yf
from loguru import logger

from broker.angel_client import angel_client
from config.instruments import get_token
from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")

# Angel One interval constants
INTERVAL_MAP = {
    "1m": "ONE_MINUTE",
    "3m": "THREE_MINUTE",
    "5m": "FIVE_MINUTE",
    "10m": "TEN_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "1d": "ONE_DAY",
}

# Max candles per request (Angel One limit)
MAX_CANDLES = {
    "ONE_MINUTE": 400,
    "THREE_MINUTE": 400,
    "FIVE_MINUTE": 400,
    "TEN_MINUTE": 400,
    "FIFTEEN_MINUTE": 400,
    "THIRTY_MINUTE": 400,
    "ONE_HOUR": 400,
    "ONE_DAY": 2000,
}


def fetch_ohlcv(
    symbol: str,
    interval: str = "15m",
    days: int = 30,
    exchange: str = "NSE",
) -> pd.DataFrame:
    """Fetch historical OHLCV data for a symbol.

    Args:
        symbol: NSE trading symbol (e.g. 'RELIANCE')
        interval: '1m', '5m', '15m', '1h', '1d', etc.
        days: Number of days of history
        exchange: 'NSE' or 'NFO'

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume
    """
    angel_interval = INTERVAL_MAP.get(interval)
    if not angel_interval:
        logger.warning("Unknown interval '{}'. Falling back to yfinance.", interval)
        return _fetch_yfinance(symbol, interval, days)

    token = get_token(symbol, exchange)
    if not token:
        logger.warning("Token not found for {}:{}, falling back to yfinance.", exchange, symbol)
        return _fetch_yfinance(symbol, interval, days)

    try:
        now = datetime.now(IST)
        to_date = now.strftime("%Y-%m-%d %H:%M")
        from_date = (now - timedelta(days=days)).strftime("%Y-%m-%d 09:15")

        candles = angel_client.get_candle_data(
            exchange=exchange,
            symbol_token=token,
            interval=angel_interval,
            from_date=from_date,
            to_date=to_date,
        )

        if not candles:
            logger.warning("No candle data from Angel One for {}. Trying yfinance.", symbol)
            return _fetch_yfinance(symbol, interval, days)

        df = pd.DataFrame(candles, columns=["datetime", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(int)

        logger.info("Fetched {} candles for {} ({}).", len(df), symbol, interval)
        return df

    except Exception as exc:
        logger.exception("Angel One candle fetch failed for {}: {}. Trying yfinance.", symbol, exc)
        return _fetch_yfinance(symbol, interval, days)


def _fetch_yfinance(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fallback: fetch data from yfinance (NSE symbols need .NS suffix)."""
    try:
        yf_symbol = f"{symbol}.NS"
        yf_interval = interval if interval != "15m" else "15m"

        # yfinance limits: 1m=7 days, 5m/15m/30m=60 days, 1h=730 days, 1d=unlimited
        period_map = {
            "1m": min(days, 7),
            "5m": min(days, 60),
            "15m": min(days, 60),
            "30m": min(days, 60),
            "1h": min(days, 730),
            "1d": days,
        }
        actual_days = period_map.get(interval, days)

        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{actual_days}d", interval=yf_interval)

        if df.empty:
            logger.warning("yfinance returned no data for {}.", yf_symbol)
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        df = df.reset_index()
        # Normalise column names
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})

        df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        logger.info("[yfinance] Fetched {} candles for {}.", len(df), yf_symbol)
        return df

    except Exception as exc:
        logger.exception("yfinance fetch failed for {}: {}", symbol, exc)
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])


def fetch_multiple(
    symbols: list[str],
    interval: str = "15m",
    days: int = 30,
    exchange: str = "NSE",
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple symbols.

    Returns:
        Dict mapping symbol → DataFrame.
    """
    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        result[sym] = fetch_ohlcv(sym, interval=interval, days=days, exchange=exchange)
    return result


def fetch_index_data(index_token: str, interval: str = "15m", days: int = 30) -> pd.DataFrame:
    """Fetch index (Nifty50, BankNifty) OHLCV data."""
    try:
        now = datetime.now(IST)
        angel_interval = INTERVAL_MAP.get(interval, "FIFTEEN_MINUTE")
        candles = angel_client.get_candle_data(
            exchange="NSE",
            symbol_token=index_token,
            interval=angel_interval,
            from_date=(now - timedelta(days=days)).strftime("%Y-%m-%d 09:15"),
            to_date=now.strftime("%Y-%m-%d %H:%M"),
        )
        if candles:
            df = pd.DataFrame(candles, columns=["datetime", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df.sort_values("datetime").reset_index(drop=True)
    except Exception as exc:
        logger.error("Index data fetch failed for token {}: {}", index_token, exc)

    return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
