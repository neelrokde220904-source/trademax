"""NSE/BSE symbol → Angel One token mapping cache utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import settings

_CACHE: dict[str, dict[str, Any]] = {}


def load_instrument_cache() -> dict[str, dict[str, Any]]:
    """Load the cached instrument master JSON into a symbol → details dict."""
    global _CACHE
    if _CACHE:
        return _CACHE

    cache_path = Path(settings.instrument_cache_path)
    if not cache_path.exists():
        logger.warning("Instrument cache not found at {}. Run instrument_master.download() first.", cache_path)
        return {}

    with open(cache_path, "r") as fh:
        raw: list[dict[str, Any]] = json.load(fh)

    for item in raw:
        symbol = item.get("symbol", "")
        exchange = item.get("exch_seg", "")
        key = f"{exchange}:{symbol}" if exchange else symbol
        _CACHE[key] = {
            "token": item.get("token", ""),
            "symbol": symbol,
            "name": item.get("name", ""),
            "exchange": exchange,
            "lot_size": int(item.get("lotsize", 1)),
            "tick_size": float(item.get("tick_size", 0.05)),
            "instrument_type": item.get("instrumenttype", ""),
            "expiry": item.get("expiry", ""),
            "strike": item.get("strike", ""),
            "option_type": item.get("optiontype", ""),
        }

    logger.info("Loaded {} instruments into cache.", len(_CACHE))
    return _CACHE


def get_token(symbol: str, exchange: str = "NSE") -> str | None:
    """Return the Angel One token for a given symbol and exchange."""
    cache = load_instrument_cache()
    key = f"{exchange}:{symbol}"
    entry = cache.get(key)
    if entry:
        return entry["token"]
    # Fallback: search by symbol only
    for k, v in cache.items():
        if v["symbol"] == symbol and v["exchange"] == exchange:
            return v["token"]
    return None


def get_instrument(symbol: str, exchange: str = "NSE") -> dict[str, Any] | None:
    """Return full instrument details for a given symbol."""
    cache = load_instrument_cache()
    key = f"{exchange}:{symbol}"
    return cache.get(key)


def invalidate_cache() -> None:
    """Clear the in-memory cache (e.g. after daily refresh)."""
    global _CACHE
    _CACHE = {}
