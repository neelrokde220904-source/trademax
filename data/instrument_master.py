"""Instrument Master — downloads and caches Angel One instrument master JSON."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytz
from loguru import logger

from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")


async def download_instrument_master() -> list[dict[str, Any]]:
    """Download the full instrument master from Angel One and cache locally.

    Returns:
        List of instrument dicts.
    """
    cache_path = Path(settings.instrument_cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading instrument master from Angel One...")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(settings.instrument_master_url)
            resp.raise_for_status()
            data: list[dict[str, Any]] = resp.json()

        with open(cache_path, "w") as fh:
            json.dump(data, fh)

        logger.info("Instrument master cached: {} instruments at {}", len(data), cache_path)
        return data

    except Exception as exc:
        logger.exception("Failed to download instrument master: {}", exc)
        # Fall back to cached version
        if cache_path.exists():
            logger.info("Using cached instrument master.")
            with open(cache_path, "r") as fh:
                return json.load(fh)
        return []


def load_instrument_master() -> list[dict[str, Any]]:
    """Load cached instrument master (synchronous)."""
    cache_path = Path(settings.instrument_cache_path)
    if not cache_path.exists():
        logger.warning("No cached instrument master found. Run download_instrument_master() first.")
        return []
    with open(cache_path, "r") as fh:
        return json.load(fh)


def build_symbol_map(instruments: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """Build a lookup dict: symbol → instrument details.

    Supports lookup by 'NSE:RELIANCE' or 'NFO:NIFTY28APR...' format.
    """
    if instruments is None:
        instruments = load_instrument_master()

    symbol_map: dict[str, dict[str, Any]] = {}
    for item in instruments:
        sym = item.get("symbol", "")
        exch = item.get("exch_seg", "")
        key = f"{exch}:{sym}"
        symbol_map[key] = {
            "token": item.get("token", ""),
            "symbol": sym,
            "name": item.get("name", ""),
            "exchange": exch,
            "lot_size": int(item.get("lotsize", 1)),
            "tick_size": float(item.get("tick_size", 0.05)),
            "instrument_type": item.get("instrumenttype", ""),
            "expiry": item.get("expiry", ""),
            "strike": item.get("strike", ""),
            "option_type": item.get("optiontype", ""),
        }

    return symbol_map


def find_nfo_options(
    underlying: str,
    expiry_contains: str = "",
    option_type: str = "",
    instruments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Find NFO options contracts matching criteria.

    Args:
        underlying: e.g. 'NIFTY', 'BANKNIFTY'
        expiry_contains: partial match on expiry string
        option_type: 'CE' or 'PE'
    """
    if instruments is None:
        instruments = load_instrument_master()

    results = []
    for item in instruments:
        if item.get("exch_seg") != "NFO":
            continue
        if item.get("name", "").upper() != underlying.upper():
            continue
        if expiry_contains and expiry_contains not in item.get("expiry", ""):
            continue
        if option_type and item.get("optiontype", "").upper() != option_type.upper():
            continue
        results.append({
            "token": item.get("token", ""),
            "symbol": item.get("symbol", ""),
            "name": item.get("name", ""),
            "expiry": item.get("expiry", ""),
            "strike": float(item.get("strike", 0)) / 100.0,  # Angel stores strike × 100
            "option_type": item.get("optiontype", ""),
            "lot_size": int(item.get("lotsize", 1)),
        })

    return sorted(results, key=lambda x: x["strike"])
