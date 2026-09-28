"""Global Instrument Cache — maps US/EU equity tickers to IBKR + Alpaca contract IDs.

Provides a unified interface for looking up instruments across brokers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "global_instruments.json"


# Default global instruments — commonly traded US equities/ETFs
DEFAULT_GLOBAL_INSTRUMENTS: dict[str, dict[str, Any]] = {
    # Major US indices ETFs
    "SPY": {"name": "SPDR S&P 500 ETF", "exchange": "ARCA", "ibkr_conid": 756733, "type": "ETF", "currency": "USD"},
    "QQQ": {"name": "Invesco QQQ Trust", "exchange": "NASDAQ", "ibkr_conid": 320227571, "type": "ETF", "currency": "USD"},
    "DIA": {"name": "SPDR Dow Jones ETF", "exchange": "ARCA", "ibkr_conid": 4905372, "type": "ETF", "currency": "USD"},
    "IWM": {"name": "iShares Russell 2000 ETF", "exchange": "ARCA", "ibkr_conid": 9579970, "type": "ETF", "currency": "USD"},
    "VTI": {"name": "Vanguard Total Stock Market ETF", "exchange": "ARCA", "ibkr_conid": 46631572, "type": "ETF", "currency": "USD"},
    # FAANG+
    "AAPL": {"name": "Apple Inc", "exchange": "NASDAQ", "ibkr_conid": 265598, "type": "EQUITY", "currency": "USD"},
    "MSFT": {"name": "Microsoft Corp", "exchange": "NASDAQ", "ibkr_conid": 272093, "type": "EQUITY", "currency": "USD"},
    "GOOGL": {"name": "Alphabet Inc", "exchange": "NASDAQ", "ibkr_conid": 208813720, "type": "EQUITY", "currency": "USD"},
    "AMZN": {"name": "Amazon.com Inc", "exchange": "NASDAQ", "ibkr_conid": 3691937, "type": "EQUITY", "currency": "USD"},
    "NVDA": {"name": "NVIDIA Corp", "exchange": "NASDAQ", "ibkr_conid": 4815747, "type": "EQUITY", "currency": "USD"},
    "META": {"name": "Meta Platforms Inc", "exchange": "NASDAQ", "ibkr_conid": 107113386, "type": "EQUITY", "currency": "USD"},
    "TSLA": {"name": "Tesla Inc", "exchange": "NASDAQ", "ibkr_conid": 76792991, "type": "EQUITY", "currency": "USD"},
    # India-linked ADRs
    "INFY": {"name": "Infosys ADR", "exchange": "NYSE", "ibkr_conid": 11375034, "type": "ADR", "currency": "USD"},
    "WIT": {"name": "Wipro ADR", "exchange": "NYSE", "ibkr_conid": 4718656, "type": "ADR", "currency": "USD"},
    "IBN": {"name": "ICICI Bank ADR", "exchange": "NYSE", "ibkr_conid": 4718655, "type": "ADR", "currency": "USD"},
    "HDB": {"name": "HDFC Bank ADR", "exchange": "NYSE", "ibkr_conid": 12325, "type": "ADR", "currency": "USD"},
    # Sector ETFs
    "XLF": {"name": "Financial Select SPDR", "exchange": "ARCA", "ibkr_conid": 8312370, "type": "ETF", "currency": "USD"},
    "XLE": {"name": "Energy Select SPDR", "exchange": "ARCA", "ibkr_conid": 8312362, "type": "ETF", "currency": "USD"},
    "XLK": {"name": "Technology Select SPDR", "exchange": "ARCA", "ibkr_conid": 8312377, "type": "ETF", "currency": "USD"},
    "XLV": {"name": "Health Care Select SPDR", "exchange": "ARCA", "ibkr_conid": 8312381, "type": "ETF", "currency": "USD"},
    # Emerging market / India ETFs
    "INDA": {"name": "iShares MSCI India ETF", "exchange": "ARCA", "ibkr_conid": 109894571, "type": "ETF", "currency": "USD"},
    "EPI": {"name": "WisdomTree India Earnings", "exchange": "ARCA", "ibkr_conid": 61410898, "type": "ETF", "currency": "USD"},
}


class GlobalInstrumentCache:
    """Cache for global instrument lookups across brokers (IBKR + Alpaca)."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached instruments from disk, falling back to defaults."""
        if CACHE_PATH.exists():
            with open(CACHE_PATH) as f:
                self._cache = json.load(f)
            logger.info("Loaded {} global instruments from cache", len(self._cache))
        else:
            self._cache = DEFAULT_GLOBAL_INSTRUMENTS.copy()
            self._save_cache()
            logger.info("Initialized global instrument cache with {} defaults", len(self._cache))

    def _save_cache(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(self._cache, f, indent=2)

    def lookup(self, ticker: str) -> dict[str, Any] | None:
        """Look up a global instrument by ticker symbol."""
        return self._cache.get(ticker.upper())

    def get_ibkr_conid(self, ticker: str) -> int | None:
        instrument = self.lookup(ticker)
        return instrument.get("ibkr_conid") if instrument else None

    def get_exchange(self, ticker: str) -> str | None:
        instrument = self.lookup(ticker)
        return instrument.get("exchange") if instrument else None

    def add_instrument(self, ticker: str, info: dict[str, Any]) -> None:
        self._cache[ticker.upper()] = info
        self._save_cache()

    def list_instruments(self, instrument_type: str | None = None) -> list[str]:
        if instrument_type:
            return [k for k, v in self._cache.items() if v.get("type") == instrument_type.upper()]
        return list(self._cache.keys())

    def is_fema_compliant(self, ticker: str) -> bool:
        """Check if an instrument is FEMA-compliant (delivery equity/ETF only)."""
        instrument = self.lookup(ticker)
        if not instrument:
            return False
        return instrument.get("type") in ("EQUITY", "ETF", "ADR")


# Singleton
global_instrument_cache = GlobalInstrumentCache()
