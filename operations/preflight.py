"""Fail-closed operational readiness checks.

Run with ``python3 main.py --preflight`` before a scheduled session.  The checker
does not authenticate, submit orders, or modify state, so it is safe in all modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.settings import settings


@dataclass(frozen=True)
class PreflightResult:
    """A machine- and human-readable preflight result."""

    passed: bool
    errors: list[str]
    warnings: list[str]


def run_preflight() -> PreflightResult:
    """Validate safe configuration and local prerequisites without calling brokers."""
    errors: list[str] = []
    warnings: list[str] = []

    if settings.initial_capital <= 0:
        errors.append("INITIAL_CAPITAL must be greater than zero.")
    if not 0 < settings.max_capital_per_trade_pct <= 0.10:
        errors.append("MAX_CAPITAL_PER_TRADE_PCT must be in (0, 0.10].")
    if settings.max_positions < 1:
        errors.append("MAX_POSITIONS must be at least 1.")
    if not 0 < settings.max_daily_loss_pct <= 0.03:
        errors.append("MAX_DAILY_LOSS_PCT must be in (0, 0.03].")
    if settings.max_nifty_vix >= settings.vix_halt_threshold:
        errors.append("MAX_NIFTY_VIX must be lower than VIX_HALT_THRESHOLD.")

    cache_path = Path(settings.instrument_cache_path)
    if not cache_path.exists():
        warnings.append("Instrument cache is absent; the 08:00 pre-market job must download it.")
    if not settings.paper_trading:
        required = {
            "ANGEL_API_KEY": settings.angel_api_key,
            "ANGEL_CLIENT_ID": settings.angel_client_id,
            "ANGEL_MPIN": settings.angel_mpin,
            "ANGEL_TOTP_SECRET": settings.angel_totp_secret,
        }
        missing = [name for name, value in required.items() if not value or value.startswith("your_")]
        if missing:
            errors.append(f"Live trading requires configured broker credentials: {', '.join(missing)}.")
    elif not settings.angel_api_key or settings.angel_api_key.startswith("your_"):
        warnings.append("No Angel One credentials: paper fills will use requested prices when LTP is unavailable.")

    return PreflightResult(not errors, errors, warnings)
