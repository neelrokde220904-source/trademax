"""Risk Engine — hard-limit enforcement and pre-trade checks for India markets."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytz
from loguru import logger

from broker.portfolio_tracker import portfolio_tracker
from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")


class RiskEngine:
    """Centralised pre-trade risk gatekeeper.

    Enforces:
    - Daily loss limit (3%)
    - Max simultaneous positions (8)
    - VIX halt threshold (28)
    - Market hours window (no first/last 15 min)
    - NSE holiday check
    - Per-trade capital limit (10%)
    - F&O ban-list filter
    """

    def __init__(self) -> None:
        self._fno_ban_list: list[str] = []
        self._current_vix: float = 0.0

    def update_vix(self, vix: float) -> None:
        self._current_vix = vix

    def update_fno_ban(self, ban_list: list[str]) -> None:
        self._fno_ban_list = list(ban_list)

    # ----- PRE-TRADE CHECKS -----

    def pre_trade_check(self, signal: dict[str, Any]) -> tuple[bool, str]:
        """Run all hard-limit checks before placing a trade.

        Returns:
            (allowed: bool, reason: str)
        """
        checks = [
            self._check_market_open,
            self._check_nse_holiday,
            self._check_time_window,
            self._check_daily_loss,
            self._check_max_positions,
            self._check_vix,
            self._check_capital_per_trade,
            self._check_fno_ban,
        ]

        for check in checks:
            allowed, reason = check(signal)
            if not allowed:
                logger.warning("[RiskEngine] BLOCKED: {} — {}", signal.get("symbol", "?"), reason)
                return False, reason

        return True, "All risk checks passed"

    def _check_market_open(self, signal: dict) -> tuple[bool, str]:
        now = datetime.now(IST)
        hour, minute = now.hour, now.minute
        if now.weekday() >= 5:
            return False, "Weekend — market closed"
        market_open = (hour == 9 and minute >= 15) or (hour > 9)
        market_close = hour < 15 or (hour == 15 and minute <= 30)
        if not (market_open and market_close):
            return False, f"Outside market hours ({hour:02d}:{minute:02d})"
        return True, ""

    def _check_nse_holiday(self, signal: dict) -> tuple[bool, str]:
        today = datetime.now(IST).strftime("%Y-%m-%d")
        if today in settings.nse_holidays:
            return False, f"NSE holiday: {today}"
        return True, ""

    def _check_time_window(self, signal: dict) -> tuple[bool, str]:
        now = datetime.now(IST)
        market_open_9_15 = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_3_30 = now.replace(hour=15, minute=30, second=0, microsecond=0)

        from datetime import timedelta

        # No trades in first N minutes after open
        no_trade_start = market_open_9_15 + timedelta(minutes=settings.no_trade_window_open_minutes)
        if now < no_trade_start:
            return False, f"Within first {settings.no_trade_window_open_minutes} min of market open"

        # No trades in last N minutes before close
        no_trade_end = market_close_3_30 - timedelta(minutes=settings.no_trade_window_close_minutes)
        if now > no_trade_end:
            return False, f"Within last {settings.no_trade_window_close_minutes} min of market close"

        return True, ""

    def _check_daily_loss(self, signal: dict) -> tuple[bool, str]:
        if portfolio_tracker.has_hit_daily_loss_limit():
            return False, f"Daily loss limit hit ({settings.max_daily_loss_pct*100:.1f}%)"
        return True, ""

    def _check_max_positions(self, signal: dict) -> tuple[bool, str]:
        if portfolio_tracker.position_count >= settings.max_positions:
            return False, f"Max {settings.max_positions} positions reached"
        return True, ""

    def _check_vix(self, signal: dict) -> tuple[bool, str]:
        if self._current_vix > settings.vix_halt_threshold:
            return False, f"VIX {self._current_vix:.1f} > halt threshold {settings.vix_halt_threshold}"
        return True, ""

    def _check_capital_per_trade(self, signal: dict) -> tuple[bool, str]:
        price = signal.get("price", 0)
        qty = signal.get("quantity", 0)
        trade_value = price * qty
        max_trade = settings.initial_capital * settings.max_capital_per_trade_pct
        if trade_value > max_trade:
            return False, f"Trade value ₹{trade_value:.0f} > max ₹{max_trade:.0f} ({settings.max_capital_per_trade_pct*100:.0f}%)"
        return True, ""

    def _check_fno_ban(self, signal: dict) -> tuple[bool, str]:
        symbol = signal.get("symbol", "")
        if symbol in self._fno_ban_list:
            return False, f"{symbol} is in F&O ban list"
        return True, ""


# Singleton
risk_engine = RiskEngine()
