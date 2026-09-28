"""Portfolio Tracker — holdings, positions, PnL tracking."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytz
from loguru import logger

from broker.angel_client import angel_client
from config.settings import settings
from database.db import get_db
from database.models import DailyPnL, Portfolio, Trade

IST = pytz.timezone("Asia/Kolkata")


class PortfolioTracker:
    """Tracks current portfolio state, positions, and PnL."""

    def __init__(self) -> None:
        self._positions: dict[str, dict[str, Any]] = {}
        self._available_capital: float = settings.initial_capital
        self._total_value: float = settings.initial_capital
        self._realised_pnl: float = 0.0
        self._daily_pnl: float = 0.0
        self._trades_today: int = 0
        self._wins_today: int = 0
        self._losses_today: int = 0

    def sync_from_broker(self) -> None:
        """Sync positions and holdings from Angel One."""
        if settings.paper_trading:
            self._sync_paper_positions()
        else:
            self._sync_live_positions()

    def _sync_live_positions(self) -> None:
        """Sync from live Angel One API."""
        positions = angel_client.get_positions()
        holdings = angel_client.get_holdings()
        rms = angel_client.get_rms_limits()

        self._positions = {}

        for pos in positions:
            symbol = pos.get("tradingsymbol", "")
            net_qty = int(pos.get("netqty", 0))
            if net_qty != 0:
                self._positions[symbol] = {
                    "symbol": symbol,
                    "quantity": net_qty,
                    "avg_price": float(pos.get("averageprice", 0)),
                    "ltp": float(pos.get("ltp", 0)),
                    "pnl": float(pos.get("pnl", 0)),
                    "product": pos.get("producttype", ""),
                }

        if rms:
            self._available_capital = float(rms.get("availablecash", self._available_capital))

        logger.info("Synced {} live positions. Available capital: ₹{:.2f}", len(self._positions), self._available_capital)

    def _sync_paper_positions(self) -> None:
        """Sync from paper trade database records."""
        with get_db() as db:
            today_str = datetime.now(IST).strftime("%Y-%m-%d")
            trades = (
                db.query(Trade)
                .filter(Trade.is_paper == True, Trade.status == "EXECUTED", Trade.exit_price.is_(None))
                .all()
            )

            self._positions = {}
            invested = 0.0

            for t in trades:
                if t.symbol not in self._positions:
                    self._positions[t.symbol] = {
                        "symbol": t.symbol,
                        "quantity": 0,
                        "avg_price": 0.0,
                        "ltp": t.price,
                        "pnl": 0.0,
                        "product": t.holding_period,
                    }

                pos = self._positions[t.symbol]
                if t.action == "BUY":
                    total_cost = pos["avg_price"] * pos["quantity"] + t.price * t.quantity
                    pos["quantity"] += t.quantity
                    pos["avg_price"] = total_cost / pos["quantity"] if pos["quantity"] else 0
                elif t.action == "SELL":
                    pos["quantity"] -= t.quantity

                invested += pos["avg_price"] * abs(pos["quantity"])

            # Remove flat positions
            self._positions = {k: v for k, v in self._positions.items() if v["quantity"] != 0}
            self._available_capital = settings.initial_capital + self._realised_pnl - invested

    @property
    def positions(self) -> dict[str, dict[str, Any]]:
        return self._positions

    @property
    def available_capital(self) -> float:
        return self._available_capital

    @property
    def total_value(self) -> float:
        return self._total_value

    @property
    def position_count(self) -> int:
        return len(self._positions)

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def daily_loss_pct(self) -> float:
        """Current daily loss as a percentage of starting capital."""
        if self._total_value <= 0:
            return 0.0
        return -self._daily_pnl / settings.initial_capital if self._daily_pnl < 0 else 0.0

    def has_hit_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been breached."""
        return self.daily_loss_pct >= settings.max_daily_loss_pct

    def update_unrealised_pnl(self) -> None:
        """Recalculate unrealised PnL using current LTPs."""
        unrealised = 0.0
        for symbol, pos in self._positions.items():
            from config.instruments import get_token
            token = get_token(symbol)
            if token:
                ltp = angel_client.get_ltp("NSE", symbol, token)
                if ltp:
                    pos["ltp"] = ltp
                    pos["pnl"] = (ltp - pos["avg_price"]) * pos["quantity"]
                    unrealised += pos["pnl"]

        self._total_value = self._available_capital + sum(
            abs(p["quantity"]) * p.get("ltp", p["avg_price"]) for p in self._positions.values()
        )
        self._daily_pnl = self._realised_pnl + unrealised

    def record_trade_close(self, symbol: str, exit_price: float, quantity: int) -> float:
        """Record a closed trade and update PnL."""
        pos = self._positions.get(symbol)
        if not pos:
            return 0.0

        pnl = (exit_price - pos["avg_price"]) * quantity
        self._realised_pnl += pnl
        self._daily_pnl = self._realised_pnl
        self._trades_today += 1

        if pnl > 0:
            self._wins_today += 1
        else:
            self._losses_today += 1

        self._available_capital += exit_price * quantity + pnl
        return pnl

    def snapshot(self) -> dict[str, Any]:
        """Return current portfolio state as a dict."""
        return {
            "total_value": round(self._total_value, 2),
            "available_capital": round(self._available_capital, 2),
            "invested_value": round(self._total_value - self._available_capital, 2),
            "realised_pnl": round(self._realised_pnl, 2),
            "unrealised_pnl": round(self._daily_pnl - self._realised_pnl, 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "position_count": self.position_count,
            "positions": self._positions,
            "trades_today": self._trades_today,
            "wins_today": self._wins_today,
            "losses_today": self._losses_today,
        }

    def save_daily_snapshot(self) -> None:
        """Persist daily PnL to the database."""
        today = datetime.now(IST).strftime("%Y-%m-%d")
        with get_db() as db:
            existing = db.query(DailyPnL).filter(DailyPnL.date == today).first()
            if existing:
                existing.portfolio_value = self._total_value
                existing.realised_pnl = self._realised_pnl
                existing.unrealised_pnl = self._daily_pnl - self._realised_pnl
                existing.trades_count = self._trades_today
                existing.win_count = self._wins_today
                existing.loss_count = self._losses_today
            else:
                db.add(DailyPnL(
                    date=today,
                    portfolio_value=self._total_value,
                    realised_pnl=self._realised_pnl,
                    unrealised_pnl=self._daily_pnl - self._realised_pnl,
                    trades_count=self._trades_today,
                    win_count=self._wins_today,
                    loss_count=self._losses_today,
                ))


# Module-level singleton
portfolio_tracker = PortfolioTracker()
