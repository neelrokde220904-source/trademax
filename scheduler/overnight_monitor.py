"""Overnight Position Monitor — monitors global cues and open positions during 15:35 → 08:00 IST gap."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytz
import yfinance as yf
from loguru import logger

from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")


class OvernightMonitor:
    """Monitors overnight risks for open positions when NSE is closed.

    Checks every 2 hours between 20:00 IST and 08:00 IST:
    - S&P 500 / Nasdaq futures (US market proxy)
    - SGX Nifty / GIFT Nifty (India proxy)
    - Crude oil (Brent) price moves
    - USD/INR movement
    - Any earnings or events after-hours

    Triggers Telegram alert if overnight move exceeds threshold.
    """

    DEFAULT_THRESHOLDS: dict[str, float] = {
        "sp500_pct": 1.5,        # Alert if S&P moves > ±1.5%
        "nasdaq_pct": 2.0,       # Alert if Nasdaq moves > ±2.0%
        "crude_pct": 3.0,        # Alert if Brent crude moves > ±3%
        "usdinr_pct": 0.5,       # Alert if USD/INR moves > ±0.5%
        "gift_nifty_pct": 1.0,   # Alert if GIFT Nifty gap > ±1%
    }

    # Yahoo Finance tickers for overnight proxies
    PROXY_TICKERS: dict[str, str] = {
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "crude_brent": "BZ=F",
        "usdinr": "USDINR=X",
        "gift_nifty": "^NSEI",  # Will use last close as proxy
    }

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        self._thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self._last_check: datetime | None = None
        self._last_values: dict[str, float] = {}
        self._notifier = None

    async def check_overnight_risks(self, open_positions: dict[str, Any] | None = None) -> dict[str, Any]:
        """Check global proxies for overnight risks.

        Returns:
            Dict with alerts and proxy values.
        """
        now = datetime.now(IST)
        alerts: list[dict[str, Any]] = []
        proxy_values: dict[str, Any] = {}

        for name, ticker in self.PROXY_TICKERS.items():
            try:
                data = yf.Ticker(ticker)
                hist = data.history(period="2d")
                if hist.empty or len(hist) < 2:
                    continue

                prev_close = float(hist["Close"].iloc[-2])
                latest = float(hist["Close"].iloc[-1])
                pct_change = (latest - prev_close) / prev_close * 100 if prev_close > 0 else 0

                proxy_values[name] = {
                    "value": round(latest, 2),
                    "prev_close": round(prev_close, 2),
                    "change_pct": round(pct_change, 2),
                }

                # Check threshold
                threshold_key = f"{name}_pct"
                threshold = self._thresholds.get(threshold_key, 2.0)
                if abs(pct_change) > threshold:
                    alerts.append({
                        "proxy": name,
                        "ticker": ticker,
                        "change_pct": round(pct_change, 2),
                        "threshold": threshold,
                        "direction": "UP" if pct_change > 0 else "DOWN",
                    })

            except Exception as exc:
                logger.debug("[OvernightMonitor] Failed to fetch {}: {}", name, exc)

        self._last_check = now
        self._last_values = proxy_values

        result = {
            "check_time": now.isoformat(),
            "proxy_values": proxy_values,
            "alerts": alerts,
            "has_open_positions": bool(open_positions),
        }

        # Fire alert if thresholds breached and there are open positions
        if alerts and open_positions:
            await self._send_overnight_alert(alerts, proxy_values, open_positions)

        return result

    async def _send_overnight_alert(
        self, alerts: list[dict], proxy_values: dict, open_positions: dict
    ) -> None:
        """Send Telegram alert for overnight risk."""
        try:
            if self._notifier is None:
                from notifications.telegram_notifier import telegram_notifier
                self._notifier = telegram_notifier

            msg = (
                f"🌙 <b>OVERNIGHT RISK ALERT</b>\n"
                f"<i>{datetime.now(IST).strftime('%H:%M IST, %d %b')}</i>\n\n"
            )

            for alert in alerts:
                direction_emoji = "📈" if alert["direction"] == "UP" else "📉"
                msg += (
                    f"{direction_emoji} <b>{alert['proxy'].upper()}</b>: "
                    f"{alert['change_pct']:+.2f}% "
                    f"(threshold: ±{alert['threshold']}%)\n"
                )

            num_positions = len(open_positions)
            msg += f"\n<b>Open positions:</b> {num_positions}\n"

            # List positions at risk
            for symbol, pos in list(open_positions.items())[:5]:
                qty = pos.get("quantity", 0) if isinstance(pos, dict) else 0
                msg += f"  • {symbol} ({qty} shares)\n"

            if num_positions > 5:
                msg += f"  ... and {num_positions - 5} more\n"

            msg += "\n<b>Action:</b> Review positions before 09:15 IST market open."

            await self._notifier.send_message(msg)
        except Exception as exc:
            logger.error("[OvernightMonitor] Failed to send alert: {}", exc)

    def get_last_check(self) -> dict[str, Any]:
        """Return last check results for dashboard."""
        return {
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "proxy_values": self._last_values,
        }


# Singleton
overnight_monitor = OvernightMonitor()
