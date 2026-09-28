"""Dead Man's Switch — consecutive failure monitor with auto-halt and Telegram alerting."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any

import pytz
from loguru import logger

from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")


class DeadMansSwitch:
    """Tracks consecutive failures across critical components and halts trading if thresholds breach.

    Monitored components:
    - langgraph: pipeline execution failures
    - anthropic_api: LLM call failures
    - hmm_classifier: regime classification failures
    - broker_api: order placement / data fetch failures
    - websocket: tick feed disconnections

    Thresholds (consecutive failures to trigger halt):
    - langgraph: 3
    - anthropic_api: 5
    - hmm_classifier: 3
    - broker_api: 3
    - websocket: 5
    """

    DEFAULT_THRESHOLDS: dict[str, int] = {
        "langgraph": 3,
        "anthropic_api": 5,
        "hmm_classifier": 3,
        "broker_api": 3,
        "websocket": 5,
    }

    def __init__(self, thresholds: dict[str, int] | None = None) -> None:
        self._thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self._consecutive_failures: dict[str, int] = defaultdict(int)
        self._halted = False
        self._halt_reason = ""
        self._halt_time: datetime | None = None
        self._failure_log: list[dict[str, Any]] = []
        self._notifier = None

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def record_success(self, component: str) -> None:
        """Reset consecutive failure counter for a component on success."""
        if component in self._consecutive_failures:
            prev = self._consecutive_failures[component]
            if prev > 0:
                logger.debug("[DMS] {} recovered after {} consecutive failures.", component, prev)
            self._consecutive_failures[component] = 0

    def record_failure(self, component: str, error: str = "") -> bool:
        """Record a failure for a component.

        Returns:
            True if this failure triggered a trading halt.
        """
        self._consecutive_failures[component] += 1
        count = self._consecutive_failures[component]
        threshold = self._thresholds.get(component, 3)

        self._failure_log.append({
            "component": component,
            "error": error[:200],
            "count": count,
            "threshold": threshold,
            "time": datetime.now(IST).isoformat(),
        })

        # Keep log bounded
        if len(self._failure_log) > 500:
            self._failure_log = self._failure_log[-250:]

        logger.warning(
            "[DMS] {} failure #{}/{} — {}",
            component, count, threshold, error[:100] if error else "no details",
        )

        if count >= threshold and not self._halted:
            self._trigger_halt(component, count, error)
            return True

        return False

    def _trigger_halt(self, component: str, count: int, error: str) -> None:
        """Halt all trading and fire alert."""
        self._halted = True
        self._halt_time = datetime.now(IST)
        self._halt_reason = (
            f"{component} failed {count} consecutive times. "
            f"Last error: {error[:150]}"
        )
        logger.critical("[DMS] ⛔ TRADING HALTED — {}", self._halt_reason)

        # Fire-and-forget Telegram alert
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send_halt_alert())
        except RuntimeError:
            # No running loop — try sync
            pass

    async def _send_halt_alert(self) -> None:
        """Send halt notification via Telegram."""
        try:
            if self._notifier is None:
                from notifications.telegram_notifier import telegram_notifier
                self._notifier = telegram_notifier

            msg = (
                f"⛔ <b>DEAD MAN'S SWITCH — TRADING HALTED</b>\n\n"
                f"<b>Reason:</b> {self._halt_reason}\n"
                f"<b>Time:</b> {self._halt_time.strftime('%H:%M:%S IST') if self._halt_time else 'N/A'}\n\n"
                f"<b>Failure counts:</b>\n"
            )
            for comp, cnt in self._consecutive_failures.items():
                threshold = self._thresholds.get(comp, 3)
                status = "🔴" if cnt >= threshold else "🟡" if cnt > 0 else "🟢"
                msg += f"  {status} {comp}: {cnt}/{threshold}\n"

            msg += (
                f"\n<b>Action required:</b> Investigate and run "
                f"<code>/resume_trading</code> to restart."
            )
            await self._notifier.send_message(msg)
        except Exception as exc:
            logger.error("[DMS] Failed to send halt alert: {}", exc)

    def resume_trading(self, reason: str = "manual") -> bool:
        """Resume trading after manual review.

        Returns:
            True if trading was resumed (was halted), False if wasn't halted.
        """
        if not self._halted:
            return False

        logger.info("[DMS] Trading RESUMED — reason: {}", reason)
        self._halted = False
        self._halt_reason = ""
        self._halt_time = None
        # Reset all counters
        self._consecutive_failures.clear()

        # Fire-and-forget Telegram notification
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send_resume_alert(reason))
        except RuntimeError:
            pass

        return True

    async def _send_resume_alert(self, reason: str) -> None:
        """Notify that trading has resumed."""
        try:
            if self._notifier is None:
                from notifications.telegram_notifier import telegram_notifier
                self._notifier = telegram_notifier

            msg = (
                f"✅ <b>TRADING RESUMED</b>\n\n"
                f"<b>Reason:</b> {reason}\n"
                f"<b>Time:</b> {datetime.now(IST).strftime('%H:%M:%S IST')}\n"
                f"All failure counters reset."
            )
            await self._notifier.send_message(msg)
        except Exception as exc:
            logger.error("[DMS] Failed to send resume alert: {}", exc)

    def get_status(self) -> dict[str, Any]:
        """Return current DMS status for the dashboard."""
        return {
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "halt_time": self._halt_time.isoformat() if self._halt_time else None,
            "failure_counts": dict(self._consecutive_failures),
            "thresholds": dict(self._thresholds),
            "recent_failures": self._failure_log[-10:],
        }

    def check_allowed(self) -> tuple[bool, str]:
        """Pre-trade check: is trading allowed?

        Returns:
            (allowed, reason) — same signature as RiskEngine checks.
        """
        if self._halted:
            return False, f"Dead Man's Switch HALTED: {self._halt_reason}"
        return True, "DMS OK"


# Singleton
dead_mans_switch = DeadMansSwitch()
