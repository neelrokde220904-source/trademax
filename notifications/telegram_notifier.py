"""Telegram Notifier — trade alerts, daily PnL summaries, error alerts."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytz
from loguru import logger

from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")


class TelegramNotifier:
    """Send notifications via Telegram Bot API.

    Message types:
    - Trade Executed (BUY/SELL with entry details)
    - Trade Closed (PnL)
    - Daily PnL Summary
    - Strategy Signal
    - Risk Alert
    - System Error
    """

    def __init__(self) -> None:
        self._bot = None
        self._enabled = bool(settings.telegram_bot_token and settings.telegram_chat_id)
        if not self._enabled:
            logger.warning("[Telegram] Bot token or chat ID missing — notifications disabled.")

    async def _get_bot(self):
        """Lazy-init the telegram bot."""
        if self._bot is None and self._enabled:
            from telegram import Bot
            self._bot = Bot(token=settings.telegram_bot_token)
        return self._bot

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured Telegram chat."""
        if not self._enabled:
            logger.debug("[Telegram] Disabled — would send: {}", text[:100])
            return False

        try:
            bot = await self._get_bot()
            if bot:
                await bot.send_message(
                    chat_id=settings.telegram_chat_id,
                    text=text,
                    parse_mode=parse_mode,
                )
                return True
        except Exception as exc:
            logger.error("[Telegram] Failed to send message: {}", exc)
        return False

    # ----- TRADE NOTIFICATIONS -----

    async def notify_trade_executed(self, trade: dict[str, Any]) -> None:
        """Send trade execution notification."""
        action = trade.get("action", "?")
        symbol = trade.get("symbol", "?")
        price = trade.get("price", 0)
        qty = trade.get("quantity", 0)
        sl = trade.get("stop_loss", 0)
        target = trade.get("target", 0)
        strategy = trade.get("strategy", "N/A")
        confidence = trade.get("confidence", 0)
        mode = "📝 PAPER" if settings.paper_trading else "🔴 LIVE"

        emoji = "🟢" if action == "BUY" else "🔴"

        msg = (
            f"{emoji} <b>Trade Executed</b> {mode}\n\n"
            f"<b>{action} {symbol}</b>\n"
            f"Price: ₹{price:,.2f} × {qty} shares\n"
            f"Value: ₹{price * qty:,.2f}\n"
            f"SL: ₹{sl:,.2f} | Target: ₹{target:,.2f}\n"
            f"Strategy: {strategy}\n"
            f"Confidence: {confidence}%\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S')}"
        )
        await self.send_message(msg)

    async def notify_trade_closed(self, trade: dict[str, Any]) -> None:
        """Send trade closure notification."""
        symbol = trade.get("symbol", "?")
        entry = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        pnl = trade.get("pnl", 0)
        pnl_pct = trade.get("pnl_pct", 0)
        reason = trade.get("exit_reason", "?")

        emoji = "💰" if pnl >= 0 else "📉"
        sign = "+" if pnl >= 0 else ""

        msg = (
            f"{emoji} <b>Trade Closed</b>\n\n"
            f"<b>{symbol}</b>\n"
            f"Entry: ₹{entry:,.2f} → Exit: ₹{exit_price:,.2f}\n"
            f"PnL: {sign}₹{pnl:,.2f} ({sign}{pnl_pct:.2f}%)\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S')}"
        )
        await self.send_message(msg)

    async def notify_daily_summary(self, summary: dict[str, Any]) -> None:
        """Send end-of-day summary."""
        pnl = summary.get("pnl", 0)
        trades = summary.get("trades", 0)
        wins = summary.get("wins", 0)
        losses = summary.get("losses", 0)
        portfolio_value = summary.get("portfolio_value", 0)
        win_rate = (wins / trades * 100) if trades > 0 else 0

        emoji = "📈" if pnl >= 0 else "📉"
        sign = "+" if pnl >= 0 else ""

        msg = (
            f"{emoji} <b>Daily Summary</b> — {datetime.now(IST).strftime('%d %b %Y')}\n\n"
            f"PnL: {sign}₹{pnl:,.2f}\n"
            f"Total Trades: {trades} (W:{wins} L:{losses})\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Portfolio Value: ₹{portfolio_value:,.2f}\n"
            f"Mode: {'📝 Paper' if settings.paper_trading else '🔴 Live'}"
        )
        await self.send_message(msg)

    async def notify_signal(self, signal: dict[str, Any]) -> None:
        """Send a raw signal alert."""
        action = signal.get("action", "?")
        symbol = signal.get("symbol", "?")
        confidence = signal.get("confidence", 0)
        strategy = signal.get("strategy", "")
        reasoning = signal.get("reasoning", "")

        msg = (
            f"🔔 <b>Signal: {action} {symbol}</b>\n"
            f"Confidence: {confidence}%\n"
            f"Strategy: {strategy}\n"
            f"Reason: {reasoning[:200]}"
        )
        await self.send_message(msg)

    async def notify_risk_alert(self, alert: str) -> None:
        """Send a risk alert."""
        msg = f"⚠️ <b>RISK ALERT</b>\n\n{alert}"
        await self.send_message(msg)

    async def notify_error(self, error: str) -> None:
        """Send a system error alert."""
        msg = (
            f"🚨 <b>System Error</b>\n\n"
            f"{error[:500]}\n"
            f"Time: {datetime.now(IST).strftime('%H:%M:%S')}"
        )
        await self.send_message(msg)

    # ----- COMMAND HANDLERS -----

    async def handle_resume_trading(self) -> None:
        """Handle /resume_trading command — resumes after Dead Man's Switch halt."""
        from risk.dead_mans_switch import dead_mans_switch

        if not dead_mans_switch.is_halted:
            await self.send_message("ℹ️ Trading is already active — no halt to resume from.")
            return

        resumed = dead_mans_switch.resume_trading(reason="Telegram /resume_trading command")
        if resumed:
            await self.send_message("✅ Trading resumed successfully. All failure counters reset.")
        else:
            await self.send_message("⚠️ Failed to resume trading. Check logs.")

    async def handle_dms_status(self) -> None:
        """Handle /dms_status command — show Dead Man's Switch status."""
        from risk.dead_mans_switch import dead_mans_switch

        status = dead_mans_switch.get_status()
        halted = "⛔ HALTED" if status["halted"] else "✅ ACTIVE"

        msg = f"🛡️ <b>Dead Man's Switch Status: {halted}</b>\n\n"

        if status["halted"]:
            msg += f"<b>Reason:</b> {status['halt_reason']}\n"
            msg += f"<b>Halted at:</b> {status['halt_time']}\n\n"

        msg += "<b>Failure counts:</b>\n"
        for comp, cnt in status["failure_counts"].items():
            threshold = status["thresholds"].get(comp, 3)
            icon = "🔴" if cnt >= threshold else "🟡" if cnt > 0 else "🟢"
            msg += f"  {icon} {comp}: {cnt}/{threshold}\n"

        if not status["failure_counts"]:
            msg += "  All clear — no failures recorded.\n"

        await self.send_message(msg)


# Singleton
telegram_notifier = TelegramNotifier()
