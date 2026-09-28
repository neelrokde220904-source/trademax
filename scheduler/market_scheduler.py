"""Market Scheduler — APScheduler-based IST market-hours orchestration."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from agents.graph import run_trading_pipeline
from broker.angel_client import angel_client
from broker.portfolio_tracker import portfolio_tracker
from broker.websocket_feed import ws_feed
from config.settings import settings
from data.instrument_master import download_instrument_master
from notifications.telegram_notifier import telegram_notifier
from risk.dead_mans_switch import dead_mans_switch
from risk.stop_loss_manager import stop_loss_manager

IST = pytz.timezone("Asia/Kolkata")


class MarketScheduler:
    """Schedules all trading activities around NSE market hours.

    Daily Schedule (IST):
    08:00 — Pre-market: download instrument master, authenticate broker, sync portfolio
    09:00 — Connect WebSocket feed
    09:30 — First scan (after 15-min no-trade window)
    10:00-14:30 — Adaptive scans (30m/60m/120m based on HMM regime)
    14:45 — Last scan before close
    15:15 — Intraday square-off check
    15:35 — Post-market: disconnect WS, daily PnL snapshot
    16:00 — Send daily summary via Telegram
    18:00 — End-of-day cleanup
    20:00-06:00 — Overnight global cue monitoring (every 2 hours)
    """

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler(timezone=IST)
        self._running = False
        self._trading_paused = False

    @property
    def is_running(self) -> bool:
        """Whether APScheduler is running (used by the read-only dashboard health API)."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Whether new trading scans are manually paused while exit monitoring continues."""
        return self._trading_paused

    def pause_trading(self) -> None:
        """Block new scan cycles without disabling stop-loss and time-exit jobs."""
        self._trading_paused = True
        logger.warning("[Scheduler] New entries paused by dashboard control.")

    def resume_trading(self) -> None:
        """Allow subsequent scheduled scan cycles after a manual review."""
        self._trading_paused = False
        logger.info("[Scheduler] New entries resumed by dashboard control.")

    def setup(self) -> None:
        """Configure all scheduled jobs."""
        # Only run Mon-Fri (day_of_week="mon-fri")
        dow = "mon-fri"

        # --- PRE-MARKET 08:00 ---
        self.scheduler.add_job(
            self._pre_market,
            CronTrigger(hour=8, minute=0, day_of_week=dow, timezone=IST),
            id="pre_market",
            name="Pre-Market Setup",
            replace_existing=True,
        )

        # --- WEBSOCKET CONNECT 09:00 ---
        self.scheduler.add_job(
            self._connect_ws,
            CronTrigger(hour=9, minute=0, day_of_week=dow, timezone=IST),
            id="ws_connect",
            name="WebSocket Connect",
            replace_existing=True,
        )

        # --- FIRST SCAN 09:30 ---
        self.scheduler.add_job(
            self._run_scan,
            CronTrigger(hour=9, minute=30, day_of_week=dow, timezone=IST),
            id="scan_0930",
            name="First Scan 09:30",
            replace_existing=True,
        )

        # --- ADAPTIVE SCANS 10:00-14:45 ---
        # Instead of fixed hourly scans, run every 30 min and let _adaptive_scan
        # decide whether to actually scan based on the current HMM regime / VIX.
        #   HIGH_VOL_MEAN_REVERT → scan every 30 min  (all triggers fire)
        #   NORMAL_TRENDING      → scan every 60 min  (skip :30 triggers)
        #   LOW_VOL_TREND        → scan every 120 min (skip :30 and odd-hour triggers)
        #   SYSTEMIC_PANIC       → skip all scans     (DMS will also block)
        self.scheduler.add_job(
            self._adaptive_scan,
            CronTrigger(minute="0,30", hour="10-14", day_of_week=dow, timezone=IST),
            id="adaptive_scan",
            name="Adaptive Scan 10:00-14:30",
            replace_existing=True,
        )

        # --- LAST SCAN 14:45 (always runs regardless of regime) ---
        self.scheduler.add_job(
            self._run_scan,
            CronTrigger(hour=14, minute=45, day_of_week=dow, timezone=IST),
            id="scan_1445",
            name="Last Scan 14:45",
            replace_existing=True,
        )

        # --- SL/TARGET CHECK every 5 min during market hours ---
        self.scheduler.add_job(
            self._check_stops,
            CronTrigger(minute="*/5", hour="9-15", day_of_week=dow, timezone=IST),
            id="check_stops",
            name="Stop-Loss Check (5m)",
            replace_existing=True,
        )

        # --- INTRADAY SQUARE-OFF 15:15 ---
        self.scheduler.add_job(
            self._intraday_squareoff,
            CronTrigger(hour=15, minute=15, day_of_week=dow, timezone=IST),
            id="squareoff",
            name="Intraday Square-Off 15:15",
            replace_existing=True,
        )

        # --- POST-MARKET 15:35 ---
        self.scheduler.add_job(
            self._post_market,
            CronTrigger(hour=15, minute=35, day_of_week=dow, timezone=IST),
            id="post_market",
            name="Post-Market 15:35",
            replace_existing=True,
        )

        # --- DAILY SUMMARY 16:00 ---
        self.scheduler.add_job(
            self._daily_summary,
            CronTrigger(hour=16, minute=0, day_of_week=dow, timezone=IST),
            id="daily_summary",
            name="Daily Summary 16:00",
            replace_existing=True,
        )

        # --- END-OF-DAY 18:00 ---
        self.scheduler.add_job(
            self._end_of_day,
            CronTrigger(hour=18, minute=0, day_of_week=dow, timezone=IST),
            id="eod_cleanup",
            name="End-of-Day Cleanup 18:00",
            replace_existing=True,
        )

        # --- OVERNIGHT MONITOR 20:00, 22:00, 00:00, 02:00, 06:00 ---
        self.scheduler.add_job(
            self._overnight_check,
            CronTrigger(hour="20,22,0,2,6", minute=0, day_of_week=dow, timezone=IST),
            id="overnight_monitor",
            name="Overnight Position Monitor",
            replace_existing=True,
        )

        logger.info("[Scheduler] All jobs configured (including overnight monitor).")

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return
        self.scheduler.start()
        self._running = True
        logger.info("[Scheduler] Started.")

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return
        self.scheduler.shutdown(wait=False)
        self._running = False
        logger.info("[Scheduler] Stopped.")

    # ---- JOB IMPLEMENTATIONS ----

    async def _pre_market(self) -> None:
        """Pre-market setup at 08:00 IST."""
        today = datetime.now(IST).strftime("%Y-%m-%d")
        if today in settings.nse_holidays:
            logger.info("[Scheduler] NSE holiday today ({}) — skipping.", today)
            return

        logger.info("[Scheduler] ===== PRE-MARKET START =====")
        try:
            # Download fresh instrument master
            await download_instrument_master()

            # Broker authentication is optional in paper-only operation.  Public
            # OHLCV fallback and simulated fills remain available without it.
            if angel_client.has_credentials:
                angel_client.authenticate()
            else:
                logger.warning("[Scheduler] Angel One credentials absent; continuing in brokerless paper mode.")

            # Sync portfolio
            portfolio_tracker.sync_from_broker()

            logger.info("[Scheduler] Pre-market setup complete.")
        except Exception as exc:
            logger.error("[Scheduler] Pre-market failed: {}", exc)
            await telegram_notifier.notify_error(f"Pre-market setup failed: {exc}")

    async def _connect_ws(self) -> None:
        """Connect WebSocket at 09:00 IST."""
        if not angel_client.has_credentials:
            logger.info("[Scheduler] WebSocket skipped: brokerless paper mode.")
            return
        try:
            from config.instruments import get_token
            tokens = []
            for symbol in settings.default_watchlist:
                token = get_token(symbol)
                if token:
                    tokens.append({"exchangeType": 1, "tokens": [token]})

            ws_feed.connect()
            if tokens:
                ws_feed.subscribe(tokens[:50])  # Angel One allows max 50 per connection
            logger.info("[Scheduler] WebSocket connected with {} tokens.", len(tokens))
        except Exception as exc:
            logger.error("[Scheduler] WebSocket connect failed: {}", exc)

    async def _run_scan(self) -> None:
        """Run the full AI trading pipeline."""
        if self._trading_paused:
            logger.warning("[Scheduler] Scan skipped: new entries are manually paused.")
            return
        # Dead Man's Switch pre-check
        allowed, reason = dead_mans_switch.check_allowed()
        if not allowed:
            logger.warning("[Scheduler] Scan blocked by DMS: {}", reason)
            return

        logger.info("[Scheduler] Running trading pipeline scan...")
        try:
            result = await run_trading_pipeline(settings.default_watchlist)
            trades = result.get("executed_orders", [])
            logger.info("[Scheduler] Scan complete: {} orders executed.", len(trades))

            dead_mans_switch.record_success("langgraph")

            for trade in trades:
                await telegram_notifier.notify_trade_executed(trade)
                stop_loss_manager.register_trade(trade)

        except Exception as exc:
            logger.error("[Scheduler] Scan failed: {}", exc)
            dead_mans_switch.record_failure("langgraph", str(exc))
            await telegram_notifier.notify_error(f"Scan failed: {exc}")

    async def _adaptive_scan(self) -> None:
        """Regime-aware scan gate — decides whether the current 30-min slot should run."""
        now = datetime.now(IST)
        minute = now.minute
        hour = now.hour

        # Determine current regime (cached from last pipeline run)
        regime = getattr(settings, "_last_regime", "NORMAL_TRENDING")
        vix = getattr(settings, "_last_vix", 15.0)

        # SYSTEMIC_PANIC → block all scans (DMS should also block, belt-and-suspenders)
        if regime == "SYSTEMIC_PANIC" or vix > 28:
            logger.info(
                "[Scheduler] Adaptive scan SKIPPED — regime={}, VIX={:.1f}",
                regime, vix,
            )
            return

        # HIGH_VOL_MEAN_REVERT → every 30 min (all triggers fire)
        if regime == "HIGH_VOL_MEAN_REVERT":
            logger.info("[Scheduler] HIGH_VOL regime — scanning (every 30 min).")
            await self._run_scan()
            return

        # LOW_VOL_TREND → every 2 hours (only even-hour :00 triggers)
        if regime == "LOW_VOL_TREND":
            if minute == 0 and hour % 2 == 0:
                logger.info("[Scheduler] LOW_VOL regime — scanning (every 2 hours).")
                await self._run_scan()
            else:
                logger.debug("[Scheduler] LOW_VOL regime — skipping slot {:02d}:{:02d}.", hour, minute)
            return

        # NORMAL_TRENDING (default) → every 60 min (only :00 triggers)
        if minute == 0:
            logger.info("[Scheduler] NORMAL regime — scanning (every 60 min).")
            await self._run_scan()
        else:
            logger.debug("[Scheduler] NORMAL regime — skipping :30 slot.")

    async def _check_stops(self) -> None:
        """Check stop-losses and targets every 5 minutes."""
        try:
            portfolio_tracker.update_unrealised_pnl()
            exits = stop_loss_manager.check_all_positions()

            for exit_info in exits:
                logger.info("[Scheduler] Exit signal: {}", exit_info)
                symbol = exit_info["symbol"]
                pos = portfolio_tracker.positions.get(symbol, {})
                if pos:
                    result = self._place_exit_order(symbol, exit_info["exit_action"], pos)
                    if not result:
                        continue
                    stop_loss_manager.unregister_trade(symbol)

                    await telegram_notifier.notify_trade_closed({
                        "symbol": symbol,
                        "exit_price": pos.get("ltp", 0),
                        "entry_price": pos.get("avg_price", 0),
                        "pnl": pos.get("pnl", 0),
                        "pnl_pct": (pos.get("pnl", 0) / (pos.get("avg_price", 1) * abs(pos.get("quantity", 1)))) * 100,
                        "exit_reason": exit_info["reason"],
                    })

        except Exception as exc:
            logger.error("[Scheduler] Stop check failed: {}", exc)

    async def _intraday_squareoff(self) -> None:
        """Square off all intraday positions at 15:15 IST."""
        logger.info("[Scheduler] Intraday square-off check...")
        exits = stop_loss_manager.check_all_positions()
        for exit_info in exits:
            if exit_info.get("exit_type") == "TIME_EXIT":
                symbol = exit_info["symbol"]
                pos = portfolio_tracker.positions.get(symbol, {})
                if pos:
                    result = self._place_exit_order(symbol, exit_info["exit_action"], pos)
                    if result:
                        stop_loss_manager.unregister_trade(symbol)
                        logger.info("[Scheduler] Squared off {} (time exit).", symbol)

    @staticmethod
    def _place_exit_order(symbol: str, action: str, position: dict) -> dict | None:
        """Close a tracked position using the normal synchronous order-manager API.

        Exit calls deliberately bypass strategy selection but retain the same broker,
        paper/live mode, database audit trail, and market-hours guard as entries.
        """
        from broker.order_manager import place_order
        from config.instruments import get_token

        token = get_token(symbol)
        quantity = abs(int(position.get("quantity", 0)))
        if not token or quantity <= 0:
            logger.error("[Scheduler] Cannot exit {}: missing token or quantity.", symbol)
            return None

        return place_order(
            symbol=symbol,
            token=token,
            action=action,
            quantity=quantity,
            order_type="MARKET",
            price=float(position.get("ltp", 0)),
            exchange="NSE",
            product_type="INTRADAY",
            strategy="risk_exit",
            reasoning="Automated stop-loss, target, or time-based exit",
        )

    async def _post_market(self) -> None:
        """Post-market tasks at 15:35 IST."""
        logger.info("[Scheduler] ===== POST-MARKET =====")
        try:
            ws_feed.disconnect()
            portfolio_tracker.sync_from_broker()
            portfolio_tracker.save_daily_snapshot()
            logger.info("[Scheduler] Post-market tasks complete.")
        except Exception as exc:
            logger.error("[Scheduler] Post-market failed: {}", exc)

    async def _daily_summary(self) -> None:
        """Send daily summary at 16:00 IST."""
        try:
            summary = {
                "pnl": portfolio_tracker.daily_pnl,
                "trades": portfolio_tracker._trades_today,
                "wins": portfolio_tracker._wins_today,
                "losses": portfolio_tracker._losses_today,
                "portfolio_value": portfolio_tracker.total_value,
            }
            await telegram_notifier.notify_daily_summary(summary)
        except Exception as exc:
            logger.error("[Scheduler] Daily summary failed: {}", exc)

    async def _end_of_day(self) -> None:
        """End-of-day cleanup at 18:00 IST."""
        logger.info("[Scheduler] End-of-day cleanup.")
        try:
            angel_client.logout()
        except Exception:
            pass

    async def _overnight_check(self) -> None:
        """Check global overnight risks for open positions."""
        try:
            from scheduler.overnight_monitor import overnight_monitor

            positions = portfolio_tracker.positions
            if not positions:
                logger.debug("[Scheduler] No open positions — skipping overnight check.")
                return

            result = await overnight_monitor.check_overnight_risks(positions)
            alerts = result.get("alerts", [])
            if alerts:
                logger.warning("[Scheduler] Overnight alerts: {}", len(alerts))
            else:
                logger.debug("[Scheduler] Overnight check clear — no threshold breaches.")
        except Exception as exc:
            logger.error("[Scheduler] Overnight check failed: {}", exc)


# Singleton
market_scheduler = MarketScheduler()
