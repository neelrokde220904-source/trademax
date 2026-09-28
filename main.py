"""India AI Trader — main entry point.

Usage:
    python main.py              # Run the full trading system (scheduler mode)
    python main.py --scan       # Run a single scan cycle
    python main.py --backtest   # Run backtest with default strategy
    python main.py --dashboard  # Start dashboard only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn
from loguru import logger

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings
from database.db import init_db
from risk.dead_mans_switch import dead_mans_switch


def setup_logging() -> None:
    """Configure loguru logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> — <level>{message}</level>",
    )
    logger.add(
        "logs/trader_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function} — {message}",
    )


async def run_scheduler_mode() -> None:
    """Run the full scheduled trading system."""
    from scheduler.market_scheduler import market_scheduler

    logger.info("=" * 60)
    logger.info("  INDIA AI TRADER — Starting")
    logger.info("  Mode: {}", "PAPER TRADING" if settings.paper_trading else "LIVE TRADING")
    logger.info("  Capital: ₹{:,.2f}", settings.initial_capital)
    logger.info("  Watchlist: {} stocks", len(settings.default_watchlist))
    logger.info("=" * 60)

    if not settings.paper_trading:
        logger.warning("⚠️  LIVE TRADING MODE — real money at risk!")

    # Setup and start scheduler
    market_scheduler.setup()
    market_scheduler.start()

    logger.info("Dead Man's Switch active (thresholds: {})", dead_mans_switch._thresholds)

    # Start dashboard in background
    config = uvicorn.Config(
        "dashboard.backend.app:app",
        host="0.0.0.0",
        port=settings.dashboard_port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    logger.info("Dashboard: http://localhost:{}", settings.dashboard_port)
    logger.info("Scheduler running. Press Ctrl+C to stop.")

    try:
        await server.serve()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down...")
        market_scheduler.stop()


async def run_single_scan() -> None:
    """Run a single scan cycle (for testing)."""
    from agents.graph import run_trading_pipeline
    from broker.angel_client import angel_client
    from broker.portfolio_tracker import portfolio_tracker
    from data.instrument_master import download_instrument_master

    logger.info("Running single scan cycle...")

    await download_instrument_master()
    angel_client.authenticate()
    portfolio_tracker.sync_from_broker()

    result = await run_trading_pipeline(settings.default_watchlist)

    executed = result.get("executed_orders", [])
    logger.info("Scan complete: {} orders executed.", len(executed))
    for order in executed:
        logger.info("  {} {} @ ₹{:.2f} × {}", order.get("action"), order.get("symbol"), order.get("price", 0), order.get("quantity", 0))


def run_backtest() -> None:
    """Run a sample backtest."""
    from backtester.engine import BacktestEngine
    from backtester.report_generator import generate_report
    from data.market_data import fetch_ohlcv

    logger.info("Running backtest...")

    engine = BacktestEngine()

    # Fetch historical data for watchlist
    import asyncio as aio

    async def _fetch():
        data = {}
        for symbol in settings.default_watchlist[:5]:  # limit for speed
            try:
                # fetch_ohlcv is synchronous; retaining the async wrapper keeps
                # the command interface consistent with the scheduler.
                df = fetch_ohlcv(symbol, "1d", days=365)
                if df is not None and not df.empty:
                    data[symbol] = df
            except Exception as e:
                logger.warning("Failed to fetch {}: {}", symbol, e)
        return data

    data = aio.run(_fetch())

    if not data:
        logger.error("No data fetched — cannot run backtest.")
        return

    # Simple Supertrend strategy function for backtest
    from strategies.supertrend_adx import SupertrendADX
    strategy = SupertrendADX()

    def strategy_fn(symbol, df, bar_index):
        if bar_index < 30:
            return []
        signals = strategy._analyse(symbol, df)
        return [signals] if signals else []

    result = engine.run(data, strategy_fn)
    report = generate_report(result, output_dir="backtest_results")

    logger.info("Backtest complete. See backtest_results/ for report.")


def run_dashboard() -> None:
    """Start only the dashboard."""
    logger.info("Starting dashboard on port {}...", settings.dashboard_port)
    uvicorn.run(
        "dashboard.backend.app:app",
        host="0.0.0.0",
        port=settings.dashboard_port,
        reload=False,
    )


def run_preflight() -> int:
    """Print non-mutating readiness diagnostics and return a shell-friendly status."""
    from operations.preflight import run_preflight as check

    result = check()
    logger.info("Preflight: {}", "PASS" if result.passed else "FAIL")
    for warning in result.warnings:
        logger.warning("Preflight warning: {}", warning)
    for error in result.errors:
        logger.error("Preflight error: {}", error)
    return 0 if result.passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="India AI Trader")
    parser.add_argument("--scan", action="store_true", help="Run a single scan cycle")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--dashboard", action="store_true", help="Start dashboard only")
    parser.add_argument("--preflight", action="store_true", help="Validate configuration without contacting brokers or placing orders")
    args = parser.parse_args()

    setup_logging()

    if args.preflight:
        raise SystemExit(run_preflight())

    # Every other command writes database state or needs the trade ledger.
    init_db()

    if args.scan:
        asyncio.run(run_single_scan())
    elif args.backtest:
        run_backtest()
    elif args.dashboard:
        run_dashboard()
    else:
        asyncio.run(run_scheduler_mode())


if __name__ == "__main__":
    main()
