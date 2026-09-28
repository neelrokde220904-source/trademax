"""Dashboard backend — FastAPI server with SSE real-time updates."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import pytz
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from broker.portfolio_tracker import portfolio_tracker
from config.settings import settings
from database.db import get_db
from database.models import DailyPnL, Signal, Trade
from risk.dead_mans_switch import dead_mans_switch

IST = pytz.timezone("Asia/Kolkata")

async def _dashboard_heartbeat() -> None:
    """Publish a compact status event for connected dashboards every five seconds."""
    while True:
        await broadcast_event("system_status", _system_status())
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Start and cleanly cancel the dashboard's non-trading status publisher."""
    task = asyncio.create_task(_dashboard_heartbeat())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="India AI Trader", version="1.1.0", lifespan=lifespan)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- MODELS ----

class TradeResponse(BaseModel):
    id: int
    symbol: str
    action: str
    price: float
    quantity: int
    strategy: str | None = None
    status: str
    pnl: float | None = None
    created_at: str


class SignalResponse(BaseModel):
    id: int
    symbol: str
    signal: str
    confidence: float | None = None
    strategy: str | None = None
    reasoning: str | None = None
    created_at: str


class PortfolioResponse(BaseModel):
    positions: list[dict[str, Any]]
    available_capital: float
    total_value: float
    daily_pnl: float
    position_count: int


class ControlRequest(BaseModel):
    action: str  # "pause" or "resume"; live-mode changes are never exposed here.


# ---- SSE EVENT STREAM ----

_sse_clients: list[asyncio.Queue] = []


def _system_status() -> dict[str, Any]:
    """Return UI-safe operational state without exposing credentials or secrets."""
    from broker.angel_client import angel_client
    from scheduler.market_scheduler import market_scheduler

    dms = dead_mans_switch.get_status()
    return {
        "mode": "PAPER" if settings.paper_trading else "LIVE",
        "paper_trading": settings.paper_trading,
        "scheduler_running": market_scheduler.is_running,
        "trading_paused": market_scheduler.is_paused,
        "broker_configured": angel_client.has_credentials,
        "broker_authenticated": angel_client.is_authenticated,
        "dead_mans_switch": dms,
        "timestamp": datetime.now(IST).isoformat(),
    }


async def broadcast_event(event_type: str, data: dict) -> None:
    """Broadcast an SSE event to all connected clients."""
    payload = json.dumps({"type": event_type, "data": data, "timestamp": datetime.now(IST).isoformat()})
    for queue in list(_sse_clients):
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass


async def sse_generator() -> AsyncGenerator[str, None]:
    """SSE event stream generator."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _sse_clients.append(queue)
    try:
        while True:
            data = await queue.get()
            yield f"data: {data}\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        _sse_clients.remove(queue)


# ---- ROUTES ----

@app.get("/")
async def root():
    return _system_status()


@app.get("/api/health")
async def get_health():
    """Expose trading-system health for the dashboard's persistent status bar."""
    return _system_status()


@app.get("/api/portfolio", response_model=PortfolioResponse)
async def get_portfolio():
    """Get current portfolio state."""
    positions = [
        {
            "symbol": k,
            "quantity": v["quantity"],
            "avg_price": v["avg_price"],
            "ltp": v.get("ltp", 0),
            "pnl": v.get("pnl", 0),
            "product": v.get("product", ""),
        }
        for k, v in portfolio_tracker.positions.items()
    ]
    return PortfolioResponse(
        positions=positions,
        available_capital=portfolio_tracker.available_capital,
        total_value=portfolio_tracker.total_value,
        daily_pnl=portfolio_tracker.daily_pnl,
        position_count=portfolio_tracker.position_count,
    )


@app.get("/api/trades")
async def get_trades(limit: int = 50, offset: int = 0):
    """Get recent trades."""
    with get_db() as db:
        trades = (
            db.query(Trade)
            .order_by(Trade.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [
            TradeResponse(
                id=t.id,
                symbol=t.symbol,
                action=t.action,
                price=t.price,
                quantity=t.quantity,
                strategy=t.strategy,
                status=t.status,
                pnl=t.pnl,
                created_at=t.created_at.isoformat() if t.created_at else "",
            )
            for t in trades
        ]


@app.get("/api/signals")
async def get_signals(limit: int = 50, offset: int = 0):
    """Get recent signals."""
    with get_db() as db:
        signals = (
            db.query(Signal)
            .order_by(Signal.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [
            SignalResponse(
                id=s.id,
                symbol=s.symbol,
                signal=s.signal,
                confidence=s.confidence,
                strategy=s.strategy,
                reasoning=s.reasoning,
                created_at=s.created_at.isoformat() if s.created_at else "",
            )
            for s in signals
        ]


@app.get("/api/pnl/daily")
async def get_daily_pnl(days: int = 30):
    """Get daily PnL history."""
    with get_db() as db:
        records = (
            db.query(DailyPnL)
            .order_by(DailyPnL.date.desc())
            .limit(days)
            .all()
        )
        return [
            {
                "date": r.date,
                "pnl": r.realised_pnl + r.unrealised_pnl,
                "portfolio_value": r.portfolio_value,
                "trades": r.trades_count,
                "wins": r.win_count,
                "losses": r.loss_count,
            }
            for r in reversed(records)
        ]


@app.get("/api/equity-curve")
async def get_equity_curve():
    """Get equity curve from daily PnL history."""
    with get_db() as db:
        records = db.query(DailyPnL).order_by(DailyPnL.date.asc()).all()
        return [
            {"date": r.date, "value": r.portfolio_value}
            for r in records
        ]


@app.get("/api/agents/status")
async def get_agent_status():
    """Return a transparent readiness view, not fabricated running state."""
    agents = [
        "MarketDataAgent", "TechnicalAgent", "FundamentalAgent",
        "SentimentAgent", "OptionsAgent", "MacroAgent",
        "IndiaSpecificAgent", "RiskManagerAgent", "PortfolioManagerAgent",
    ]
    scheduler_status = "paused" if _system_status()["trading_paused"] else "ready"
    return [{"name": a, "status": scheduler_status, "last_run": None} for a in agents]


@app.get("/api/stream")
async def sse_endpoint():
    """SSE endpoint for real-time updates."""
    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/control")
async def control(req: ControlRequest):
    """Pause or resume scan scheduling; mode changes remain configuration-only."""
    from scheduler.market_scheduler import market_scheduler

    if req.action == "pause":
        market_scheduler.pause_trading()
    elif req.action == "resume":
        market_scheduler.resume_trading()
    else:
        raise HTTPException(status_code=400, detail="Supported actions are: pause, resume")

    status = _system_status()
    await broadcast_event("control", {"action": req.action, "status": status})
    return {"status": "ok", "action": req.action, "system": status}


@app.get("/api/config")
async def get_config():
    """Get current (safe) config values."""
    return {
        "paper_trading": settings.paper_trading,
        "initial_capital": settings.initial_capital,
        "max_positions": settings.max_positions,
        "max_daily_loss_pct": settings.max_daily_loss_pct,
        "max_capital_per_trade_pct": settings.max_capital_per_trade_pct,
        "watchlist": settings.default_watchlist,
        "dashboard_port": settings.dashboard_port,
    }
