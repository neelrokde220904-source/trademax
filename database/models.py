"""SQLAlchemy models for the India AI Trader system."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class Trade(Base):
    """Executed or paper-traded order record."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(10), default="NSE")
    action = Column(String(10), nullable=False)  # BUY / SELL
    quantity = Column(Integer, nullable=False)
    order_type = Column(String(20), default="LIMIT")  # LIMIT / MARKET / SL / SL-M
    price = Column(Float, nullable=False)
    trigger_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    target = Column(Float, nullable=True)
    holding_period = Column(String(20), default="INTRADAY")  # INTRADAY / DELIVERY / SWING
    status = Column(String(20), default="PENDING")  # PENDING / EXECUTED / CANCELLED / REJECTED
    order_id = Column(String(50), nullable=True)  # Broker order ID
    is_paper = Column(Boolean, default=True)
    pnl = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_timestamp = Column(DateTime, nullable=True)
    strategy = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())


class Signal(Base):
    """Agent signal record — every signal from every agent is logged."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    agent_name = Column(String(50), nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    signal = Column(String(20), nullable=False)  # BUY / SELL / HOLD / STRONG_BUY / STRONG_SELL
    confidence = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)
    data_json = Column(Text, nullable=True)  # Serialised dict of full signal payload
    created_at = Column(DateTime, default=func.now())


class Portfolio(Base):
    """Current portfolio snapshot — updated on every trade."""

    __tablename__ = "portfolio"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    total_value = Column(Float, nullable=False)
    available_capital = Column(Float, nullable=False)
    invested_value = Column(Float, nullable=False)
    realised_pnl = Column(Float, default=0.0)
    unrealised_pnl = Column(Float, default=0.0)
    positions_json = Column(Text, nullable=True)  # JSON of open positions
    created_at = Column(DateTime, default=func.now())


class BacktestResult(Base):
    """Backtest run results."""

    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(50), nullable=False)
    start_date = Column(String(20), nullable=False)
    end_date = Column(String(20), nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return_pct = Column(Float, nullable=True)
    cagr = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    max_drawdown_duration_days = Column(Integer, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    avg_holding_period_hours = Column(Float, nullable=True)
    vs_nifty_return_pct = Column(Float, nullable=True)
    metrics_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())


class DailyPnL(Base):
    """Daily PnL snapshot for equity curve."""

    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, unique=True, index=True)
    portfolio_value = Column(Float, nullable=False)
    realised_pnl = Column(Float, default=0.0)
    unrealised_pnl = Column(Float, default=0.0)
    trades_count = Column(Integer, default=0)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
