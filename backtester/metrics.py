"""Backtester Metrics — performance calculation with India risk-free rate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from backtester.engine import BacktestResult
from config.settings import settings


@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    cagr: float = 0.0
    total_return_pct: float = 0.0
    avg_holding_days: float = 0.0
    risk_free_rate: float = 0.0
    expectancy: float = 0.0


def calculate_metrics(result: BacktestResult) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics from backtest result.

    Uses India 10Y bond yield (~6.5%) as risk-free rate for Sharpe/Sortino/Calmar.
    """
    m = PerformanceMetrics()
    m.risk_free_rate = settings.risk_free_rate

    trades = result.trades
    m.total_trades = len(trades)

    if m.total_trades == 0:
        return m

    # --- Basic stats ---
    pnls = [t.net_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    m.winning_trades = len(wins)
    m.losing_trades = len(losses)
    m.win_rate = m.winning_trades / m.total_trades if m.total_trades else 0
    m.total_pnl = sum(t.pnl for t in trades)
    m.total_costs = sum(t.costs for t in trades)
    m.net_pnl = sum(pnls)
    m.gross_profit = sum(wins) if wins else 0
    m.gross_loss = abs(sum(losses)) if losses else 0
    m.profit_factor = m.gross_profit / m.gross_loss if m.gross_loss > 0 else float("inf")
    m.avg_trade_pnl = m.net_pnl / m.total_trades
    m.avg_win = sum(wins) / len(wins) if wins else 0
    m.avg_loss = sum(losses) / len(losses) if losses else 0
    m.max_win = max(pnls)
    m.max_loss = min(pnls)

    # --- Return ---
    m.total_return_pct = (result.final_capital - result.initial_capital) / result.initial_capital * 100

    # --- Holding period ---
    holding_days = []
    for t in trades:
        if t.entry_date and t.exit_date:
            days = (t.exit_date - t.entry_date).days
            holding_days.append(max(1, days))
    m.avg_holding_days = sum(holding_days) / len(holding_days) if holding_days else 0

    # --- Equity curve metrics ---
    equity = result.equity_curve
    if len(equity) < 2:
        return m

    # Max drawdown
    peak = equity[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    for val in equity:
        peak = max(peak, val)
        dd = peak - val
        dd_pct = dd / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd_pct)
    m.max_drawdown = max_dd
    m.max_drawdown_pct = max_dd_pct * 100

    # Daily returns from equity curve
    eq_arr = np.array(equity, dtype=float)
    daily_returns = np.diff(eq_arr) / eq_arr[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    if len(daily_returns) < 2:
        return m

    # --- CAGR ---
    num_days = len(equity) - 1
    years = num_days / 252  # trading days
    if years > 0 and result.final_capital > 0:
        m.cagr = ((result.final_capital / result.initial_capital) ** (1 / years) - 1) * 100

    # --- Sharpe Ratio (annualised, India risk-free 6.5%) ---
    daily_rf = m.risk_free_rate / 252
    excess = daily_returns - daily_rf
    std = float(np.std(excess, ddof=1))
    if std > 0:
        m.sharpe_ratio = float(np.mean(excess)) / std * math.sqrt(252)

    # --- Sortino Ratio (using downside deviation) ---
    downside = excess[excess < 0]
    if len(downside) > 0:
        downside_std = float(np.std(downside, ddof=1))
        if downside_std > 0:
            m.sortino_ratio = float(np.mean(excess)) / downside_std * math.sqrt(252)

    # --- Calmar Ratio (CAGR / Max Drawdown) ---
    if m.max_drawdown_pct > 0:
        m.calmar_ratio = m.cagr / m.max_drawdown_pct

    # --- Expectancy ---
    m.expectancy = m.win_rate * m.avg_win + (1 - m.win_rate) * m.avg_loss

    return m


@dataclass
class MonteCarloResult:
    """Results of Monte Carlo trade-sequence simulation."""

    num_simulations: int = 0
    median_return_pct: float = 0.0
    p5_return_pct: float = 0.0
    p10_return_pct: float = 0.0
    p25_return_pct: float = 0.0
    p75_return_pct: float = 0.0
    p95_return_pct: float = 0.0
    median_max_drawdown_pct: float = 0.0
    p95_max_drawdown_pct: float = 0.0
    ruin_probability: float = 0.0  # % of paths that lose > 50%
    median_sharpe: float = 0.0
    confidence_interval_95: tuple[float, float] = (0.0, 0.0)


def monte_carlo_analysis(
    result: BacktestResult,
    num_simulations: int = 1000,
    ruin_threshold: float = 0.50,
) -> MonteCarloResult:
    """Run Monte Carlo simulation by shuffling trade sequence.

    Preserves the same set of trades but randomizes their order,
    generating different equity paths to assess strategy robustness.

    Args:
        result: BacktestResult with completed trades.
        num_simulations: Number of shuffled simulations.
        ruin_threshold: Fraction of capital loss considered "ruin" (default 50%).

    Returns:
        MonteCarloResult with percentile statistics.
    """
    mc = MonteCarloResult(num_simulations=num_simulations)

    trades = result.trades
    if len(trades) < 5:
        return mc

    # Extract net PnL from each trade
    pnls = np.array([t.net_pnl for t in trades], dtype=float)
    initial = result.initial_capital

    rng = np.random.default_rng(seed=42)
    final_returns: list[float] = []
    max_drawdowns: list[float] = []
    sharpes: list[float] = []
    ruin_count = 0

    for _ in range(num_simulations):
        shuffled = rng.permutation(pnls)
        equity = np.empty(len(shuffled) + 1, dtype=float)
        equity[0] = initial

        for i, pnl in enumerate(shuffled):
            equity[i + 1] = equity[i] + pnl

        final_ret = (equity[-1] - initial) / initial * 100
        final_returns.append(final_ret)

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        dd_pct = (peak - equity) / np.where(peak > 0, peak, 1.0)
        max_dd = float(np.max(dd_pct)) * 100
        max_drawdowns.append(max_dd)

        # Simple annualized Sharpe from trade-level returns
        trade_returns = shuffled / initial
        if np.std(trade_returns) > 0:
            sharpe = float(np.mean(trade_returns) / np.std(trade_returns) * math.sqrt(252 / max(len(trades), 1)))
            sharpes.append(sharpe)

        # Ruin check
        if equity[-1] < initial * (1 - ruin_threshold):
            ruin_count += 1

    returns_arr = np.array(final_returns)
    dd_arr = np.array(max_drawdowns)

    mc.median_return_pct = float(np.median(returns_arr))
    mc.p5_return_pct = float(np.percentile(returns_arr, 5))
    mc.p10_return_pct = float(np.percentile(returns_arr, 10))
    mc.p25_return_pct = float(np.percentile(returns_arr, 25))
    mc.p75_return_pct = float(np.percentile(returns_arr, 75))
    mc.p95_return_pct = float(np.percentile(returns_arr, 95))
    mc.median_max_drawdown_pct = float(np.median(dd_arr))
    mc.p95_max_drawdown_pct = float(np.percentile(dd_arr, 95))
    mc.ruin_probability = ruin_count / num_simulations * 100
    mc.confidence_interval_95 = (
        float(np.percentile(returns_arr, 2.5)),
        float(np.percentile(returns_arr, 97.5)),
    )

    if sharpes:
        mc.median_sharpe = float(np.median(sharpes))

    return mc
