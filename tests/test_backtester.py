"""Tests for backtester engine and metrics."""

import pandas as pd
import numpy as np

from backtester.engine import BacktestEngine
from backtester.metrics import calculate_metrics


def _make_data(symbols: list[str], n: int = 252) -> dict[str, pd.DataFrame]:
    """Generate synthetic daily data for backtesting."""
    np.random.seed(42)
    data = {}
    for symbol in symbols:
        dates = pd.bdate_range("2024-01-01", periods=n)
        close = 100 + np.cumsum(np.random.randn(n) * 1.0)
        close = np.maximum(close, 10)  # prevent negative prices
        df = pd.DataFrame({
            "open": close + np.random.randn(n) * 0.3,
            "high": close + abs(np.random.randn(n) * 1.0),
            "low": close - abs(np.random.randn(n) * 1.0),
            "close": close,
            "volume": np.random.randint(100_000, 1_000_000, n),
        }, index=dates)
        data[symbol] = df
    return data


def _dummy_strategy(symbol: str, df: pd.DataFrame, bar_index: int):
    """Buy every 30 bars with simple SL/target."""
    if bar_index < 30 or bar_index % 30 != 0:
        return []
    close = float(df.iloc[-1]["close"])
    return [{
        "symbol": symbol,
        "action": "BUY",
        "price": close,
        "stop_loss": close * 0.95,
        "target": close * 1.10,
        "strategy": "test",
        "holding_period": "SWING",
    }]


def test_backtest_runs():
    """Backtest should execute without errors."""
    data = _make_data(["RELIANCE", "TCS"])
    engine = BacktestEngine(initial_capital=100_000)
    result = engine.run(data, _dummy_strategy)

    assert result.initial_capital == 100_000
    assert result.final_capital > 0
    assert len(result.equity_curve) > 0


def test_backtest_metrics():
    """Metrics calculation should produce valid numbers."""
    data = _make_data(["RELIANCE"])
    engine = BacktestEngine(initial_capital=100_000)
    result = engine.run(data, _dummy_strategy)

    metrics = calculate_metrics(result)
    assert metrics.total_trades >= 0
    assert metrics.win_rate >= 0
    assert metrics.win_rate <= 1
    assert metrics.risk_free_rate == 0.065


def test_backtest_costs():
    """Trades should have nonzero transaction costs."""
    data = _make_data(["RELIANCE"])
    engine = BacktestEngine(initial_capital=100_000)
    result = engine.run(data, _dummy_strategy)

    if result.trades:
        assert any(t.costs > 0 for t in result.trades)
