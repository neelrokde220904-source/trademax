"""Backtester Engine — event-driven backtesting with realistic Indian market costs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings


@dataclass
class BacktestTrade:
    symbol: str
    action: str  # BUY / SELL
    entry_date: datetime
    entry_price: float
    quantity: int
    stop_loss: float
    target: float
    strategy: str
    exit_date: datetime | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0


@dataclass
class BacktestResult:
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    dates: list[datetime] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0


class BacktestEngine:
    """Event-driven backtester with Indian market cost model.

    Costs modelled:
    - Brokerage: ₹20 per executed order (Angel One flat fee)
    - STT: 0.1% delivery / 0.025% intraday (on sell side)
    - Exchange charges: 0.00345%
    - GST: 18% on brokerage
    - Stamp duty: 0.015% on buy side
    - Slippage: 0.05% conservative
    """

    def __init__(
        self,
        initial_capital: float | None = None,
        max_positions: int | None = None,
        max_capital_pct: float | None = None,
    ) -> None:
        self.initial_capital = initial_capital or settings.initial_capital
        self.max_positions = max_positions or settings.max_positions
        self.max_capital_pct = max_capital_pct or settings.max_capital_per_trade_pct

        self._capital = self.initial_capital
        self._positions: dict[str, BacktestTrade] = {}
        self._closed_trades: list[BacktestTrade] = []
        self._equity_curve: list[float] = []
        self._dates: list[datetime] = []

    def run(
        self,
        data: dict[str, pd.DataFrame],
        strategy_fn: Callable[[str, pd.DataFrame, int], list[dict[str, Any]]],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BacktestResult:
        """Run backtest over historical data.

        Args:
            data: {symbol: DataFrame} with OHLCV, indexed by date.
            strategy_fn: (symbol, df_up_to_bar, bar_index) -> list of signals.
            start_date: Optional start filter (YYYY-MM-DD).
            end_date: Optional end filter (YYYY-MM-DD).

        Returns:
            BacktestResult with trades, equity curve.
        """
        self._capital = self.initial_capital
        self._positions = {}
        self._closed_trades = []
        self._equity_curve = [self.initial_capital]
        self._dates = []

        # Build unified date index
        all_dates: set[datetime] = set()
        for df in data.values():
            if "date" in df.columns:
                all_dates.update(pd.to_datetime(df["date"]).tolist())
            else:
                all_dates.update(df.index.tolist())
        sorted_dates = sorted(all_dates)

        if start_date:
            sorted_dates = [d for d in sorted_dates if d >= pd.Timestamp(start_date)]
        if end_date:
            sorted_dates = [d for d in sorted_dates if d <= pd.Timestamp(end_date)]

        logger.info(
            "[Backtest] Running {} symbols over {} days (₹{:,.0f} capital)",
            len(data), len(sorted_dates), self.initial_capital,
        )

        for i, date in enumerate(sorted_dates):
            # Check exits first for open positions
            self._check_exits(data, date)

            # Generate signals for each symbol
            for symbol, df in data.items():
                df_period = self._get_data_up_to(df, date)
                if df_period.empty:
                    continue

                bar_index = len(df_period) - 1
                signals = strategy_fn(symbol, df_period, bar_index)

                for signal in signals:
                    self._process_signal(signal, date)

            # Record equity
            equity = self._calculate_equity(data, date)
            self._equity_curve.append(equity)
            self._dates.append(date)

        # Close remaining positions at last available price
        self._close_all(data, sorted_dates[-1] if sorted_dates else datetime.now())

        result = BacktestResult(
            trades=self._closed_trades,
            equity_curve=self._equity_curve,
            dates=self._dates,
            initial_capital=self.initial_capital,
            final_capital=self._equity_curve[-1] if self._equity_curve else self.initial_capital,
        )

        logger.info(
            "[Backtest] Complete: {} trades, ₹{:,.0f} → ₹{:,.0f}",
            len(result.trades), result.initial_capital, result.final_capital,
        )
        return result

    def _get_data_up_to(self, df: pd.DataFrame, date: datetime) -> pd.DataFrame:
        if "date" in df.columns:
            return df[pd.to_datetime(df["date"]) <= date]
        return df[df.index <= date]

    def _process_signal(self, signal: dict[str, Any], date: datetime) -> None:
        symbol = signal.get("symbol", "")
        action = signal.get("action", "")
        price = signal.get("price", 0)

        if not symbol or not action or price <= 0:
            return

        # Skip if already in position
        if symbol in self._positions:
            return

        # Position limit
        if len(self._positions) >= self.max_positions:
            return

        # Capital per trade limit
        max_trade = self._capital * self.max_capital_pct
        qty = signal.get("quantity", 0)
        if qty <= 0:
            qty = int(max_trade / price)
        if qty <= 0:
            return

        trade_value = price * qty
        if trade_value > self._capital:
            qty = int(self._capital / price)
            if qty <= 0:
                return

        # Apply slippage
        slippage = price * settings.slippage_pct
        entry_price = price + slippage if action == "BUY" else price - slippage

        # Deduct capital
        cost_entry = self._calculate_costs(entry_price, qty, is_buy=True, is_intraday=signal.get("holding_period") == "INTRADAY")
        self._capital -= (entry_price * qty + cost_entry)

        trade = BacktestTrade(
            symbol=symbol,
            action=action,
            entry_date=date,
            entry_price=entry_price,
            quantity=qty,
            stop_loss=signal.get("stop_loss", 0),
            target=signal.get("target", 0),
            strategy=signal.get("strategy", "unknown"),
        )
        self._positions[symbol] = trade

    def _check_exits(self, data: dict[str, pd.DataFrame], date: datetime) -> None:
        for symbol in list(self._positions.keys()):
            trade = self._positions[symbol]
            df = data.get(symbol)
            if df is None:
                continue

            row = self._get_bar(df, date)
            if row is None:
                continue

            high = float(row.get("high", 0))
            low = float(row.get("low", 0))
            close = float(row.get("close", 0))

            exit_price = None
            reason = ""

            if trade.action == "BUY":
                if trade.stop_loss > 0 and low <= trade.stop_loss:
                    exit_price = trade.stop_loss
                    reason = "STOP_LOSS"
                elif trade.target > 0 and high >= trade.target:
                    exit_price = trade.target
                    reason = "TARGET"
            elif trade.action == "SELL":
                if trade.stop_loss > 0 and high >= trade.stop_loss:
                    exit_price = trade.stop_loss
                    reason = "STOP_LOSS"
                elif trade.target > 0 and low <= trade.target:
                    exit_price = trade.target
                    reason = "TARGET"

            if exit_price is not None:
                self._close_trade(symbol, exit_price, date, reason)

    def _close_trade(self, symbol: str, exit_price: float, date: datetime, reason: str) -> None:
        trade = self._positions.pop(symbol, None)
        if not trade:
            return

        # Apply slippage on exit
        slippage = exit_price * settings.slippage_pct
        exit_price = exit_price - slippage if trade.action == "BUY" else exit_price + slippage

        trade.exit_date = date
        trade.exit_price = exit_price
        trade.exit_reason = reason

        if trade.action == "BUY":
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity

        cost_exit = self._calculate_costs(exit_price, trade.quantity, is_buy=False, is_intraday=True)
        cost_entry = self._calculate_costs(trade.entry_price, trade.quantity, is_buy=True, is_intraday=True)
        trade.costs = cost_entry + cost_exit
        trade.net_pnl = trade.pnl - trade.costs

        # Return capital
        self._capital += exit_price * trade.quantity - cost_exit
        self._closed_trades.append(trade)

    def _close_all(self, data: dict[str, pd.DataFrame], date: datetime) -> None:
        for symbol in list(self._positions.keys()):
            df = data.get(symbol)
            if df is not None:
                row = self._get_bar(df, date)
                if row is not None:
                    self._close_trade(symbol, float(row["close"]), date, "BACKTEST_END")

    def _get_bar(self, df: pd.DataFrame, date: datetime) -> dict | None:
        if "date" in df.columns:
            mask = pd.to_datetime(df["date"]) == date
            if mask.any():
                return df[mask].iloc[-1].to_dict()
        else:
            if date in df.index:
                return df.loc[date].to_dict()
        return None

    def _calculate_equity(self, data: dict[str, pd.DataFrame], date: datetime) -> float:
        equity = self._capital
        for symbol, trade in self._positions.items():
            df = data.get(symbol)
            if df is not None:
                row = self._get_bar(df, date)
                if row:
                    equity += float(row["close"]) * trade.quantity
                else:
                    equity += trade.entry_price * trade.quantity
        return equity

    def _calculate_costs(self, price: float, qty: int, is_buy: bool, is_intraday: bool = False) -> float:
        """Calculate all Indian market transaction costs."""
        turnover = price * qty

        # Brokerage: ₹20 per order or 0.03% whichever is lower
        brokerage = min(settings.brokerage_per_order, turnover * 0.0003)

        # STT
        if is_buy and not is_intraday:
            stt = turnover * settings.stt_delivery_pct
        elif not is_buy:
            stt_rate = settings.stt_intraday_pct if is_intraday else settings.stt_delivery_pct
            stt = turnover * stt_rate
        else:
            stt = 0

        # Exchange transaction charges
        exchange = turnover * settings.exchange_charges_pct

        # GST on brokerage + exchange charges
        gst = (brokerage + exchange) * settings.gst_on_brokerage_pct

        # Stamp duty (only on buy)
        stamp = turnover * settings.stamp_duty_pct if is_buy else 0

        total = brokerage + stt + exchange + gst + stamp
        return total

    def run_walk_forward(
        self,
        data: dict[str, pd.DataFrame],
        strategy_fn: Callable[[str, pd.DataFrame, int], list[dict[str, Any]]],
        train_days: int = 252,
        test_days: int = 63,
        step_days: int = 63,
        optimize_fn: Callable[[dict[str, pd.DataFrame]], Callable] | None = None,
    ) -> WalkForwardResult:
        """Run anchored walk-forward validation.

        Splits data into rolling train/test windows. If optimize_fn is provided,
        it re-optimizes strategy_fn on each train window.

        Args:
            data: {symbol: DataFrame} with OHLCV, indexed by date.
            strategy_fn: Default strategy if optimize_fn is None.
            train_days: Training window size in trading days.
            test_days: Out-of-sample test window size.
            step_days: Step size between windows.
            optimize_fn: (train_data) -> optimized_strategy_fn. Optional.

        Returns:
            WalkForwardResult with per-fold results and aggregated metrics.
        """
        # Build unified sorted date index
        all_dates: set[datetime] = set()
        for df in data.values():
            if "date" in df.columns:
                all_dates.update(pd.to_datetime(df["date"]).tolist())
            else:
                all_dates.update(df.index.tolist())
        sorted_dates = sorted(all_dates)

        total_days = len(sorted_dates)
        min_required = train_days + test_days
        if total_days < min_required:
            logger.error(
                "[WalkForward] Need {} days, have {}. Cannot proceed.",
                min_required, total_days,
            )
            return WalkForwardResult(folds=[])

        folds: list[BacktestResult] = []
        fold_ranges: list[dict[str, str]] = []
        fold_idx = 0
        start = 0

        while start + train_days + test_days <= total_days:
            train_end = start + train_days
            test_end = min(train_end + test_days, total_days)

            train_start_date = sorted_dates[start]
            train_end_date = sorted_dates[train_end - 1]
            test_start_date = sorted_dates[train_end]
            test_end_date = sorted_dates[test_end - 1]

            logger.info(
                "[WalkForward] Fold {} — Train: {} to {}, Test: {} to {}",
                fold_idx,
                str(train_start_date)[:10], str(train_end_date)[:10],
                str(test_start_date)[:10], str(test_end_date)[:10],
            )

            # Slice data for train and test periods
            def _slice(df: pd.DataFrame, d_start: datetime, d_end: datetime) -> pd.DataFrame:
                if "date" in df.columns:
                    mask = (pd.to_datetime(df["date"]) >= d_start) & (pd.to_datetime(df["date"]) <= d_end)
                    return df[mask].copy()
                return df[(df.index >= d_start) & (df.index <= d_end)].copy()

            train_data = {sym: _slice(df, train_start_date, train_end_date) for sym, df in data.items()}
            train_data = {sym: df for sym, df in train_data.items() if not df.empty}

            test_data = {sym: _slice(df, test_start_date, test_end_date) for sym, df in data.items()}
            test_data = {sym: df for sym, df in test_data.items() if not df.empty}

            # Optimize on train window if optimizer provided
            if optimize_fn is not None:
                try:
                    current_strategy = optimize_fn(train_data)
                except Exception as exc:
                    logger.warning("[WalkForward] Fold {} optimize failed: {}, using default.", fold_idx, exc)
                    current_strategy = strategy_fn
            else:
                current_strategy = strategy_fn

            # Run backtest on OOS test window
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                max_positions=self.max_positions,
                max_capital_pct=self.max_capital_pct,
            )
            result = engine.run(test_data, current_strategy)
            folds.append(result)
            fold_ranges.append({
                "fold": fold_idx,
                "train_start": str(train_start_date)[:10],
                "train_end": str(train_end_date)[:10],
                "test_start": str(test_start_date)[:10],
                "test_end": str(test_end_date)[:10],
            })

            fold_idx += 1
            start += step_days

        # Aggregate OOS equity curves
        combined_equity = [self.initial_capital]
        combined_trades: list[Any] = []
        for f in folds:
            if f.equity_curve and len(f.equity_curve) > 1:
                scale = combined_equity[-1] / f.equity_curve[0] if f.equity_curve[0] > 0 else 1.0
                combined_equity.extend([v * scale for v in f.equity_curve[1:]])
            combined_trades.extend(f.trades)

        wf_result = WalkForwardResult(
            folds=folds,
            fold_ranges=fold_ranges,
            combined_equity=combined_equity,
            combined_trades=combined_trades,
            initial_capital=self.initial_capital,
            final_capital=combined_equity[-1] if combined_equity else self.initial_capital,
        )

        logger.info(
            "[WalkForward] {} folds, {} total OOS trades, ₹{:,.0f} → ₹{:,.0f}",
            len(folds), len(combined_trades), self.initial_capital, wf_result.final_capital,
        )
        return wf_result


@dataclass
class WalkForwardResult:
    """Result of walk-forward validation."""

    folds: list[BacktestResult] = field(default_factory=list)
    fold_ranges: list[dict[str, str]] = field(default_factory=list)
    combined_equity: list[float] = field(default_factory=list)
    combined_trades: list[Any] = field(default_factory=list)
    initial_capital: float = 0.0
    final_capital: float = 0.0

    @property
    def num_folds(self) -> int:
        return len(self.folds)

    @property
    def oos_return_pct(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return (self.final_capital - self.initial_capital) / self.initial_capital * 100

    def per_fold_returns(self) -> list[float]:
        """Return % returns for each fold."""
        returns = []
        for f in self.folds:
            if f.initial_capital > 0:
                returns.append((f.final_capital - f.initial_capital) / f.initial_capital * 100)
            else:
                returns.append(0.0)
        return returns
