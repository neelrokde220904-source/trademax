"""Pairs Trading Strategy — statistical arbitrage on co-integrated NSE pairs."""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from config.settings import settings
from data.cointegration import (
    NSE_PAIR_CANDIDATES,
    compute_spread_zscore,
    find_cointegrated_pairs,
)


class PairsTrading:
    """Mean-reversion pairs trading on co-integrated NSE stocks.

    ENTRY: Spread z-score crosses ±2.0 (expect reversion to mean)
    EXIT:  Spread z-score crosses 0 (reverted) OR stop at ±3.5
    UNIVERSE: Pre-screened NSE pair candidates (same sector)

    Each trade is market-neutral: LONG cheap leg, SHORT expensive leg.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.name = "PairsTrading"
        p = params or {}
        self.entry_zscore = p.get("entry_zscore", 2.0)
        self.exit_zscore = p.get("exit_zscore", 0.5)
        self.stop_zscore = p.get("stop_zscore", 3.5)
        self.lookback = p.get("lookback", 252)
        self.min_half_life = p.get("min_half_life", 5)
        self.max_half_life = p.get("max_half_life", 60)
        self._cointegrated_pairs: list[dict[str, Any]] = []
        self._last_calibration: str = ""

    def calibrate(self, data: dict[str, pd.DataFrame]) -> int:
        """Find co-integrated pairs from historical data.

        Returns:
            Number of valid pairs found.
        """
        self._cointegrated_pairs = find_cointegrated_pairs(
            data,
            candidates=NSE_PAIR_CANDIDATES,
            p_threshold=0.05,
            min_bars=self.lookback,
        )

        # Filter by half-life
        self._cointegrated_pairs = [
            p for p in self._cointegrated_pairs
            if p.get("half_life_days") is not None
            and self.min_half_life <= p["half_life_days"] <= self.max_half_life
        ]

        logger.info(
            "[PairsTrading] Calibrated: {} valid pairs (half-life {}-{} days).",
            len(self._cointegrated_pairs), self.min_half_life, self.max_half_life,
        )
        return len(self._cointegrated_pairs)

    def scan(self, market_data: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        """Scan calibrated pairs for entry/exit signals.

        Args:
            market_data: {symbol: DataFrame} with current OHLCV.

        Returns:
            List of pairs trade signals.
        """
        if not self._cointegrated_pairs:
            logger.debug("[PairsTrading] No calibrated pairs. Call calibrate() first.")
            return []

        signals: list[dict[str, Any]] = []

        for pair_info in self._cointegrated_pairs:
            sym_a, sym_b = pair_info["pair"]
            hedge_ratio = pair_info["hedge_ratio"]
            spread_mean = pair_info["spread_mean"]
            spread_std = pair_info["spread_std"]

            df_a = market_data.get(sym_a)
            df_b = market_data.get(sym_b)

            if df_a is None or df_b is None:
                continue
            if isinstance(df_a, pd.DataFrame) and df_a.empty:
                continue
            if isinstance(df_b, pd.DataFrame) and df_b.empty:
                continue

            try:
                price_a = float(df_a["close"].iloc[-1]) if isinstance(df_a, pd.DataFrame) else float(df_a.get("close", 0))
                price_b = float(df_b["close"].iloc[-1]) if isinstance(df_b, pd.DataFrame) else float(df_b.get("close", 0))
            except (IndexError, KeyError, TypeError):
                continue

            if price_a <= 0 or price_b <= 0:
                continue

            zscore = compute_spread_zscore(price_a, price_b, hedge_ratio, spread_mean, spread_std)

            signal = self._evaluate_zscore(sym_a, sym_b, zscore, price_a, price_b, hedge_ratio, pair_info)
            if signal:
                signals.append(signal)

        logger.info("[PairsTrading] {} signals from {} pairs.", len(signals), len(self._cointegrated_pairs))
        return signals

    def _evaluate_zscore(
        self, sym_a: str, sym_b: str, zscore: float,
        price_a: float, price_b: float, hedge_ratio: float,
        pair_info: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Generate trade signal based on z-score deviation."""
        half_life = pair_info.get("half_life_days", 20)
        confidence_base = max(40, min(80, int(70 - pair_info["p_value"] * 1000)))

        if zscore > self.entry_zscore:
            # Spread is HIGH — SHORT A, LONG B (expect A to fall relative to B)
            return {
                "pair": (sym_a, sym_b),
                "action": "SHORT_SPREAD",
                "leg_a": {"symbol": sym_a, "action": "SELL", "price": price_a},
                "leg_b": {"symbol": sym_b, "action": "BUY", "price": price_b},
                "hedge_ratio": hedge_ratio,
                "zscore": round(zscore, 3),
                "expected_reversion_days": half_life,
                "strategy": self.name,
                "confidence": confidence_base + min(10, int((zscore - self.entry_zscore) * 5)),
                "reasoning": (
                    f"Spread z={zscore:.2f} > {self.entry_zscore} threshold. "
                    f"Short {sym_a}, Long {sym_b}. "
                    f"Half-life: {half_life:.0f} days. p={pair_info['p_value']:.4f}"
                ),
                "holding_period": "SWING",
            }

        elif zscore < -self.entry_zscore:
            # Spread is LOW — LONG A, SHORT B (expect A to rise relative to B)
            return {
                "pair": (sym_a, sym_b),
                "action": "LONG_SPREAD",
                "leg_a": {"symbol": sym_a, "action": "BUY", "price": price_a},
                "leg_b": {"symbol": sym_b, "action": "SELL", "price": price_b},
                "hedge_ratio": hedge_ratio,
                "zscore": round(zscore, 3),
                "expected_reversion_days": half_life,
                "strategy": self.name,
                "confidence": confidence_base + min(10, int((abs(zscore) - self.entry_zscore) * 5)),
                "reasoning": (
                    f"Spread z={zscore:.2f} < -{self.entry_zscore} threshold. "
                    f"Long {sym_a}, Short {sym_b}. "
                    f"Half-life: {half_life:.0f} days. p={pair_info['p_value']:.4f}"
                ),
                "holding_period": "SWING",
            }

        return None

    def get_calibrated_pairs(self) -> list[dict[str, Any]]:
        """Return list of currently calibrated pairs with stats."""
        return self._cointegrated_pairs
