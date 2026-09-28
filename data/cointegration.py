"""Cointegration Analysis — find co-integrated NSE stock pairs for pairs trading."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    from statsmodels.tsa.stattools import coint, adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# Well-known NSE pair candidates (same sector / correlated businesses)
NSE_PAIR_CANDIDATES: list[tuple[str, str]] = [
    # Banking
    ("HDFCBANK", "ICICIBANK"),
    ("SBIN", "AXISBANK"),
    ("KOTAKBANK", "HDFCBANK"),
    ("BAJFINANCE", "BAJAJFINSV"),
    # IT
    ("TCS", "INFY"),
    ("WIPRO", "HCLTECH"),
    ("INFY", "HCLTECH"),
    # Auto
    ("MARUTI", "TATAMOTORS"),
    # FMCG
    ("HINDUNILVR", "ITC"),
    # Metals
    ("TATASTEEL", "HINDALCO"),
    # Pharma
    ("SUNPHARMA", "DRREDDY"),
    # Energy
    ("RELIANCE", "ONGC"),
    ("BPCL", "IOC"),
    # Telecom
    ("BHARTIARTL", "IDEA"),
]


def find_cointegrated_pairs(
    data: dict[str, pd.DataFrame],
    candidates: list[tuple[str, str]] | None = None,
    p_threshold: float = 0.05,
    min_bars: int = 100,
) -> list[dict[str, Any]]:
    """Find co-integrated stock pairs from historical close prices.

    Args:
        data: {symbol: DataFrame} with 'close' column.
        candidates: Specific pairs to test. If None, test all combinations.
        p_threshold: p-value threshold for Engle-Granger test.
        min_bars: Minimum overlapping bars required.

    Returns:
        List of cointegrated pair results sorted by p-value.
    """
    if not STATSMODELS_AVAILABLE:
        logger.error("[Cointegration] statsmodels not installed. Run: pip install statsmodels")
        return []

    # Extract close price series
    close_data: dict[str, pd.Series] = {}
    for symbol, df in data.items():
        if isinstance(df, pd.DataFrame) and "close" in df.columns and len(df) >= min_bars:
            s = df["close"].dropna()
            if len(s) >= min_bars:
                close_data[symbol] = s

    if len(close_data) < 2:
        logger.warning("[Cointegration] Need >= 2 symbols with {} bars, got {}.", min_bars, len(close_data))
        return []

    # Determine pairs to test
    if candidates:
        pairs = [(a, b) for a, b in candidates if a in close_data and b in close_data]
    else:
        pairs = list(combinations(close_data.keys(), 2))

    results: list[dict[str, Any]] = []

    for sym_a, sym_b in pairs:
        try:
            s_a = close_data[sym_a]
            s_b = close_data[sym_b]

            # Align on common index
            aligned = pd.DataFrame({"a": s_a, "b": s_b}).dropna()
            if len(aligned) < min_bars:
                continue

            # Engle-Granger cointegration test
            score, pvalue, _ = coint(aligned["a"].values, aligned["b"].values)

            if pvalue < p_threshold:
                # Compute hedge ratio via OLS
                hedge_ratio = float(np.polyfit(aligned["b"].values, aligned["a"].values, 1)[0])

                # Compute spread
                spread = aligned["a"] - hedge_ratio * aligned["b"]
                spread_mean = float(spread.mean())
                spread_std = float(spread.std())
                current_zscore = float((spread.iloc[-1] - spread_mean) / spread_std) if spread_std > 0 else 0.0

                # Half-life of mean reversion (Ornstein-Uhlenbeck)
                spread_lag = spread.shift(1).dropna()
                spread_ret = spread.iloc[1:].values - spread_lag.values
                beta_ou = float(np.polyfit(spread_lag.values, spread_ret, 1)[0])
                half_life = -np.log(2) / beta_ou if beta_ou < 0 else float("inf")

                results.append({
                    "pair": (sym_a, sym_b),
                    "p_value": round(pvalue, 6),
                    "test_statistic": round(score, 4),
                    "hedge_ratio": round(hedge_ratio, 4),
                    "spread_mean": round(spread_mean, 2),
                    "spread_std": round(spread_std, 2),
                    "current_zscore": round(current_zscore, 4),
                    "half_life_days": round(half_life, 1) if half_life != float("inf") else None,
                    "n_bars": len(aligned),
                })

        except Exception as exc:
            logger.debug("[Cointegration] Error testing {}-{}: {}", sym_a, sym_b, exc)

    results.sort(key=lambda x: x["p_value"])
    logger.info("[Cointegration] Found {} co-integrated pairs from {} tested.", len(results), len(pairs))
    return results


def compute_spread_zscore(
    price_a: float, price_b: float,
    hedge_ratio: float, spread_mean: float, spread_std: float,
) -> float:
    """Compute the current z-score of the spread (for live trading signals)."""
    if spread_std <= 0:
        return 0.0
    spread = price_a - hedge_ratio * price_b
    return (spread - spread_mean) / spread_std
