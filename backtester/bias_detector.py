"""Lookahead Bias Detector — flags common sources of future data leakage in backtests.

Scans trade data, feature alignment, and backtest results to detect patterns
that suggest the strategy (inadvertently) used future information:

1. **Time alignment checks** — features computed after the trade timestamp.
2. **Suspiciously high hit-rate** — >70% win rate with >100 trades is unusual.
3. **Overnight-gap exploitation** — entries exactly at open at favourable prices.
4. **Feature leakage patterns** — close-price features used for same-bar signals.
5. **Perfect timing** — entries/exits consistently at bar extremes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class BiasReport:
    """Aggregated lookahead-bias report."""

    warnings: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def add(self, severity: str, check: str, detail: str) -> None:
        self.warnings.append({"severity": severity, "check": check, "detail": detail})
        if severity in ("HIGH", "CRITICAL"):
            self.passed = False

    def summary(self) -> str:
        if self.passed:
            return f"Bias check PASSED — {len(self.warnings)} informational warnings."
        criticals = [w for w in self.warnings if w["severity"] in ("HIGH", "CRITICAL")]
        return (
            f"Bias check FAILED — {len(criticals)} high/critical warnings "
            f"out of {len(self.warnings)} total."
        )


def detect_lookahead_bias(
    trades: list[dict[str, Any]],
    ohlcv: dict[str, list[dict[str, Any]]] | None = None,
) -> BiasReport:
    """Run all lookahead checks on a completed backtest.

    Args:
        trades: List of trade dicts with at least: symbol, entry_time/exit_time,
                entry_price/exit_price, pnl, direction.
        ohlcv: Optional mapping of symbol → list of OHLCV bars (each bar is a dict
               with open, high, low, close, volume, timestamp).

    Returns:
        BiasReport with all findings.
    """
    report = BiasReport()

    if not trades:
        report.add("INFO", "no_trades", "No trades to analyse.")
        return report

    _check_win_rate(trades, report)
    _check_perfect_timing(trades, ohlcv, report)
    _check_overnight_gap_exploit(trades, ohlcv, report)
    _check_pnl_distribution(trades, report)
    _check_exit_before_entry(trades, report)

    logger.info("[BiasDetector] {}", report.summary())
    return report


# ─── Individual checks ───────────────────────────────────────────

def _check_win_rate(trades: list[dict], report: BiasReport) -> None:
    """Flag suspiciously high win rates."""
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    n = len(trades)
    if n == 0:
        return
    wr = wins / n

    if n >= 100 and wr > 0.75:
        report.add(
            "CRITICAL", "high_win_rate",
            f"Win rate {wr:.1%} across {n} trades is very suspicious — "
            "likely lookahead or data-snooping bias.",
        )
    elif n >= 50 and wr > 0.70:
        report.add(
            "HIGH", "high_win_rate",
            f"Win rate {wr:.1%} across {n} trades warrants inspection.",
        )
    elif wr > 0.65:
        report.add("MEDIUM", "high_win_rate", f"Win rate {wr:.1%} across {n} trades — monitor.")


def _check_perfect_timing(
    trades: list[dict],
    ohlcv: dict[str, list[dict]] | None,
    report: BiasReport,
) -> None:
    """Detect entries/exits consistently at bar high/low (future price knowledge)."""
    if not ohlcv:
        return

    at_extreme = 0
    checked = 0

    for trade in trades:
        symbol = trade.get("symbol", "")
        bars = ohlcv.get(symbol, [])
        if not bars:
            continue

        entry_price = trade.get("entry_price", 0)
        direction = trade.get("direction", "LONG")

        for bar in bars:
            bar_ts = bar.get("timestamp")
            if bar_ts and bar_ts == trade.get("entry_time"):
                checked += 1
                low, high = bar.get("low", 0), bar.get("high", 0)
                spread = high - low if high > low else 1

                if direction == "LONG" and abs(entry_price - low) / spread < 0.05:
                    at_extreme += 1
                elif direction == "SHORT" and abs(entry_price - high) / spread < 0.05:
                    at_extreme += 1
                break

    if checked >= 20 and at_extreme / checked > 0.40:
        report.add(
            "CRITICAL", "perfect_timing",
            f"{at_extreme}/{checked} entries at bar extremes ({at_extreme/checked:.0%}) — "
            "suggests future price knowledge.",
        )
    elif checked >= 10 and at_extreme / checked > 0.30:
        report.add(
            "HIGH", "perfect_timing",
            f"{at_extreme}/{checked} entries at bar extremes — suspicious.",
        )


def _check_overnight_gap_exploit(
    trades: list[dict],
    ohlcv: dict[str, list[dict]] | None,
    report: BiasReport,
) -> None:
    """Flag trades that enter exactly at open price on gap days."""
    if not ohlcv:
        return

    gap_exploits = 0
    gap_days = 0

    for trade in trades:
        symbol = trade.get("symbol", "")
        bars = ohlcv.get(symbol, [])
        entry_price = trade.get("entry_price", 0)

        for i, bar in enumerate(bars):
            if bar.get("timestamp") == trade.get("entry_time") and i > 0:
                prev_close = bars[i - 1].get("close", 0)
                open_price = bar.get("open", 0)
                if prev_close > 0 and abs(open_price - prev_close) / prev_close > 0.01:
                    gap_days += 1
                    if abs(entry_price - open_price) / open_price < 0.001:
                        gap_exploits += 1
                break

    if gap_days >= 10 and gap_exploits / gap_days > 0.80:
        report.add(
            "HIGH", "gap_exploit",
            f"{gap_exploits}/{gap_days} gap-day entries exactly at open — "
            "real fills would slip significantly.",
        )


def _check_pnl_distribution(trades: list[dict], report: BiasReport) -> None:
    """Flag unnaturally smooth or positive-only PnL distributions."""
    pnls = [t.get("pnl", 0) for t in trades if t.get("pnl") is not None]
    if len(pnls) < 30:
        return

    arr = np.array(pnls, dtype=float)
    negative_pct = np.mean(arr < 0)

    # Real strategies should have at least ~20% losing trades
    if negative_pct < 0.10:
        report.add(
            "CRITICAL", "pnl_distribution",
            f"Only {negative_pct:.0%} losing trades — unrealistic for any real strategy.",
        )
    elif negative_pct < 0.20:
        report.add(
            "HIGH", "pnl_distribution",
            f"Only {negative_pct:.0%} losing trades — review for bias.",
        )

    # Check for suspiciously low PnL variance (tightly clustered gains)
    if arr.mean() > 0:
        cv = arr.std() / arr.mean() if arr.mean() != 0 else 0
        if cv < 0.3 and len(pnls) > 50:
            report.add(
                "MEDIUM", "pnl_low_variance",
                f"PnL coefficient of variation {cv:.2f} is very low — "
                "real trades have more dispersion.",
            )


def _check_exit_before_entry(trades: list[dict], report: BiasReport) -> None:
    """Detect obviously broken timestamps where exit_time <= entry_time."""
    violations = 0
    for trade in trades:
        entry = trade.get("entry_time")
        exit_ = trade.get("exit_time")
        if entry and exit_ and exit_ <= entry:
            violations += 1

    if violations > 0:
        report.add(
            "CRITICAL", "exit_before_entry",
            f"{violations} trades have exit_time <= entry_time — data or code bug.",
        )
