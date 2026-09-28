"""Backtester Report Generator — text + JSON summary of backtest results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from backtester.engine import BacktestResult
from backtester.metrics import PerformanceMetrics, calculate_metrics


def generate_report(result: BacktestResult, output_dir: str | None = None) -> dict[str, Any]:
    """Generate a comprehensive backtest report.

    Returns:
        Dict with all metrics + trade list.
    """
    metrics = calculate_metrics(result)

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return_pct": round(metrics.total_return_pct, 2),
            "cagr_pct": round(metrics.cagr, 2),
            "net_pnl": round(metrics.net_pnl, 2),
            "total_costs": round(metrics.total_costs, 2),
        },
        "trades": {
            "total": metrics.total_trades,
            "winners": metrics.winning_trades,
            "losers": metrics.losing_trades,
            "win_rate_pct": round(metrics.win_rate * 100, 2),
            "profit_factor": round(metrics.profit_factor, 2),
            "avg_pnl": round(metrics.avg_trade_pnl, 2),
            "avg_win": round(metrics.avg_win, 2),
            "avg_loss": round(metrics.avg_loss, 2),
            "max_win": round(metrics.max_win, 2),
            "max_loss": round(metrics.max_loss, 2),
            "avg_holding_days": round(metrics.avg_holding_days, 1),
            "expectancy": round(metrics.expectancy, 2),
        },
        "risk": {
            "max_drawdown": round(metrics.max_drawdown, 2),
            "max_drawdown_pct": round(metrics.max_drawdown_pct, 2),
            "sharpe_ratio": round(metrics.sharpe_ratio, 3),
            "sortino_ratio": round(metrics.sortino_ratio, 3),
            "calmar_ratio": round(metrics.calmar_ratio, 3),
            "risk_free_rate_pct": round(metrics.risk_free_rate * 100, 2),
        },
        "trade_log": [
            {
                "symbol": t.symbol,
                "action": t.action,
                "strategy": t.strategy,
                "entry_date": t.entry_date.isoformat() if t.entry_date else None,
                "entry_price": round(t.entry_price, 2),
                "exit_date": t.exit_date.isoformat() if t.exit_date else None,
                "exit_price": round(t.exit_price, 2) if t.exit_price else None,
                "quantity": t.quantity,
                "pnl": round(t.pnl, 2),
                "costs": round(t.costs, 2),
                "net_pnl": round(t.net_pnl, 2),
                "exit_reason": t.exit_reason,
            }
            for t in result.trades
        ],
    }

    # Save to file if output_dir specified
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = out / f"backtest_{ts}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("[Report] Saved JSON: {}", json_path)

        txt_path = out / f"backtest_{ts}.txt"
        with open(txt_path, "w") as f:
            f.write(format_text_report(report))
        logger.info("[Report] Saved TXT: {}", txt_path)

    return report


def format_text_report(report: dict) -> str:
    """Format report as human-readable text."""
    s = report["summary"]
    t = report["trades"]
    r = report["risk"]

    lines = [
        "=" * 60,
        "  INDIA AI TRADER — BACKTEST REPORT",
        "=" * 60,
        f"  Generated: {report['generated_at']}",
        "",
        "--- SUMMARY ---",
        f"  Initial Capital:  ₹{s['initial_capital']:>12,.2f}",
        f"  Final Capital:    ₹{s['final_capital']:>12,.2f}",
        f"  Net PnL:          ₹{s['net_pnl']:>12,.2f}",
        f"  Total Costs:      ₹{s['total_costs']:>12,.2f}",
        f"  Total Return:      {s['total_return_pct']:>10.2f}%",
        f"  CAGR:              {s['cagr_pct']:>10.2f}%",
        "",
        "--- TRADES ---",
        f"  Total Trades:       {t['total']:>8}",
        f"  Winners / Losers:   {t['winners']} / {t['losers']}",
        f"  Win Rate:           {t['win_rate_pct']:>8.1f}%",
        f"  Profit Factor:      {t['profit_factor']:>8.2f}",
        f"  Avg PnL:           ₹{t['avg_pnl']:>10,.2f}",
        f"  Avg Win:           ₹{t['avg_win']:>10,.2f}",
        f"  Avg Loss:          ₹{t['avg_loss']:>10,.2f}",
        f"  Max Win:           ₹{t['max_win']:>10,.2f}",
        f"  Max Loss:          ₹{t['max_loss']:>10,.2f}",
        f"  Avg Holding:        {t['avg_holding_days']:>6.1f} days",
        f"  Expectancy:        ₹{t['expectancy']:>10,.2f}",
        "",
        "--- RISK ---",
        f"  Max Drawdown:      ₹{r['max_drawdown']:>10,.2f}",
        f"  Max Drawdown %:     {r['max_drawdown_pct']:>8.2f}%",
        f"  Sharpe Ratio:       {r['sharpe_ratio']:>8.3f}",
        f"  Sortino Ratio:      {r['sortino_ratio']:>8.3f}",
        f"  Calmar Ratio:       {r['calmar_ratio']:>8.3f}",
        f"  Risk-Free Rate:     {r['risk_free_rate_pct']:>8.2f}%",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)
