"""Risk Manager Agent — Kelly Criterion, VaR, drawdown, position limits, correlation check."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from agents.base_agent import BaseAgent, TradingState
from config.settings import settings


class RiskManagerAgent(BaseAgent):
    """Evaluates risk for proposed trades and enforces hard limits."""

    def __init__(self) -> None:
        super().__init__("RiskManagerAgent")

    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []

        # Gather all signals
        tech = state.get("technical_signals", {})
        fund = state.get("fundamental_signals", {})
        sent = state.get("sentiment_signals", {})
        opts = state.get("options_signals", {})
        macro = state.get("macro_signals", {})
        india = state.get("india_signals", {})

        india_vix = state.get("india_vix", 0)
        portfolio_value = state.get("portfolio_value", settings.initial_capital)
        available_capital = state.get("available_capital", settings.initial_capital)
        current_positions = state.get("current_positions", {})
        market_data = state.get("market_data", {})

        # --- Global Risk Checks ---
        # Daily loss limit
        from broker.portfolio_tracker import portfolio_tracker
        if portfolio_tracker.has_hit_daily_loss_limit():
            logs.append(self.log("⛔ DAILY LOSS LIMIT HIT — blocking all new trades."))
            return {
                "risk_assessment": {"blocked": True, "reason": "Daily loss limit exceeded"},
                "agent_logs": logs,
                "errors": errors,
            }

        # Max positions
        if len(current_positions) >= settings.max_positions:
            logs.append(self.log("⛔ MAX POSITIONS ({}) reached — no new entries.", settings.max_positions))
            return {
                "risk_assessment": {"blocked": True, "reason": f"Max positions ({settings.max_positions}) reached"},
                "agent_logs": logs,
                "errors": errors,
            }

        # VIX check
        vix_factor = 1.0
        if india_vix > settings.vix_halt_threshold:
            logs.append(self.log("⛔ VIX ({:.1f}) > halt threshold ({}) — halting trades.", india_vix, settings.vix_halt_threshold))
            return {
                "risk_assessment": {"blocked": True, "reason": f"VIX too high: {india_vix}"},
                "agent_logs": logs,
                "errors": errors,
            }
        elif india_vix > settings.max_nifty_vix:
            vix_factor = 0.5
            logs.append(self.log("⚠️ VIX ({:.1f}) elevated — reducing position sizes by 50%.", india_vix))

        # --- Per-Symbol Risk Assessment ---
        risk_assessment: dict[str, Any] = {"blocked": False}

        # Merge all symbols that have any signal
        all_symbols = set()
        for sigs in [tech, fund, sent]:
            all_symbols.update(sigs.keys())

        for symbol in all_symbols:
            try:
                # Collect signals
                t_sig = tech.get(symbol, {})
                f_sig = fund.get(symbol, {})
                s_sig = sent.get(symbol, {})

                # Skip if India-specific says AVOID
                i_sig = india.get(symbol, {})
                if i_sig.get("signal") == "AVOID":
                    risk_assessment[symbol] = {
                        "approved": False,
                        "rejection_reason": i_sig.get("reasoning", "India-specific block"),
                    }
                    continue

                # Only evaluate symbols with actionable signals
                action_signals = [
                    s for s in [t_sig.get("signal"), f_sig.get("signal"), s_sig.get("signal")]
                    if s and s not in ("HOLD", None)
                ]
                if not action_signals:
                    continue

                # Position sizing (Kelly-based)
                max_position_value = self._kelly_position_size(
                    available_capital, vix_factor
                )

                # VaR calculation
                df = market_data.get(symbol, pd.DataFrame())
                var_1day = self._calculate_var(df) if not df.empty else 0.02

                # Check VaR limit
                if var_1day > 0.02:
                    risk_assessment[symbol] = {
                        "approved": False,
                        "rejection_reason": f"VaR too high: {var_1day:.4f} (limit: 2%)",
                        "var_1day": var_1day,
                    }
                    logs.append(self.log("{}: REJECTED — VaR {:.2%} > 2%", symbol, var_1day))
                    continue

                # Correlation check with existing positions
                if self._is_too_correlated(symbol, current_positions, market_data):
                    risk_assessment[symbol] = {
                        "approved": False,
                        "rejection_reason": "Too correlated with existing position (>0.8)",
                    }
                    logs.append(self.log("{}: REJECTED — high correlation", symbol))
                    continue

                # Stop-loss from technical signal
                stop_loss = t_sig.get("stop_loss", 0)
                target = t_sig.get("target", 0)
                close = t_sig.get("indicators", {}).get("close", 0)
                atr = t_sig.get("indicators", {}).get("atr_14", 0)

                if not stop_loss and close and atr:
                    stop_loss = round(close - settings.stop_loss_atr_multiplier * atr, 2)

                # Calculate max quantity
                max_qty = 0
                if close and close > 0:
                    max_qty = int(max_position_value / close)

                # Risk score (0-100, lower is better)
                risk_score = self._compute_risk_score(var_1day, india_vix, len(current_positions))

                risk_assessment[symbol] = {
                    "approved": True,
                    "max_quantity": max_qty,
                    "max_position_value": round(max_position_value, 2),
                    "stop_loss": stop_loss,
                    "target": target,
                    "risk_score": risk_score,
                    "var_1day": round(var_1day, 4),
                    "vix_factor": vix_factor,
                    "rejection_reason": None,
                }

                logs.append(
                    self.log(
                        "{}: APPROVED — max qty {}, SL ₹{}, risk score {}/100",
                        symbol, max_qty, stop_loss, risk_score,
                    )
                )

            except Exception as exc:
                errors.append(self.log("{}: Risk assessment error — {}", symbol, exc))

        return {
            "risk_assessment": risk_assessment,
            "agent_logs": logs,
            "errors": errors,
        }

    def _kelly_position_size(self, capital: float, vix_factor: float) -> float:
        """Fractional Kelly position sizing (25% Kelly)."""
        # Default assumptions — updated with actual track record over time
        win_rate = 0.55
        avg_win = 0.02  # 2% average win
        avg_loss = 0.015  # 1.5% average loss
        kelly_fraction = 0.25

        if avg_loss == 0:
            return capital * settings.max_capital_per_trade_pct * vix_factor

        kelly = (win_rate / avg_loss) - ((1 - win_rate) / avg_win)
        safe_kelly = kelly * kelly_fraction

        max_position = capital * max(0, min(safe_kelly, settings.max_capital_per_trade_pct))
        return max_position * vix_factor

    def _calculate_var(self, df: pd.DataFrame, confidence: float = 0.95) -> float:
        """Historical VaR at given confidence level."""
        if df.empty or len(df) < 20:
            return 0.02  # Conservative default

        returns = df["close"].pct_change().dropna()
        if returns.empty:
            return 0.02

        var = float(np.percentile(returns, (1 - confidence) * 100))
        return abs(var)

    def _is_too_correlated(
        self,
        symbol: str,
        positions: dict[str, Any],
        market_data: dict[str, Any],
        threshold: float = 0.8,
    ) -> bool:
        """Check if symbol is too correlated with existing positions."""
        if not positions:
            return False

        sym_df = market_data.get(symbol, pd.DataFrame())
        if sym_df.empty or len(sym_df) < 20:
            return False

        sym_returns = sym_df["close"].pct_change().dropna()

        for pos_symbol in positions:
            pos_df = market_data.get(pos_symbol, pd.DataFrame())
            if pos_df.empty or len(pos_df) < 20:
                continue

            pos_returns = pos_df["close"].pct_change().dropna()

            # Align lengths
            min_len = min(len(sym_returns), len(pos_returns))
            if min_len < 10:
                continue

            corr = np.corrcoef(
                sym_returns.values[-min_len:],
                pos_returns.values[-min_len:],
            )[0, 1]

            if abs(corr) > threshold:
                return True

        return False

    def _compute_risk_score(self, var: float, vix: float, n_positions: int) -> int:
        """Compute a 0–100 risk score (lower = safer)."""
        score = 0
        score += int(var * 1000)  # VaR contribution
        score += int(vix * 1.5)  # VIX contribution
        score += n_positions * 5  # Position count contribution
        return min(100, max(0, score))


def risk_manager_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return RiskManagerAgent().run(state)
