"""Technical Agent — RSI, MACD, EMA, Supertrend, ADX, Bollinger, VWAP analysis."""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent, TradingState
from data.indicators import compute_all_indicators, get_latest_signals


class TechnicalAgent(BaseAgent):
    """Computes technical indicators and generates signals for all watchlist symbols."""

    def __init__(self) -> None:
        super().__init__("TechnicalAgent")

    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []
        signals: dict[str, Any] = {}

        market_data = state.get("market_data", {})
        live_quotes = state.get("live_quotes", {})

        if not market_data:
            logs.append(self.log("No market data available. Skipping technical analysis."))
            return {"technical_signals": {}, "agent_logs": logs, "errors": errors}

        for symbol, df in market_data.items():
            try:
                if df.empty or len(df) < 30:
                    logs.append(self.log("{}: Insufficient data ({} rows).", symbol, len(df)))
                    continue

                # Compute all indicators
                df_with_indicators = compute_all_indicators(df)

                # Extract signals
                signal_data = get_latest_signals(df_with_indicators)

                # Add stop-loss and target using ATR
                atr = signal_data["indicators"].get("atr_14", 0)
                close = signal_data["indicators"].get("close", 0)

                if atr and close:
                    if signal_data["signal"] in ("BUY", "STRONG_BUY"):
                        signal_data["stop_loss"] = round(close - 1.5 * atr, 2)
                        signal_data["target"] = round(close + 2.0 * atr, 2)
                    elif signal_data["signal"] in ("SELL", "STRONG_SELL"):
                        signal_data["stop_loss"] = round(close + 1.5 * atr, 2)
                        signal_data["target"] = round(close - 2.0 * atr, 2)

                signals[symbol] = signal_data

                if signal_data["signal"] != "HOLD":
                    logs.append(
                        self.log(
                            "{}: {} (confidence: {}%) — {}",
                            symbol,
                            signal_data["signal"],
                            signal_data["confidence"],
                            signal_data["reasoning"],
                        )
                    )

            except Exception as exc:
                errors.append(self.log("{}: Technical analysis error — {}", symbol, exc))

        logs.append(self.log("Technical analysis complete for {} symbols.", len(signals)))

        return {
            "technical_signals": signals,
            "agent_logs": logs,
            "errors": errors,
        }


def technical_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return TechnicalAgent().run(state)
