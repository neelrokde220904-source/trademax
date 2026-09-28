"""Portfolio Manager Agent — final decision aggregator using Claude Opus."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytz
from loguru import logger

from agents.base_agent import BaseAgent, TradingState
from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None


class PortfolioManagerAgent(BaseAgent):
    """Final decision-making node — aggregates all signals and generates trade orders.

    Uses Claude Opus (most capable model) for the final decision.
    """

    def __init__(self) -> None:
        super().__init__("PortfolioManagerAgent")
        self._llm = None
        if ChatAnthropic and settings.anthropic_api_key and not settings.anthropic_api_key.startswith("your_"):
            self._llm = ChatAnthropic(
                model=settings.llm_decision_model,
                api_key=settings.anthropic_api_key,
                temperature=0.2,
                max_tokens=4096,
            )

    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []

        risk = state.get("risk_assessment", {})

        # If globally blocked, no decisions
        if risk.get("blocked"):
            logs.append(self.log("Trading blocked: {}", risk.get("reason", "Unknown")))
            return {"trading_decisions": [], "agent_logs": logs, "errors": errors}

        # Check time window
        now = datetime.now(IST)
        market_open = now.replace(hour=9, minute=15, second=0)
        no_trade_start = now.replace(hour=9, minute=15 + settings.no_trade_window_open_minutes, second=0)
        no_trade_end = now.replace(hour=15, minute=30 - settings.no_trade_window_close_minutes, second=0)

        if now < no_trade_start:
            logs.append(self.log("In opening no-trade window (first {} mins).", settings.no_trade_window_open_minutes))
            return {"trading_decisions": [], "agent_logs": logs, "errors": errors}

        if now > no_trade_end:
            logs.append(self.log("In closing no-trade window (last {} mins).", settings.no_trade_window_close_minutes))
            return {"trading_decisions": [], "agent_logs": logs, "errors": errors}

        # Gather approved symbols
        approved = {
            sym: data for sym, data in risk.items()
            if isinstance(data, dict) and data.get("approved")
        }

        if not approved:
            logs.append(self.log("No risk-approved trades available."))
            return {"trading_decisions": [], "agent_logs": logs, "errors": errors}

        # Build decision
        if self._llm:
            decisions = self._llm_decide(state, approved)
        else:
            decisions = self._rule_based_decide(state, approved)

        # Filter out any F&O banned stocks
        fo_ban = state.get("india_signals", {}).get("fo_ban_list", [])
        decisions = [d for d in decisions if d.get("symbol") not in fo_ban]

        logs.append(self.log("Generated {} trading decisions.", len(decisions)))
        for d in decisions:
            logs.append(
                self.log(
                    "Decision: {} {} × {} @ ₹{} (conf: {}%)",
                    d["action"], d.get("quantity", 0), d["symbol"],
                    d.get("price", 0), d.get("confidence", 0),
                )
            )

        return {
            "trading_decisions": decisions,
            "agent_logs": logs,
            "errors": errors,
        }

    def _llm_decide(self, state: TradingState, approved: dict) -> list[dict[str, Any]]:
        """Use Claude Opus for final decision making."""
        india_vix = state.get("india_vix", 0)
        is_expiry = state.get("india_signals", {}).get("is_expiry_day", False)
        fo_ban = state.get("india_signals", {}).get("fo_ban_list", [])

        prompt = f"""You are the Portfolio Manager of an AI hedge fund trading NSE/BSE Indian markets.
Today is {state.get('date')}. Current time: {datetime.now(IST).strftime('%H:%M IST')}.
India VIX: {india_vix} ({'HIGH RISK - reduce size' if india_vix > 20 else 'NORMAL'})

Current portfolio value: ₹{state.get('portfolio_value', 0):,.2f}
Available capital: ₹{state.get('available_capital', 0):,.2f}
Current positions: {json.dumps(state.get('current_positions', {}), default=str)[:500]}

ANALYST SIGNALS:
Technical: {json.dumps(state.get('technical_signals', {}), default=str)[:1500]}
Fundamental: {json.dumps(state.get('fundamental_signals', {}), default=str)[:1000]}
Sentiment: {json.dumps(state.get('sentiment_signals', {}), default=str)[:1000]}
Options: {json.dumps(state.get('options_signals', {}), default=str)[:800]}
Macro: {json.dumps(state.get('macro_signals', {}), default=str)[:800]}

RISK-APPROVED TRADES:
{json.dumps(approved, default=str)[:1500]}

INDIA-SPECIFIC RULES:
1. NEVER trade stocks in F&O ban list: {fo_ban}
2. On F&O expiry (today: {'YES' if is_expiry else 'NO'}), expect higher volatility
3. Prefer liquid large-caps for intraday
4. Be conservative — missing a trade is better than a bad trade

Output ONLY valid JSON (no markdown, no explanation):
{{"decisions": [{{"symbol": "XXX", "action": "BUY/SELL", "quantity": N, "order_type": "LIMIT/MARKET", "price": X.XX, "stop_loss": X.XX, "target": X.XX, "holding_period": "INTRADAY/DELIVERY", "confidence": 0-100, "reasoning": "..."}}], "market_view": "BULLISH/BEARISH/NEUTRAL", "session_notes": "..."}}"""

        try:
            resp = self._llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
                return result.get("decisions", [])
        except Exception as exc:
            logger.exception("LLM decision failed: {}", exc)

        return self._rule_based_decide(state, approved)

    def _rule_based_decide(self, state: TradingState, approved: dict) -> list[dict[str, Any]]:
        """Fallback rule-based decision making."""
        decisions = []
        tech = state.get("technical_signals", {})
        fund = state.get("fundamental_signals", {})

        for symbol, risk_data in approved.items():
            t_sig = tech.get(symbol, {})
            f_sig = fund.get(symbol, {})

            signal = t_sig.get("signal", "HOLD")
            confidence = t_sig.get("confidence", 0)

            # Boost confidence if fundamental agrees
            if f_sig.get("signal") == signal:
                confidence = min(95, confidence + 15)

            if signal in ("BUY", "STRONG_BUY") and confidence >= 60:
                close = t_sig.get("indicators", {}).get("close", 0)
                decisions.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": min(risk_data.get("max_quantity", 1), 10),
                    "order_type": "LIMIT",
                    "price": close,
                    "stop_loss": risk_data.get("stop_loss", 0),
                    "target": risk_data.get("target", 0),
                    "holding_period": "INTRADAY",
                    "confidence": confidence,
                    "reasoning": t_sig.get("reasoning", "Technical signal"),
                })
            elif signal in ("SELL", "STRONG_SELL") and confidence >= 60:
                close = t_sig.get("indicators", {}).get("close", 0)
                # Only sell if we hold the stock
                if symbol in state.get("current_positions", {}):
                    decisions.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "quantity": min(risk_data.get("max_quantity", 1), 10),
                        "order_type": "LIMIT",
                        "price": close,
                        "stop_loss": risk_data.get("stop_loss", 0),
                        "target": risk_data.get("target", 0),
                        "holding_period": "INTRADAY",
                        "confidence": confidence,
                        "reasoning": t_sig.get("reasoning", "Technical signal"),
                    })

        return decisions


def portfolio_manager_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return PortfolioManagerAgent().run(state)
