"""Fundamental Agent — P/E, P/B, ROE, debt ratios, promoter holdings via LLM synthesis."""

from __future__ import annotations

import json
from typing import Any

import yfinance as yf
from loguru import logger

from agents.base_agent import BaseAgent, TradingState
from config.settings import settings

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None


class FundamentalAgent(BaseAgent):
    """Fetches fundamentals from yfinance/NSE and uses Claude Haiku to synthesise signals."""

    def __init__(self) -> None:
        super().__init__("FundamentalAgent")
        self._llm = None
        if ChatAnthropic and settings.anthropic_api_key and not settings.anthropic_api_key.startswith("your_"):
            self._llm = ChatAnthropic(
                model=settings.llm_fast_model,
                api_key=settings.anthropic_api_key,
                temperature=0.1,
                max_tokens=1024,
            )

    def run(self, state: TradingState) -> dict[str, Any]:
        logs: list[str] = []
        errors: list[str] = []
        signals: dict[str, Any] = {}

        watchlist = state.get("watchlist", [])

        for symbol in watchlist:
            try:
                fundamentals = self._fetch_fundamentals(symbol)
                if not fundamentals:
                    continue

                if self._llm:
                    signal = self._llm_synthesise(symbol, fundamentals)
                else:
                    signal = self._rule_based_signal(symbol, fundamentals)

                signals[symbol] = signal

                if signal.get("signal") != "HOLD":
                    logs.append(
                        self.log("{}: {} (confidence: {}%)", symbol, signal["signal"], signal.get("confidence", 0))
                    )

            except Exception as exc:
                errors.append(self.log("{}: Fundamental analysis error — {}", symbol, exc))

        logs.append(self.log("Fundamental analysis complete for {} symbols.", len(signals)))

        return {
            "fundamental_signals": signals,
            "agent_logs": logs,
            "errors": errors,
        }

    def _fetch_fundamentals(self, symbol: str) -> dict[str, Any] | None:
        """Fetch fundamental data from yfinance."""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info

            return {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "roce": info.get("returnOnAssets"),  # Approximation
                "debt_to_equity": info.get("debtToEquity"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "profit_margins": info.get("profitMargins"),
                "market_cap": info.get("marketCap"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "dividend_yield": info.get("dividendYield"),
                "book_value": info.get("bookValue"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "current_price": info.get("currentPrice"),
                "target_mean_price": info.get("targetMeanPrice"),
                "recommendation": info.get("recommendationKey"),
            }
        except Exception as exc:
            logger.debug("yfinance fundamentals failed for {}: {}", symbol, exc)
            return None

    def _llm_synthesise(self, symbol: str, data: dict) -> dict[str, Any]:
        """Use Claude Haiku to synthesise fundamental signals."""
        from agents.state_condenser import state_to_prompt_json

        context_json = state_to_prompt_json(data, max_chars=2000)
        prompt = f"""You are a fundamental analyst for Indian stocks (NSE/BSE).
Analyze {symbol} with these metrics:
{context_json}

Indian market context:
- SEBI regulations, GST impact, RBI rate decisions matter
- Promoter holding trends are more significant in India than Western markets
- Sector: {data.get('sector', 'Unknown')} — consider sector-specific India risks

Output ONLY valid JSON (no markdown):
{{"signal": "BUY/SELL/HOLD", "confidence": 0-100, "reasoning": "...", "key_risks": ["..."]}}"""

        try:
            resp = self._llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except Exception as exc:
            logger.error("LLM fundamental synthesis failed for {}: {}", symbol, exc)

        return self._rule_based_signal(symbol, data)

    def _rule_based_signal(self, symbol: str, data: dict) -> dict[str, Any]:
        """Simple rule-based fundamental scoring when LLM unavailable."""
        score = 0
        reasons = []

        pe = data.get("pe_ratio")
        if pe is not None:
            if pe < 15:
                score += 2
                reasons.append(f"Low P/E ({pe:.1f})")
            elif pe > 40:
                score -= 2
                reasons.append(f"High P/E ({pe:.1f})")

        roe = data.get("roe")
        if roe is not None:
            if roe > 0.15:
                score += 1
                reasons.append(f"Good ROE ({roe:.1%})")
            elif roe < 0.05:
                score -= 1
                reasons.append(f"Low ROE ({roe:.1%})")

        de = data.get("debt_to_equity")
        if de is not None:
            if de > 150:
                score -= 2
                reasons.append(f"High debt ({de:.0f})")
            elif de < 50:
                score += 1

        growth = data.get("revenue_growth")
        if growth is not None and growth > 0.10:
            score += 1
            reasons.append(f"Revenue growing ({growth:.1%})")

        if score >= 3:
            signal = "BUY"
            confidence = min(75, 45 + score * 5)
        elif score <= -3:
            signal = "SELL"
            confidence = min(75, 45 + abs(score) * 5)
        else:
            signal = "HOLD"
            confidence = 30

        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": " | ".join(reasons) if reasons else "Mixed fundamentals",
            "key_risks": [],
        }


def fundamental_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return FundamentalAgent().run(state)
