"""Sentiment Agent — news NLP + social sentiment scoring via LLM."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from agents.base_agent import BaseAgent, TradingState
from config.settings import settings
from data.news_fetcher import fetch_all_news, format_news_for_prompt

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None


class SentimentAgent(BaseAgent):
    """Fetches news and uses Claude to score sentiment for each symbol."""

    def __init__(self) -> None:
        super().__init__("SentimentAgent")
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

        # Fetch general market news once
        try:
            loop = asyncio.new_event_loop()
            general_news = loop.run_until_complete(fetch_all_news())
            loop.close()
            logs.append(self.log("Fetched {} general market news articles.", len(general_news)))
        except Exception as exc:
            general_news = []
            errors.append(self.log("General news fetch failed: {}", exc))

        for symbol in watchlist:
            try:
                # Fetch symbol-specific news
                loop = asyncio.new_event_loop()
                symbol_news = loop.run_until_complete(fetch_all_news(symbol))
                loop.close()

                all_news = symbol_news or general_news[:5]

                if not all_news:
                    signals[symbol] = {"signal": "HOLD", "confidence": 0, "reasoning": "No news data available"}
                    continue

                news_text = format_news_for_prompt(all_news)

                if self._llm:
                    signal = self._llm_sentiment(symbol, news_text)
                else:
                    signal = {"signal": "HOLD", "confidence": 20, "reasoning": "LLM not configured for sentiment analysis"}

                signals[symbol] = signal

                if signal.get("signal") != "HOLD":
                    logs.append(
                        self.log("{}: {} (sentiment score: {})", symbol, signal["signal"], signal.get("sentiment_score", "N/A"))
                    )

            except Exception as exc:
                errors.append(self.log("{}: Sentiment analysis error — {}", symbol, exc))

        logs.append(self.log("Sentiment analysis complete for {} symbols.", len(signals)))

        return {
            "sentiment_signals": signals,
            "agent_logs": logs,
            "errors": errors,
        }

    def _llm_sentiment(self, symbol: str, news_text: str) -> dict[str, Any]:
        """Use Claude to score sentiment."""
        prompt = f"""Analyze sentiment for {symbol} from these Indian financial news articles.
Focus on: earnings announcements, management changes, regulatory issues,
          sector tailwinds/headwinds, FII interest, analyst upgrades/downgrades.

Articles:
{news_text}

Output ONLY valid JSON (no markdown):
{{"sentiment_score": -100, "signal": "BUY/SELL/HOLD", "confidence": 0-100, "key_events": ["..."], "reasoning": "..."}}

Where sentiment_score ranges from -100 (extremely negative) to +100 (extremely positive)."""

        try:
            resp = self._llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except Exception as exc:
            logger.error("LLM sentiment failed for {}: {}", symbol, exc)

        return {"signal": "HOLD", "confidence": 10, "reasoning": "LLM sentiment analysis failed"}


def sentiment_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node function."""
    return SentimentAgent().run(state)
