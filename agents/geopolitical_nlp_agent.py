"""Geopolitical NLP Agent — monitors geopolitical developments and maps them to
second/third-order impacts on Indian equities via supply chain graphs.

India-specific geopolitical risks:
- India-China border tensions → Defence sector UP, China-dependent FMCG DOWN
- US tariffs on India → IT exports DOWN (if targeted), pharma concerns
- Middle East escalation → Crude oil UP → OMCs DOWN, aviation DOWN, paints DOWN
- Russia-Ukraine → fertilizer prices UP → agri input costs UP → agrochemicals DOWN
- Taiwan tensions → semiconductor supply → electronics UP (domestic manufacturing)
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from agents.base_agent import TradingState
from config.settings import settings

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Supply chain knowledge graph — maps geopolitical events to market impacts
SUPPLY_CHAIN_KNOWLEDGE_GRAPH: dict[str, dict[str, Any]] = {
    "crude_oil_price_increase": {
        "direct_negative": ["BPCL", "HINDPETRO", "IOC", "INDIGO", "SPICEJET"],
        "direct_positive": ["ONGC", "OIL", "RELIANCE", "GAIL"],
        "indirect_negative": ["ASIANPAINT", "BERGEPAINT", "PIDILITIND"],
        "hedge": "BUY ONGC, SHORT OMCs (BPCL, HINDPETRO)",
        "confidence": 0.85,
    },
    "china_supply_disruption": {
        "direct_negative": ["HAVELLS", "VOLTAS", "BLUESTARLTD"],
        "direct_positive": ["DIXON", "AMBER", "KAYNES"],
        "indirect_positive": ["TATAELXSI", "LTTS"],
        "hedge": "BUY domestic EMS plays (DIXON, AMBER)",
        "confidence": 0.75,
    },
    "us_tariffs_on_india": {
        "direct_negative": ["TCS", "INFY", "WIPRO", "HCLTECH"],
        "direct_positive": [],
        "indirect_positive": ["DOMESTIC_CONSUMPTION"],
        "hedge": "Reduce IT exposure, increase domestic consumption",
        "confidence": 0.70,
    },
    "middle_east_escalation": {
        "direct_negative": ["BPCL", "HINDPETRO", "IOC", "INDIGO", "SPICEJET", "BHARATFORG"],
        "direct_positive": ["ONGC", "OIL", "HAL", "BEL", "BDL"],
        "indirect_negative": ["ASIANPAINT", "BERGEPAINT"],
        "hedge": "BUY defence (HAL, BEL), hedge crude exposure",
        "confidence": 0.80,
    },
    "russia_ukraine_escalation": {
        "direct_negative": ["CHAMBAL", "COROMANDEL", "UPL"],
        "direct_positive": ["HAL", "BEL", "BDL"],
        "indirect_negative": ["TATASTEEL", "HINDALCO"],
        "hedge": "Avoid fertilizer/metals, BUY defence",
        "confidence": 0.70,
    },
    "taiwan_tensions": {
        "direct_negative": ["VEDL"],
        "direct_positive": ["DIXON", "KAYNES", "TATAELXSI"],
        "indirect_positive": ["RANEENGINEERING"],
        "hedge": "BUY domestic semiconductor/EMS",
        "confidence": 0.65,
    },
    "rbi_rate_cut": {
        "direct_positive": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
                            "BAJFINANCE", "BAJAJFINSV", "HDFC"],
        "direct_negative": [],
        "indirect_positive": ["GODREJPROP", "DLF", "OBEROIRLTY"],
        "hedge": "BUY banking + real estate on rate cuts",
        "confidence": 0.90,
    },
    "rbi_rate_hike": {
        "direct_negative": ["BAJFINANCE", "BAJAJFINSV", "LICHSGFIN"],
        "direct_positive": [],
        "indirect_negative": ["GODREJPROP", "DLF", "OBEROIRLTY"],
        "hedge": "Reduce NBFC exposure on rate hikes",
        "confidence": 0.85,
    },
    "india_pakistan_tensions": {
        "direct_positive": ["HAL", "BEL", "BDL", "BHEL"],
        "direct_negative": [],
        "indirect_negative": ["NIFTY_BROAD"],
        "hedge": "BUY defence, reduce broad exposure temporarily",
        "confidence": 0.75,
    },
    "global_recession_fears": {
        "direct_negative": ["TCS", "INFY", "WIPRO", "HCLTECH", "TATASTEEL", "HINDALCO"],
        "direct_positive": ["NESTLEIND", "HINDUNILVR", "ITC", "DABUR"],
        "indirect_positive": ["GOLD_ETFS"],
        "hedge": "Rotate to defensive FMCG + gold",
        "confidence": 0.80,
    },
}


def _call_claude(model: str, prompt: str, max_tokens: int = 1500) -> str:
    """Call Claude API."""
    if not ANTHROPIC_AVAILABLE or not settings.anthropic_api_key:
        return "{}"
    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text if response.content else "{}"


class GeopoliticalNLPAgent:
    """Monitors geopolitical developments via supply chain knowledge graph + LLM analysis."""

    def __init__(self) -> None:
        self.knowledge_graph = SUPPLY_CHAIN_KNOWLEDGE_GRAPH
        self.fast_model = settings.llm_fast_model

    def run(self, state: TradingState) -> dict[str, Any]:
        """Analyze geopolitical risks affecting the watchlist."""
        logs: list[str] = []
        errors: list[str] = []
        geo_risks: dict[str, Any] = {}

        # Get news from sentiment agent's fetched articles
        sentiment_data = state.get("sentiment_signals", {})
        news_articles = sentiment_data.get("articles", [])

        # Also check for active macro events from Kafka
        active_event = state.get("active_macro_event")

        try:
            if news_articles or active_event:
                geo_analysis = self.analyze_geopolitical_risk(
                    news_articles=news_articles,
                    active_event=active_event,
                    state=state,
                )
                geo_risks = geo_analysis
                logs.append(
                    f"[GeopoliticalNLP] Analyzed {len(news_articles)} articles. "
                    f"Risks identified: {len(geo_analysis.get('identified_risks', []))}"
                )
            else:
                # No news — check knowledge graph against current market conditions
                geo_risks = self._rule_based_analysis(state)
                logs.append("[GeopoliticalNLP] No news articles. Rule-based analysis applied.")

        except Exception as e:
            errors.append(f"[GeopoliticalNLP] Analysis error: {e}")
            geo_risks = {"identified_risks": [], "affected_stocks": {}, "hedge_recommendations": []}

        return {
            "geopolitical_risks": geo_risks,
            "agent_logs": logs,
            "errors": errors,
        }

    def analyze_geopolitical_risk(self, news_articles: list[Any],
                                  active_event: dict[str, Any] | None,
                                  state: TradingState) -> dict[str, Any]:
        """Use LLM to classify geopolitical risks from news + knowledge graph mapping."""
        watchlist = state.get("watchlist", [])
        articles_text = "\n".join(
            str(a.get("title", a) if isinstance(a, dict) else a)
            for a in news_articles[:10]
        )

        event_text = json.dumps(active_event, default=str) if active_event else "None"

        # Known risk categories for the LLM to classify against
        risk_categories = list(self.knowledge_graph.keys())

        prompt = f"""Analyze these news articles and events for geopolitical risks to Indian equities:

NEWS:
{articles_text[:2000]}

ACTIVE MACRO EVENT:
{event_text[:500]}

WATCHLIST: {', '.join(watchlist)}

KNOWN RISK CATEGORIES:
{json.dumps(risk_categories)}

For each identified risk, determine:
1. Which risk category it maps to (from the list above, or "new_risk")
2. Severity: LOW / MEDIUM / HIGH / CRITICAL
3. Time horizon: IMMEDIATE / SHORT_TERM / MEDIUM_TERM
4. Affected stocks from the watchlist

Respond ONLY in JSON:
{{
    "identified_risks": [
        {{
            "category": "crude_oil_price_increase",
            "description": "brief description",
            "severity": "MEDIUM",
            "time_horizon": "SHORT_TERM",
            "affected_watchlist_stocks": ["RELIANCE", "BPCL"],
            "recommended_action": "Reduce OMC exposure"
        }}
    ],
    "overall_risk_level": "LOW",
    "market_outlook": "brief outlook"
}}"""

        response = _call_claude(self.fast_model, prompt)
        try:
            result = json.loads(response) if response.strip().startswith("{") else {}
            # Try extracting JSON from code block
            if not result:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    result = json.loads(response[start:end])
        except json.JSONDecodeError:
            result = {}

        # Enrich with knowledge graph data
        enriched = self._enrich_with_knowledge_graph(result)
        return enriched

    def _enrich_with_knowledge_graph(self, llm_analysis: dict[str, Any]) -> dict[str, Any]:
        """Add supply chain impact data from knowledge graph to LLM analysis."""
        risks = llm_analysis.get("identified_risks", [])
        affected_stocks: dict[str, list[str]] = {}
        hedge_recommendations: list[str] = []

        for risk in risks:
            category = risk.get("category", "")
            if category in self.knowledge_graph:
                kg_data = self.knowledge_graph[category]
                risk["supply_chain_impact"] = {
                    "positive": kg_data.get("direct_positive", []),
                    "negative": kg_data.get("direct_negative", []),
                    "indirect": kg_data.get("indirect_positive", kg_data.get("indirect_negative", [])),
                }
                risk["kg_confidence"] = kg_data.get("confidence", 0.5)

                for stock in kg_data.get("direct_negative", []):
                    affected_stocks.setdefault(stock, []).append(f"NEGATIVE: {category}")
                for stock in kg_data.get("direct_positive", []):
                    affected_stocks.setdefault(stock, []).append(f"POSITIVE: {category}")

                if kg_data.get("hedge"):
                    hedge_recommendations.append(kg_data["hedge"])

        llm_analysis["affected_stocks"] = affected_stocks
        llm_analysis["hedge_recommendations"] = hedge_recommendations
        return llm_analysis

    def _rule_based_analysis(self, state: TradingState) -> dict[str, Any]:
        """Fallback rule-based geopolitical check using market data."""
        risks = []

        # Check crude oil impact via macro signals
        macro = state.get("macro_signals", {})
        crude_price = macro.get("crude_oil", {}).get("price", 0)
        crude_change = macro.get("crude_oil", {}).get("change_pct", 0)

        if crude_change > 3.0:
            risks.append({
                "category": "crude_oil_price_increase",
                "description": f"Crude oil up {crude_change:.1f}% — potential pressure on OMCs and aviation",
                "severity": "HIGH" if crude_change > 5 else "MEDIUM",
                "time_horizon": "IMMEDIATE",
                "affected_watchlist_stocks": ["RELIANCE", "BPCL", "IOC"],
            })

        # Check USD/INR stress
        usd_inr_change = macro.get("usd_inr", {}).get("change_pct", 0)
        if usd_inr_change > 0.5:  # INR weakening > 0.5%
            risks.append({
                "category": "currency_stress",
                "description": f"INR weakening {usd_inr_change:.2f}% — import cost pressure",
                "severity": "MEDIUM",
                "time_horizon": "SHORT_TERM",
                "affected_watchlist_stocks": [],
            })

        return {
            "identified_risks": risks,
            "affected_stocks": {},
            "hedge_recommendations": [],
            "overall_risk_level": "HIGH" if len(risks) > 2 else "MEDIUM" if risks else "LOW",
        }

    def get_portfolio_exposure(self, positions: dict[str, Any]) -> dict[str, list[str]]:
        """Check current portfolio exposure to geopolitical risk categories."""
        exposure: dict[str, list[str]] = {}

        for symbol in positions:
            for category, data in self.knowledge_graph.items():
                if symbol in data.get("direct_negative", []):
                    exposure.setdefault(category, []).append(f"{symbol} (NEGATIVE)")
                elif symbol in data.get("direct_positive", []):
                    exposure.setdefault(category, []).append(f"{symbol} (POSITIVE)")

        return exposure


def geopolitical_nlp_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node wrapper for Geopolitical NLP Agent."""
    agent = GeopoliticalNLPAgent()
    return agent.run(state)
