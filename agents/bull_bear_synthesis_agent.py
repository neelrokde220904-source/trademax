"""Bull vs. Bear Adversarial Debate Protocol — the most critical v2 upgrade.

Instead of unstructured LLM conversation, forces each side to produce
machine-readable JSON arguments. Only trades where conviction_score > 70
from the debate proceed to the Risk Manager.

This single filter eliminates the majority of false-positive trades.
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


CONVICTION_THRESHOLD = 70  # FIRM — never lower this threshold


def _call_claude(model: str, prompt: str, max_tokens: int = 1500) -> str:
    """Call Claude API with the given prompt."""
    if not ANTHROPIC_AVAILABLE or not settings.anthropic_api_key:
        return "{}"
    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text if response.content else "{}"


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON from Claude response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from mixed text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse JSON from LLM response")
    return {}


class BullBearSynthesisAgent:
    """Adversarial debate protocol between Bull and Bear researchers.

    Flow:
    1. Collect all analyst JSON theses
    2. Bull Researcher (claude-sonnet) argues FOR the trade
    3. Bear Researcher (claude-sonnet) argues AGAINST, attacking bull's case
    4. Portfolio Manager (claude-opus) synthesizes the debate
    5. Only conviction_score > 70 proceeds
    """

    def __init__(self) -> None:
        self.fast_model = settings.llm_fast_model
        self.decision_model = settings.llm_decision_model

    def run(self, state: TradingState) -> dict[str, Any]:
        """Run debate for all symbols in watchlist with analyst signals.

        Returns state update with debate_results dict.
        """
        logs: list[str] = []
        errors: list[str] = []
        debate_results: dict[str, Any] = {}

        watchlist = state.get("watchlist", [])
        regime = state.get("hmm_regime", {})

        # If SYSTEMIC_PANIC, skip debate entirely
        if regime.get("regime") == "SYSTEMIC_PANIC":
            logs.append("[BullBearDebate] SYSTEMIC_PANIC regime — all debates skipped, no new positions")
            return {
                "debate_results": {},
                "agent_logs": logs,
                "errors": errors,
            }

        # Collect analyst signals for each symbol
        for symbol in watchlist:
            try:
                analyst_signals = self._collect_signals(symbol, state)
                if not analyst_signals:
                    continue

                result = self.run_debate(symbol, analyst_signals, state)
                debate_results[symbol] = result

                conviction = result.get("conviction_score", 0)
                action = result.get("action", "HOLD")
                winner = result.get("winning_argument", "neutral")

                logs.append(
                    f"[BullBearDebate] {symbol}: {action} (conviction={conviction}, "
                    f"winner={winner})"
                )

                if conviction >= CONVICTION_THRESHOLD:
                    logs.append(
                        f"[BullBearDebate] {symbol} PASSED threshold ({conviction} >= {CONVICTION_THRESHOLD})"
                    )
                else:
                    logs.append(
                        f"[BullBearDebate] {symbol} REJECTED (conviction {conviction} < {CONVICTION_THRESHOLD})"
                    )

            except Exception as e:
                errors.append(f"[BullBearDebate] Error debating {symbol}: {e}")

        return {
            "debate_results": debate_results,
            "agent_logs": logs,
            "errors": errors,
        }

    def run_debate(self, symbol: str, analyst_signals: dict[str, Any],
                   state: TradingState) -> dict[str, Any]:
        """Run the full Bull vs. Bear debate for a single symbol."""
        # Step 1: Bull Researcher argues FOR
        bull_argument = self._run_bull_researcher(symbol, analyst_signals, state)

        # Step 2: Bear Researcher argues AGAINST (sees the bull case)
        bear_argument = self._run_bear_researcher(symbol, analyst_signals, bull_argument, state)

        # Step 3: Synthesis — Portfolio Manager judges the debate
        verdict = self._synthesize_debate(symbol, bull_argument, bear_argument, state)

        return verdict

    def _collect_signals(self, symbol: str, state: TradingState) -> dict[str, Any]:
        """Collect all analyst signals for a symbol."""
        signals: dict[str, Any] = {}

        for key in ("technical_signals", "fundamental_signals", "sentiment_signals",
                     "options_signals", "macro_signals", "india_signals"):
            agent_signals = state.get(key, {})
            if isinstance(agent_signals, dict):
                sym_signal = agent_signals.get(symbol, agent_signals.get("overall", {}))
                if sym_signal:
                    signals[key.replace("_signals", "")] = sym_signal

        return signals

    def _run_bull_researcher(self, symbol: str, reports: dict[str, Any],
                             state: TradingState) -> dict[str, Any]:
        """Bull side — build the strongest possible case FOR buying."""
        from agents.state_condenser import condense_for_debate, state_to_prompt_json

        ltp = state.get("live_quotes", {}).get(symbol, {}).get("ltp", "N/A")
        regime = state.get("hmm_regime", {}).get("regime", "UNKNOWN")

        condensed = condense_for_debate(state, symbol)
        reports_str = state_to_prompt_json(condensed, max_chars=3000)

        prompt = f"""You are the BULL RESEARCHER. Your sole job is to build the strongest 
possible case FOR buying {symbol}.

Analyst reports:
{reports_str}

Current price: ₹{ltp}
Market regime: {regime}

ATTACK any weaknesses in potential bear arguments. Find every reason this stock 
will go up. Be specific with numbers and catalysts.

Respond ONLY in this JSON format:
{{
    "thesis": "One paragraph bull thesis",
    "upside_target": 0.0,
    "probability": 0.75,
    "supporting_factors": ["factor1", "factor2", "factor3"],
    "key_catalysts": ["catalyst1", "catalyst2"],
    "bull_confidence": 75
}}"""

        response = _call_claude(self.fast_model, prompt)
        result = _parse_json_response(response)

        if not result:
            result = {
                "thesis": "Unable to generate bull thesis",
                "upside_target": 0.0,
                "probability": 0.0,
                "supporting_factors": [],
                "key_catalysts": [],
                "bull_confidence": 0,
            }

        return result

    def _run_bear_researcher(self, symbol: str, reports: dict[str, Any],
                             bull_argument: dict[str, Any],
                             state: TradingState) -> dict[str, Any]:
        """Bear side — destroy the bull case and identify all downside risks."""
        from agents.state_condenser import condense_for_debate, state_to_prompt_json

        ltp = state.get("live_quotes", {}).get(symbol, {}).get("ltp", "N/A")
        regime = state.get("hmm_regime", {}).get("regime", "UNKNOWN")

        condensed = condense_for_debate(state, symbol)
        reports_str = state_to_prompt_json(condensed, max_chars=2000)
        bull_str = json.dumps(bull_argument, default=str, indent=2)[:1500]

        prompt = f"""You are the BEAR RESEARCHER. Your sole job is to destroy the bull case 
for {symbol} and identify all downside risks.

Bull argument to attack:
{bull_str}

Analyst reports:
{reports_str}

Current price: ₹{ltp}
Market regime: {regime}

Be ruthless. Find margin compression, hidden debt, management red flags,
sector headwinds, regulatory risks, global macro threats. Be specific.

Respond ONLY in this JSON format:
{{
    "counter_thesis": "One paragraph bear thesis",
    "downside_risk": 0.0,
    "probability": 0.60,
    "risk_factors": ["risk1", "risk2", "risk3"],
    "bull_case_weaknesses": ["weakness1", "weakness2"],
    "bear_confidence": 60
}}"""

        response = _call_claude(self.fast_model, prompt)
        result = _parse_json_response(response)

        if not result:
            result = {
                "counter_thesis": "Unable to generate bear thesis",
                "downside_risk": 0.0,
                "probability": 0.0,
                "risk_factors": [],
                "bull_case_weaknesses": [],
                "bear_confidence": 0,
            }

        return result

    def _synthesize_debate(self, symbol: str,
                           bull: dict[str, Any],
                           bear: dict[str, Any],
                           state: TradingState) -> dict[str, Any]:
        """Portfolio Manager synthesizes the debate — final judgment.

        Uses claude-opus for the most capable reasoning.
        """
        regime = state.get("hmm_regime", {}).get("regime", "UNKNOWN")
        vix = state.get("india_vix", 0)

        bull_str = json.dumps(bull, default=str, indent=2)[:1500]
        bear_str = json.dumps(bear, default=str, indent=2)[:1500]

        prompt = f"""You are the PORTFOLIO MANAGER making the final trading decision for {symbol}.

You have heard both sides of an adversarial debate:

BULL CASE:
{bull_str}

BEAR CASE:
{bear_str}

CONTEXT:
- Market regime: {regime}
- India VIX: {vix}
- Conviction threshold: {CONVICTION_THRESHOLD} (only invest if your conviction exceeds this)

Evaluate BOTH arguments critically. Consider:
1. Quality of evidence (specific numbers vs vague claims)
2. Relevance to current market regime
3. Risk/reward asymmetry
4. Time horizon alignment

Respond ONLY in this JSON format:
{{
    "action": "BUY",
    "conviction_score": 75,
    "reasoning": "Brief synthesis of why this is the right call",
    "winning_argument": "bull",
    "risk_reward_ratio": 2.5,
    "recommended_position_pct": 0.05,
    "stop_loss_pct": 0.03,
    "time_horizon": "SWING"
}}

Where:
- action: BUY, SELL, or HOLD
- conviction_score: 0-100 (only BUY if > {CONVICTION_THRESHOLD})
- winning_argument: "bull", "bear", or "neutral"
- time_horizon: "INTRADAY", "SWING", or "POSITIONAL"
"""

        response = _call_claude(self.decision_model, prompt)
        result = _parse_json_response(response)

        if not result:
            return {
                "action": "HOLD",
                "conviction_score": 0,
                "reasoning": "Failed to synthesize debate",
                "winning_argument": "neutral",
            }

        # Enforce conviction threshold
        conviction = result.get("conviction_score", 0)
        if conviction < CONVICTION_THRESHOLD:
            result["action"] = "HOLD"
            result["reasoning"] = (
                f"Conviction {conviction} below threshold {CONVICTION_THRESHOLD}. "
                + result.get("reasoning", "")
            )

        return result


def bull_bear_synthesis_agent(state: TradingState) -> dict[str, Any]:
    """LangGraph node wrapper for the Bull/Bear Debate."""
    agent = BullBearSynthesisAgent()
    return agent.run(state)
