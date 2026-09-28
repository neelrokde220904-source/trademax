"""State Condenser — reduce TradingState to essential fields before LLM calls (~40% token savings)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def condense_for_llm(state: dict[str, Any], symbol: str | None = None) -> dict[str, Any]:
    """Extract only the fields LLM agents actually use from the full TradingState.

    Instead of dumping 40+ fields into every prompt, we extract:
    - watchlist (symbol list)
    - live_quotes for the target symbol (ltp, volume, change_pct)
    - india_vix (scalar)
    - hmm_regime (regime name + confidence)
    - fii_dii_data (net buy/sell summary)
    - technical_signals for the symbol (if available)
    - fundamental_signals for the symbol (if available)
    - sentiment_signals for the symbol (if available)
    - options_signals for the symbol (if available)
    - macro_signals summary
    - india_signals for the symbol (if available)
    - geopolitical_risks summary (risk_level + top risks)
    - current_positions count + value
    - portfolio_value

    This avoids sending raw DataFrames, full position dicts, and unused v2 fields.
    """
    condensed: dict[str, Any] = {}

    # Basics
    condensed["watchlist"] = state.get("watchlist", [])
    condensed["date"] = state.get("date", "")
    condensed["india_vix"] = state.get("india_vix", 0)
    condensed["gift_nifty"] = state.get("gift_nifty", 0)

    # HMM regime — just regime name and confidence
    hmm = state.get("hmm_regime", {})
    if isinstance(hmm, dict):
        condensed["hmm_regime"] = {
            "regime": hmm.get("regime", "UNKNOWN"),
            "confidence": hmm.get("confidence", 0),
            "directive": hmm.get("directive", ""),
        }

    # FII/DII — summary only
    fii_dii = state.get("fii_dii_data", {})
    if isinstance(fii_dii, dict):
        condensed["fii_dii_summary"] = {
            "fii_net": fii_dii.get("fii_net", fii_dii.get("fii_net_buy", 0)),
            "dii_net": fii_dii.get("dii_net", fii_dii.get("dii_net_buy", 0)),
        }

    # Live quote for target symbol
    if symbol:
        quotes = state.get("live_quotes", {})
        if isinstance(quotes, dict) and symbol in quotes:
            q = quotes[symbol]
            condensed["live_quote"] = {
                "ltp": q.get("ltp", 0),
                "volume": q.get("volume", 0),
                "change_pct": q.get("change_pct", q.get("pct_change", 0)),
            }

        # Per-symbol signals
        for signal_key in (
            "technical_signals", "fundamental_signals", "sentiment_signals",
            "options_signals", "india_signals",
        ):
            signals = state.get(signal_key, {})
            if isinstance(signals, dict) and symbol in signals:
                val = signals[symbol]
                # Drop raw DataFrames — keep dicts/scalars
                if not isinstance(val, pd.DataFrame):
                    condensed[signal_key.replace("_signals", "")] = _truncate_dict(val, max_keys=10)

    # Macro signals — general summary
    macro = state.get("macro_signals", {})
    if isinstance(macro, dict):
        condensed["macro_summary"] = _truncate_dict(macro.get("overall", macro), max_keys=8)

    # Geopolitical — risk level + top 3 risks
    geo = state.get("geopolitical_risks", {})
    if isinstance(geo, dict):
        condensed["geo_risk_level"] = geo.get("overall_risk_level", "UNKNOWN")
        risks = geo.get("identified_risks", [])
        condensed["geo_top_risks"] = [
            {"category": r.get("category", ""), "severity": r.get("severity", "")}
            for r in risks[:3]
        ]

    # Portfolio context — counts only
    positions = state.get("current_positions", {})
    condensed["open_positions_count"] = len(positions) if isinstance(positions, dict) else 0
    condensed["portfolio_value"] = state.get("portfolio_value", 0)
    condensed["available_capital"] = state.get("available_capital", 0)

    return condensed


def condense_for_debate(state: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Condense state specifically for the Bull/Bear debate — includes all agent signals."""
    base = condense_for_llm(state, symbol)

    # Debate needs all analyst signal summaries for the symbol
    for signal_key in (
        "technical_signals", "fundamental_signals", "sentiment_signals",
        "options_signals", "macro_signals", "india_signals",
    ):
        signals = state.get(signal_key, {})
        if isinstance(signals, dict):
            sym_sig = signals.get(symbol, signals.get("overall", {}))
            if sym_sig and not isinstance(sym_sig, pd.DataFrame):
                key = signal_key.replace("_signals", "")
                base[key] = _truncate_dict(sym_sig, max_keys=12)

    # OFI signals if available
    ofi = state.get("ofi_signals", {})
    if isinstance(ofi, dict) and symbol in ofi:
        base["ofi"] = ofi[symbol]

    return base


def state_to_prompt_json(condensed: dict[str, Any], max_chars: int = 3000) -> str:
    """Serialize condensed state to compact JSON for prompt injection."""
    text = json.dumps(condensed, default=str, indent=1)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


def _truncate_dict(d: Any, max_keys: int = 10) -> Any:
    """Truncate a dict to max_keys entries, keeping the most informative."""
    if not isinstance(d, dict):
        return d
    if len(d) <= max_keys:
        return {k: _clean_value(v) for k, v in d.items()}
    # Keep first max_keys items
    return {k: _clean_value(v) for k, v in list(d.items())[:max_keys]}


def _clean_value(v: Any) -> Any:
    """Clean values for JSON serialization — drop DataFrames, trim long strings."""
    if isinstance(v, pd.DataFrame):
        return f"<DataFrame {v.shape[0]}×{v.shape[1]}>"
    if isinstance(v, str) and len(v) > 300:
        return v[:300] + "..."
    if isinstance(v, list) and len(v) > 10:
        return v[:10]
    return v
