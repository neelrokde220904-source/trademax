"""Tests for agents base state and agent creation."""

from agents.base_agent import TradingState, create_initial_state


def test_create_initial_state():
    """create_initial_state should return a valid TradingState."""
    state = create_initial_state(["RELIANCE", "TCS"])

    assert state["watchlist"] == ["RELIANCE", "TCS"]
    assert state["agent_logs"] == []
    assert state["errors"] == []
    assert state["market_data"] == {}
    assert state["technical_signals"] == {}
    assert state["trading_decisions"] == []


def test_initial_state_has_all_keys():
    """State dict should contain all required keys."""
    state = create_initial_state([])
    required_keys = [
        "watchlist", "market_data", "technical_signals", "fundamental_signals",
        "sentiment_signals", "options_signals", "macro_signals",
        "india_signals", "risk_assessment", "trading_decisions",
        "executed_orders", "current_positions", "agent_logs", "errors",
    ]
    for key in required_keys:
        assert key in state, f"Missing key: {key}"
