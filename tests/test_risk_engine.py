"""Tests for risk engine and position sizer."""

import pytest

from risk.risk_engine import RiskEngine
from risk.position_sizer import PositionSizer


def test_risk_engine_fno_ban():
    """Should block trades for F&O banned symbols."""
    engine = RiskEngine()
    engine.update_fno_ban(["INFY", "TCS"])

    allowed, reason = engine._check_fno_ban({"symbol": "INFY"})
    assert allowed is False
    assert "F&O ban" in reason

    allowed, _ = engine._check_fno_ban({"symbol": "RELIANCE"})
    assert allowed is True


def test_risk_engine_vix_halt():
    """Should block trades when VIX is above halt threshold."""
    engine = RiskEngine()
    engine.update_vix(30.0)

    allowed, reason = engine._check_vix({})
    assert allowed is False
    assert "VIX" in reason

    engine.update_vix(15.0)
    allowed, _ = engine._check_vix({})
    assert allowed is True


def test_position_sizer_kelly():
    """Kelly calculation should return a sensible fraction."""
    sizer = PositionSizer()

    # 55% win rate, 1.5:1 ratio
    kelly = sizer._kelly(0.55, 1.5)
    assert 0 < kelly < 0.25

    # 30% win rate should give 0 or very small
    kelly_low = sizer._kelly(0.30, 1.0)
    assert kelly_low == 0.0


def test_position_sizer_var():
    """VaR calculation should return a positive number."""
    sizer = PositionSizer()
    import numpy as np

    np.random.seed(42)
    returns = list(np.random.randn(60) * 0.01)
    var = sizer._calculate_var(returns, 100_000)
    assert var > 0
