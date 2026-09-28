"""Tests for config/settings.py."""

from config.settings import Settings, settings


def test_settings_defaults():
    """Settings should load with sensible defaults."""
    assert settings.paper_trading is True
    assert settings.initial_capital == 25_000.0
    assert settings.max_positions == 8
    assert settings.max_daily_loss_pct == 0.03
    assert settings.max_capital_per_trade_pct == 0.10


def test_settings_market_constants():
    assert settings.brokerage_per_order == 20.0
    assert settings.risk_free_rate == 0.065
    assert settings.slippage_pct == 0.0005


def test_settings_watchlist():
    assert len(settings.default_watchlist) == 20
    assert "RELIANCE" in settings.default_watchlist
    assert "TCS" in settings.default_watchlist


def test_settings_nse_holidays():
    assert len(settings.nse_holidays) > 10
    assert "2026-01-26" in settings.nse_holidays  # Republic Day
