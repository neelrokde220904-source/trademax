"""Pydantic BaseSettings — centralised application configuration loaded from .env."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All configuration for the India AI Trader system."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        # List settings are deliberately parsed by explicit validators so the
        # documented comma-separated .env syntax is accepted.
        enable_decoding=False,
    )

    # --- Angel One SmartAPI ---
    angel_api_key: str = Field(default="", description="Angel One API key")
    angel_client_id: str = Field(default="", description="Angel One client ID")
    angel_mpin: str = Field(default="", description="Angel One MPIN")
    angel_totp_secret: str = Field(default="", description="TOTP secret for Angel One 2FA")

    # --- Anthropic ---
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # --- News ---
    news_api_key: str = Field(default="", description="NewsAPI.org API key")

    # --- Telegram ---
    telegram_bot_token: str = Field(default="", description="Telegram bot token")
    telegram_chat_id: str = Field(default="", description="Telegram chat ID for notifications")

    # --- Trading Config ---
    paper_trading: bool = Field(default=True, description="Paper trading mode (default ON)")
    live_trading_acknowledgement: str = Field(
        default="",
        description="Required literal acknowledgement before live orders can be enabled.",
    )
    initial_capital: float = Field(default=25_000.0, description="Starting capital in ₹")
    max_capital_per_trade_pct: float = Field(default=0.10, description="Max 10% per trade")
    max_positions: int = Field(default=8, description="Max simultaneous open positions")
    max_daily_loss_pct: float = Field(default=0.03, description="Stop trading if daily loss > 3%")

    # --- Risk Hard Limits ---
    min_adx_for_trade: float = Field(default=20.0)
    no_trade_window_open_minutes: int = Field(default=15)
    no_trade_window_close_minutes: int = Field(default=15)
    max_nifty_vix: float = Field(default=22.0)
    vix_halt_threshold: float = Field(default=28.0)
    stop_loss_atr_multiplier: float = Field(default=1.5)
    trailing_stop_atr_multiplier: float = Field(default=2.0)

    # --- System ---
    database_url: str = Field(default="sqlite:///./india_trader.db")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    dashboard_port: int = Field(default=8000)

    # --- LLM Models ---
    llm_decision_model: str = Field(default="claude-opus-4-6", description="Primary decision model")
    llm_fast_model: str = Field(default="claude-haiku-4-5-20251001", description="Fast signal model")

    # --- Market Constants ---
    risk_free_rate: float = Field(default=0.065, description="India 10Y bond yield ~6.5%")
    brokerage_per_order: float = Field(default=20.0, description="₹20 per executed order")
    stt_delivery_pct: float = Field(default=0.001, description="STT 0.1% on delivery")
    stt_intraday_pct: float = Field(default=0.00025, description="STT 0.025% on intraday")
    exchange_charges_pct: float = Field(default=0.0000345)
    gst_on_brokerage_pct: float = Field(default=0.18)
    stamp_duty_pct: float = Field(default=0.00015, description="0.015% on buy")
    slippage_pct: float = Field(default=0.0005, description="0.05% conservative slippage")

    # --- Instrument Master ---
    instrument_master_url: str = Field(
        default="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    )
    instrument_cache_path: str = Field(
        default=str(PROJECT_ROOT / "data" / "cache" / "instrument_master.json")
    )

    # --- WebSocket ---
    ws_max_tokens: int = Field(default=1000, description="Max tokens per WS session")
    ws_max_connections: int = Field(default=3, description="Max concurrent WS connections")

    # --- Watchlist Defaults ---
    default_watchlist: list[str] = Field(
        default=[
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
            "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
            "SUNPHARMA", "TATAMOTORS", "WIPRO", "HCLTECH", "ADANIENT",
        ],
    )

    # --- Index Tokens (Angel One) ---
    nifty50_token: str = Field(default="99926000")
    banknifty_token: str = Field(default="99926009")
    nifty_midcap_token: str = Field(default="99926037")

    # --- NSE Holidays (2026 — update yearly) ---
    nse_holidays: list[str] = Field(
        default=[
            "2026-01-26", "2026-03-10", "2026-03-30", "2026-03-31",
            "2026-04-02", "2026-04-03", "2026-04-14", "2026-05-01",
            "2026-06-26", "2026-07-07", "2026-08-15", "2026-08-25",
            "2026-10-02", "2026-10-20", "2026-10-21", "2026-10-22",
            "2026-11-04", "2026-11-05", "2026-12-25",
        ],
    )

    # ────────────── v2.0 additions ──────────────────────

    # --- Interactive Brokers ---
    ibkr_host: str = Field(default="127.0.0.1", description="TWS/Gateway host")
    ibkr_port: int = Field(default=7497, description="7497=TWS paper, 7496=TWS live")
    ibkr_client_id: int = Field(default=1)
    ibkr_account: str = Field(default="", description="IBKR account ID")

    # --- Alpaca ---
    alpaca_api_key: str = Field(default="", description="Alpaca API key")
    alpaca_secret_key: str = Field(default="", description="Alpaca secret key")
    alpaca_paper: bool = Field(default=True, description="Use Alpaca paper trading")

    # --- Setu Banking ---
    setu_api_key: str = Field(default="", description="Setu UPI API key")
    setu_base_url: str = Field(default="https://sandbox.setu.co")

    # --- Kafka ---
    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    kafka_group_id: str = Field(default="india-ai-trader")
    kafka_events_topic: str = Field(default="market-events")
    kafka_enabled: bool = Field(default=False, description="Enable Kafka consumer")

    # --- FEMA / LRS ---
    lrs_annual_limit_usd: float = Field(default=250_000.0, description="RBI LRS annual limit")
    tcs_threshold_inr: float = Field(default=700_000.0, description="TCS 20% threshold ₹7L")

    # --- HMM Regime ---
    hmm_model_path: str = Field(default="", description="Path to pre-trained HMM model pickle")
    hmm_n_states: int = Field(default=4)
    hmm_retrain_days: int = Field(default=90, description="Retrain HMM every N days")

    # --- Vector DB / ChromaDB ---
    chromadb_path: str = Field(default="", description="ChromaDB persistent path (default: vector_db/trade_memory)")

    # --- Alpha Miner ---
    alpha_miner_enabled: bool = Field(default=True)
    alpha_miner_max_hypotheses: int = Field(default=3, description="Hypotheses per overnight run")
    alpha_miner_max_generations: int = Field(default=5, description="Max mutation rounds")

    # --- Global Watchlist (US/EU instruments) ---
    global_watchlist: list[str] = Field(
        default=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        description="US/global symbols for IBKR/Alpaca",
    )

    # --- Bull/Bear Debate ---
    debate_conviction_threshold: int = Field(default=70, description="Min conviction to act (0-100)")

    # --- Order Routing ---
    large_order_threshold_inr: int = Field(default=500_000, description="₹5L triggers VWAP/TWAP")
    large_order_threshold_usd: int = Field(default=10_000, description="$10K triggers VWAP/TWAP")

    @field_validator("global_watchlist", mode="before")
    @classmethod
    def parse_global_watchlist(cls, value: Any) -> Any:
        """Accept JSON arrays and the comma-separated .env form documented for users."""
        if isinstance(value, str):
            return [symbol.strip().upper() for symbol in value.split(",") if symbol.strip()]
        return value

    @model_validator(mode="after")
    def require_explicit_live_trading_acknowledgement(self) -> "Settings":
        """Fail closed: changing PAPER_TRADING alone can never enable live orders."""
        if not self.paper_trading and self.live_trading_acknowledgement != "I_ACCEPT_REAL_MONEY_RISK":
            raise ValueError(
                "Live trading requires LIVE_TRADING_ACKNOWLEDGEMENT=I_ACCEPT_REAL_MONEY_RISK. "
                "Keep PAPER_TRADING=true until paper results and broker controls are reviewed."
            )
        return self


# Singleton
settings = Settings()
