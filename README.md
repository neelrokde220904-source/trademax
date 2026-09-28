
#  TradeMax

Autonomous AI trading system for Indian stock markets (NSE/BSE) powered by LangGraph multi-agent orchestration and Anthropic Claude.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   SCHEDULER                       │
│          (APScheduler — IST market hours)         │
├──────────────────────────────────────────────────┤
│                 LANGGRAPH PIPELINE                │
│                                                   │
│  MarketData ──┬── Technical ──┐                   │
│               ├── Fundamental ─┤                   │
│               ├── Sentiment ───┼── RiskManager     │
│               ├── Options ─────┤    ↓              │
│               ├── Macro ───────┤  PortfolioManager │
│               └── IndiaSpecific┘    ↓              │
│                                 OrderExecutor      │
├──────────────────────────────────────────────────┤
│  BROKER: Angel One SmartAPI + WebSocket Feed      │
│  DATA:   yfinance, NewsAPI, NSE feeds             │
│  RISK:   Kelly sizing, VaR, ATR trailing stops    │
│  DB:     SQLite (trades, signals, PnL)            │
│  UI:     FastAPI + React dashboard                │
│  ALERTS: Telegram bot notifications               │
└──────────────────────────────────────────────────┘
```

## Features

- **9 Specialised AI Agents** — Market Data, Technical, Fundamental, Sentiment, Options, Macro, India-Specific, Risk Manager, Portfolio Manager
- **4 Trading Strategies** — NIFTY Options Seller, Supertrend+ADX, Momentum Breakout, Mean Reversion
- **India Market Costs** — STT, exchange charges, GST, stamp duty, ₹20 brokerage
- **Paper Trading Default** — Must complete 30 days paper trading before going live
- **Risk Hard Limits** — 3% daily loss cap, 8 max positions, VIX halt at 28
- **Full Backtester** — With Indian cost model, Sharpe/Sortino/Calmar using 6.5% risk-free rate
- **Real-time Dashboard** — Dark terminal theme, equity curve, agent status, signal feed
- **Telegram Alerts** — Trade executions, daily PnL, risk alerts

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Node.js 18+ (for dashboard frontend)
- Angel One account with SmartAPI access
- Anthropic API key

### 2. Install

```bash
cd india-ai-trader

# Python dependencies
pip install -e .

# Dashboard frontend (optional)
cd dashboard/frontend
npm install
cd ../..
```

### 3. Configure

```bash
# Copy env template and fill in your credentials
cp .env.example .env
```

Edit `.env` with:
- `ANGEL_API_KEY` — Your Angel One API key
- `ANGEL_CLIENT_ID` — Your client ID
- `ANGEL_MPIN` — Your MPIN
- `ANGEL_TOTP_SECRET` — TOTP secret for 2FA
- `ANTHROPIC_API_KEY` — Your Anthropic API key
- `TELEGRAM_BOT_TOKEN` — (optional) Telegram bot token
- `TELEGRAM_CHAT_ID` — (optional) Your chat ID

### 4. Run

```bash
# Full autonomous trading (paper mode by default)
python main.py

# Single scan cycle (for testing)
python main.py --scan

# Run backtest
python main.py --backtest

# Dashboard only
python main.py --dashboard
```

### 5. Dashboard

- Backend: `http://localhost:8000`
- Frontend: `cd dashboard/frontend && npm run dev` → `http://localhost:3000`

## Project Structure

```
india-ai-trader/
├── main.py                    # Entry point
├── config/
│   ├── settings.py            # All configuration (Pydantic BaseSettings)
│   └── instruments.py         # Symbol → token cache
├── database/
│   ├── models.py              # SQLAlchemy models (Trade, Signal, Portfolio, etc.)
│   └── db.py                  # Database session management
├── broker/
│   ├── angel_client.py        # SmartAPI wrapper with TOTP
│   ├── order_manager.py       # Paper/live order execution
│   ├── portfolio_tracker.py   # Position & PnL tracking
│   └── websocket_feed.py      # Real-time tick data
├── data/
│   ├── instrument_master.py   # NSE instrument download
│   ├── market_data.py         # OHLCV fetching (Angel + yfinance fallback)
│   ├── indicators.py          # Technical indicators (pandas-ta)
│   ├── news_fetcher.py        # News aggregation
│   └── options_data.py        # Options chain analysis
├── agents/
│   ├── base_agent.py          # TradingState TypedDict + BaseAgent ABC
│   ├── market_data_agent.py   # OHLCV + live quotes
│   ├── technical_agent.py     # RSI, MACD, Supertrend, etc.
│   ├── fundamental_agent.py   # PE, ROCE, promoter holding
│   ├── sentiment_agent.py     # News sentiment via Claude
│   ├── options_agent.py       # PCR, max pain, unusual OI
│   ├── macro_agent.py         # Global cues, FII/DII, VIX
│   ├── india_specific_agent.py # F&O ban, bulk deals, circuits
│   ├── risk_manager_agent.py  # Hard limits, Kelly, VaR
│   ├── portfolio_manager_agent.py # Final AI decisions
│   └── graph.py               # LangGraph StateGraph wiring
├── strategies/
│   ├── nifty_options_seller.py # OTM options selling
│   ├── supertrend_adx.py      # Trend-following
│   ├── momentum_breakout.py   # 52-week high breakout
│   └── mean_reversion.py      # Bollinger Band reversion
├── risk/
│   ├── risk_engine.py         # Pre-trade risk checks
│   ├── position_sizer.py      # Kelly + VaR sizing
│   └── stop_loss_manager.py   # ATR trailing stops
├── notifications/
│   └── telegram_notifier.py   # Trade & PnL alerts
├── scheduler/
│   └── market_scheduler.py    # IST market-hours scheduler
├── backtester/
│   ├── engine.py              # Event-driven backtester
│   ├── metrics.py             # Sharpe, Sortino, Calmar, etc.
│   └── report_generator.py    # JSON + text reports
├── dashboard/
│   ├── backend/
│   │   └── app.py             # FastAPI server with SSE
│   └── frontend/
│       ├── src/App.jsx         # React dashboard (dark theme)
│       ├── package.json
│       └── vite.config.js
├── tests/
│   ├── test_settings.py
│   ├── test_indicators.py
│   ├── test_risk_engine.py
│   ├── test_backtester.py
│   └── test_agents.py
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Trading Schedule (IST)

| Time  | Activity                              |
|-------|---------------------------------------|
| 08:00 | Pre-market — auth, instrument master  |
| 09:00 | Connect WebSocket feed                |
| 09:30 | First scan (after 15-min quiet window)|
| 10:00–14:00 | Hourly scans                   |
| 14:45 | Last scan                             |
| Every 5 min | Stop-loss / target checks       |
| 15:15 | Intraday square-off                   |
| 15:35 | Post-market — save snapshots          |
| 16:00 | Telegram daily summary                |
| 18:00 | Cleanup & logout                      |

## Risk Controls

- **Daily Loss Limit**: 3% of capital → auto-stop all trading
- **Max Positions**: 8 simultaneous
- **Per-Trade Capital**: Max 10%
- **VIX Halt**: No new trades if India VIX > 28
- **No-Trade Windows**: First/last 15 min of market
- **F&O Ban Filter**: Auto-skip banned stocks
- **Kelly Position Sizing**: Half-Kelly for safety
- **VaR Constraint**: Max 2% portfolio VaR per position

## Running Tests

```bash
pytest tests/ -v
```

## Operations and architecture

The paper-trading runbook, MVD scope, safety controls, architecture decisions, and
future live-pilot gate are maintained in [docs/OPERATIONS.md](docs/OPERATIONS.md).

## ⚠️ Disclaimer

This software is for educational purposes. Trading involves significant risk of financial loss. Paper trade for at least 30 days before considering live trading. The authors are not responsible for any losses incurred.
