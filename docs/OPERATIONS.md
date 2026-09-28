# India AI Trader — Operational Guide and Living Architecture Record

**Status:** Minimum viable deployment (MVD), paper trading by default  
**Owner:** Trading-system operator  
**Last reviewed:** 2026-09-28  
**Change policy:** Update this document and add an ADR whenever execution, risk limits, data sources, or a broker integration changes.

## 1. What this system does

India AI Trader is a scheduled, **short-horizon NSE equity paper trader**. It scans a configurable list of liquid Indian equities during NSE cash-market hours, builds technical/fundamental/sentiment/macro inputs, applies portfolio risk checks, and records paper fills in SQLite. It can use Angel One for authenticated data and live execution, with yfinance as an OHLCV fallback.

The currently operational MVD trades the technical-agent pipeline. It is not an HFT system, does not promise profitability, and should not be promoted to live trading merely because a scan completes.

### Implemented decision model

1. Fetch 15-minute OHLCV, live quotes, India VIX, and portfolio state.
2. Screen for liquidity, trend strength, and adequate range.
3. Run the analyst fan-out: technical, fundamental, sentiment, options, macro, India-market, and geopolitical context.
4. Apply regime classification, bull/bear synthesis, risk assessment, allocation/hedging, and compliance context.
5. Make final BUY/SELL/HOLD decisions using Claude when configured; otherwise use deterministic technical/fundamental agreement.
6. Route Indian orders through Angel One's order manager. In paper mode, simulated fills include configured slippage and are written to SQLite.
7. Monitor registered orders every five minutes and send stop, target, and time exits through the same order manager.

### Strategy modules present in the repository

| Module | Intended style | Status in scheduled MVD |
| --- | --- | --- |
| `SupertrendADX` | 15-minute intraday trend following | Used by the example backtest; not directly selected by the scheduler |
| `MomentumBreakout` | Daily 52-week-high swing momentum | Available for backtests/research only |
| `MeanReversion` | Daily Bollinger/RSI swing mean reversion | Available for backtests/research only |
| `NiftyOptionsSeller` | Short OTM index-option premium/theta strategy | Research only; not enabled for autonomous execution |

This distinction is deliberate: the MVD has one governed execution path. A strategy must pass backtesting, paper evaluation, and an ADR before being connected to production execution.

## 2. Features and safeguards

- Paper trading is the default (`PAPER_TRADING=true`). Paper fills use 0.05% configured slippage and persist to `india_trader.db`.
- A **fail-closed live-trading guard** rejects `PAPER_TRADING=false` unless `LIVE_TRADING_ACKNOWLEDGEMENT=I_ACCEPT_REAL_MONEY_RISK` is explicitly supplied.
- New positions are constrained by max 10% per trade, 8 simultaneous positions, a 3% daily-loss limit, historical VaR checks, and correlation screening.
- VIX above 22 reduces size; VIX above 28 blocks new entries.
- The opening and closing 15-minute windows reject new entries. Intraday positions are checked for time exit at 15:15 IST.
- Stops and targets are ATR-derived; the stop manager supports ATR trailing stops and now receives entry metadata from the order manager.
- A dead-man's switch halts scans after repeated critical component failures.
- The scheduler runs a pre-market data/authentication job, intraday scans, five-minute exit checks, post-market snapshots, and optional Telegram notifications.
- An XGBoost signal gate is available, but is inactive until it has a trained model from a valid historical dataset.

## 3. Repository structure

| Path | Responsibility |
| --- | --- |
| `main.py` | CLI entry point, scheduler, scan, backtest, dashboard, preflight |
| `config/` | Validated environment configuration and instrument lookup |
| `agents/` | LangGraph state and analysis/decision agents |
| `data/` | OHLCV, indicators, news, options, and market-event sources |
| `risk/` | Entry risk controls, sizing, exit manager, regime classifier, circuit breaker |
| `broker/` | Angel One client, paper/live order manager, portfolio state, router |
| `scheduler/` | IST market-hour orchestration and exit lifecycle |
| `backtester/` | Historical simulation, metrics, optimization, reports |
| `operations/` | Non-mutating operational readiness checks |
| `database/` | SQLite models and session lifecycle |
| `docs/` | This living operating and architecture record |

## 4. Runbook

### Initial setup

```bash
cd /Users/neelrokde/Desktop/trading\ agent2/india-ai-trader
poetry install
cp .env.example .env
python3 main.py --preflight
pytest tests/ -q
```

Use a broker account only for data/session access while evaluating paper trades. Do not put secrets in source control; `.env` is ignored.

### Operating commands

```bash
# Safe, no broker calls and no writes
python3 main.py --preflight

# One paper-trading pipeline cycle
python3 main.py --scan

# Scheduled paper operation plus dashboard
python3 main.py

# Historical test of the Supertrend example
python3 main.py --backtest
```

Before every session, run preflight, confirm the system is still in paper mode, review logs, and confirm the instrument master downloaded successfully. During a session, monitor the dashboard/Telegram and stop the process if data quality is suspect. After the session, review the SQLite trade ledger, daily PnL snapshot, rejected signals, and any dead-man's-switch events.

### Readiness gate for a future live pilot

1. Obtain a representative out-of-sample backtest, including Indian charges and slippage.
2. Complete at least 30 market days of paper trading with reconciled fills and no unresolved circuit-breaker events.
3. Perform a manual review of risk limits, broker order types, token mapping, holiday calendar, and position reconciliation.
4. Create an ADR approving the exact strategy, universe, order type, size, and rollback procedure.
5. Set the explicit live acknowledgement only for a limited pilot. Remove it to return to fail-closed paper mode.

## 5. New changes in this MVD

- Fixed `.env` parsing for the documented comma-separated `GLOBAL_WATCHLIST`; this was preventing settings—and therefore the application—from starting.
- Added `--preflight`, a read-only readiness check which validates capital/risk settings, execution mode, and cache availability before the session.
- Added explicit acknowledgement required for live execution; simply changing `PAPER_TRADING` cannot silently enable real-money orders.
- Carried fill price, stop loss, target, holding period, and strategy through order results so the scheduled stop manager has the information it needs.
- Repaired scheduled exits: it now calls the synchronous order API with a resolved instrument token rather than awaiting an incompatible dictionary call.
- Made paper mode broker-optional: absent Angel One credentials no longer trigger repeated login attempts for quotes, WebSocket setup, or simulated fills; public OHLCV fallback remains available.
- Added regression coverage for preflight and watchlist parsing.

## 6. Architecture Decision Records (ADRs)

### ADR-0001 — Paper-first execution (accepted)

**Decision:** The default and MVD execution mode is paper trading. Live execution requires a separate literal acknowledgement in configuration.  
**Why:** Order bugs, data gaps, and model instability can create irreversible losses. A safer default makes an accidental mode change non-operational.  
**Consequences:** Real-money use has an explicit operational step; paper fills are simulations and do not establish broker fill quality.

### ADR-0002 — One governed MVD strategy path (accepted)

**Decision:** The scheduled execution path uses the multi-agent technical decision pipeline. Research strategy classes are not automatically merged into it.  
**Why:** Connecting four incompatible holding periods and instrument types without portfolio-level validation would make results uninterpretable and risk controls incomplete.  
**Consequences:** Options selling, breakout, and mean-reversion modules remain research/backtest work until individually approved.

### ADR-0003 — Fail closed on configuration/data ambiguity (accepted)

**Decision:** Invalid settings prevent startup; missing instrument tokens prevent an exit/order rather than guessing.  
**Why:** A bad configuration or unresolvable instrument identity must not create an unintended trade.  
**Consequences:** Operators must remediate configuration/cache errors before a session. Preflight surfaces these conditions early.

### ADR template

Copy this section for every material change:

```text
### ADR-NNNN — Title (proposed | accepted | superseded)
Decision:
Context and evidence:
Alternatives considered:
Risk and rollback:
Validation / success metrics:
Owner and review date:
```

## 7. Known limitations and roadmap

- The dashboard is an operator console: it provides health, portfolio, P&L, positions, signals, execution history, and pause/resume controls. It intentionally cannot enable live mode; that remains an explicit configuration-and-review procedure.
- Agent cards show pipeline readiness (or paused state), not fabricated per-agent execution telemetry. Persisted per-node timing and failure telemetry is the next observability increment.
- Paper portfolio accounting and broker reconciliation need a dedicated end-to-end reconciliation test before a live pilot.
- The scheduler uses a static 2026 holiday list; make it an annually reviewed data source before 2027.
- LLM output is schema-parsed but should be constrained with a formal structured-output model and independent order validator.
- The L2, HMM, MARL, Kafka, global-broker, and episodic-memory components are optional/experimental integrations; treat them as non-essential until monitored and tested in the deployed environment.
- The strategy modules require walk-forward validation, transaction-cost stress tests, and portfolio-level conflict rules before production wiring.

## 8. Self-documenting code conventions

- Public modules, classes, and functions must state purpose, inputs, outputs, failure mode, and side effects in docstrings.
- Execution functions must state whether they can place an order or only simulate one.
- Risk checks must return a human-readable rejection reason and log it.
- Configuration additions require `.env.example`, a preflight check where applicable, and a corresponding test.
- Any real-money, sizing, data-source, or schedule change requires an ADR update above.
