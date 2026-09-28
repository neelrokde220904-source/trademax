"""LangGraph workflow — multi-agent trading pipeline (v2.0).

v2.0 Flow:
    market_data + kafka_events + l2_order_flow
    → hmm_regime
    → [technical, fundamental, sentiment, options, macro, india_specific, geopolitical_nlp] (parallel)
    → bull_bear_debate
    → [marl_optimizer, delta_hedger] (parallel)
    → compliance_check
    → portfolio_manager
    → order_router
    → episodic_reflection
    → END
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from agents.base_agent import TradingState, create_initial_state
from agents.fundamental_agent import fundamental_agent
from agents.india_specific_agent import india_specific_agent
from agents.macro_agent import macro_agent
from agents.market_data_agent import market_data_agent
from agents.options_agent import options_agent
from agents.portfolio_manager_agent import portfolio_manager_agent
from agents.risk_manager_agent import risk_manager_agent
from agents.sentiment_agent import sentiment_agent
from agents.technical_agent import technical_agent
from broker.order_manager import place_order
from config.instruments import get_token
from config.settings import settings

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    StateGraph = None
    END = None

# Deferred import to avoid circular dependency
from agents.quantitative_screener import quantitative_screener_node


# ─── v2 Node Wrappers (safe imports) ────────────────────────────

def _kafka_events_node(state: TradingState) -> dict[str, Any]:
    """Consume event-driven Kafka/polling stream."""
    try:
        from data.event_driven_kafka_stream import event_stream
        events = event_stream.consume_events(max_events=10)
        critical = [e for e in events if e.get("priority") == "CRITICAL"]
        return {
            "active_macro_event": critical[0] if critical else {},
            "agent_logs": [f"[KafkaEvents] Consumed {len(events)} events, {len(critical)} critical"],
        }
    except Exception as e:
        return {"agent_logs": [f"[KafkaEvents] Error: {e}"], "errors": [str(e)]}


def _l2_order_flow_node(state: TradingState) -> dict[str, Any]:
    """Analyze L2 order book for OFI signals."""
    try:
        from data.l2_order_flow_analyzer import l2_analyzer
        signals = {}
        for symbol in state.get("watchlist", [])[:10]:
            sig = l2_analyzer.get_signal(symbol)
            if sig:
                signals[symbol] = sig
        return {
            "ofi_signals": signals,
            "agent_logs": [f"[L2OrderFlow] OFI signals for {len(signals)} symbols"],
        }
    except Exception as e:
        return {"agent_logs": [f"[L2OrderFlow] Error: {e}"], "errors": [str(e)]}


def _hmm_regime_node(state: TradingState) -> dict[str, Any]:
    """Classify current market regime using HMM."""
    try:
        from risk.hmm_regime_classifier import hmm_classifier
        regime = hmm_classifier.classify_current_regime(
            market_data=state.get("market_data", {}),
            india_vix=state.get("india_vix", 0),
            fii_dii_data=state.get("fii_dii_data", {}),
        )
        return {
            "hmm_regime": regime,
            "agent_logs": [f"[HMMRegime] {regime.get('regime', '?')} (conf={regime.get('confidence', 0):.0%})"],
        }
    except Exception as e:
        return {"agent_logs": [f"[HMMRegime] Error: {e}"], "errors": [str(e)]}


def _geopolitical_nlp_node(state: TradingState) -> dict[str, Any]:
    """Geopolitical NLP risk analysis."""
    try:
        from agents.geopolitical_nlp_agent import geopolitical_nlp_agent as geo_agent
        return geo_agent(state)
    except Exception as e:
        return {"agent_logs": [f"[GeopoliticalNLP] Error: {e}"], "errors": [str(e)]}


def _bull_bear_debate_node(state: TradingState) -> dict[str, Any]:
    """Bull/Bear adversarial debate synthesis."""
    try:
        from agents.bull_bear_synthesis_agent import bull_bear_synthesis_agent as debate_fn
        return debate_fn(state)
    except Exception as e:
        return {"agent_logs": [f"[BullBearDebate] Error: {e}"], "errors": [str(e)]}


def _marl_optimizer_node(state: TradingState) -> dict[str, Any]:
    """MARL + HRP portfolio optimization."""
    try:
        from risk.marl_portfolio_optimizer import marl_optimizer
        import pandas as pd

        returns_data = {}
        for sym, data in state.get("market_data", {}).items():
            if isinstance(data, pd.DataFrame) and "close" in data.columns:
                returns_data[sym] = data["close"].pct_change().dropna()

        if returns_data:
            returns_df = pd.DataFrame(returns_data).dropna()
            allocations = marl_optimizer.get_optimal_allocation(returns_df)
        else:
            allocations = {}

        return {
            "optimal_allocations": allocations,
            "agent_logs": [f"[MARLOptimizer] Allocations for {len(allocations)} symbols"],
        }
    except Exception as e:
        return {"agent_logs": [f"[MARLOptimizer] Error: {e}"], "errors": [str(e)]}


def _delta_hedger_node(state: TradingState) -> dict[str, Any]:
    """AI delta hedging for options positions."""
    try:
        from risk.ai_options_delta_hedger import delta_hedger
        positions = state.get("current_positions", {})
        portfolio_greeks = delta_hedger.compute_portfolio_greeks(
            positions, state.get("live_quotes", {}),
        )
        return {
            "portfolio_greeks": portfolio_greeks,
            "agent_logs": [f"[DeltaHedger] Portfolio delta: {portfolio_greeks.get('portfolio_delta', 0):.2f}"],
        }
    except Exception as e:
        return {"agent_logs": [f"[DeltaHedger] Error: {e}"], "errors": [str(e)]}


def _compliance_check_node(state: TradingState) -> dict[str, Any]:
    """FEMA/LRS compliance gate."""
    try:
        from config.fema_lrs_controller import fema_controller
        lrs_status = fema_controller.get_lrs_status()
        return {
            "lrs_status": lrs_status,
            "agent_logs": [f"[Compliance] LRS remaining: ${lrs_status.get('remaining_usd', 0):,.0f}"],
        }
    except Exception as e:
        return {"agent_logs": [f"[Compliance] Error: {e}"], "errors": [str(e)]}


def _order_router_node(state: TradingState) -> dict[str, Any]:
    """Intelligent order routing across Angel/IBKR/Alpaca."""
    try:
        from broker.intelligent_order_router import order_router
    except ImportError:
        order_router = None

    logs: list[str] = []
    errors: list[str] = []
    executed: list[dict[str, Any]] = []

    decisions = state.get("trading_decisions", [])

    for decision in decisions:
        symbol = decision.get("symbol", "")
        action = decision.get("action", "")
        quantity = decision.get("quantity", 0)
        price = decision.get("price", 0)
        currency = decision.get("currency", "INR")

        if not symbol or not action or quantity <= 0:
            continue

        if order_router and currency != "INR":
            # Use intelligent router for global trades
            try:
                report = order_router.execute_order(
                    symbol=symbol, action=action, quantity=quantity,
                    price=price, currency=currency,
                )
                executed.append({
                    "symbol": symbol, "action": action, "quantity": report.filled_qty,
                    "fill_price": report.avg_price, "broker": report.broker,
                    "algo": report.algo, "slippage_bps": report.slippage_bps,
                })
                logs.append(f"[OrderRouter] {action} {quantity}x {symbol} via {report.broker}")
                continue
            except Exception as e:
                errors.append(f"[OrderRouter] Global routing failed for {symbol}: {e}")

        # Fall back to Angel One for Indian instruments
        token = get_token(symbol)
        if not token:
            errors.append(f"[OrderRouter] Token not found for {symbol}")
            continue

        try:
            result = place_order(
                symbol=symbol, token=token, action=action,
                quantity=quantity,
                order_type=decision.get("order_type", "LIMIT"),
                price=price,
                stop_loss=decision.get("stop_loss"),
                target=decision.get("target"),
                product_type="INTRADAY" if decision.get("holding_period") == "INTRADAY" else "DELIVERY",
                strategy=decision.get("strategy", "agent_pipeline"),
                confidence=decision.get("confidence", 0),
                reasoning=decision.get("reasoning", ""),
            )
            if result:
                executed.append(result)
                logs.append(
                    f"[OrderRouter] {'PAPER' if result.get('is_paper') else 'LIVE'} "
                    f"{action} {quantity}x {symbol} @ ₹{result.get('fill_price', price)}"
                )
            else:
                errors.append(f"[OrderRouter] Order failed for {symbol}")
        except Exception as exc:
            errors.append(f"[OrderRouter] {symbol} execution error: {exc}")

    return {"executed_orders": executed, "agent_logs": logs, "errors": errors}


def _episodic_reflection_node(state: TradingState) -> dict[str, Any]:
    """Post-execution episodic reflection."""
    try:
        from agents.episodic_reflection_agent import episodic_reflection_agent as reflect_fn
        return reflect_fn(state)
    except Exception as e:
        return {"agent_logs": [f"[EpisodicReflection] Error: {e}"], "errors": [str(e)]}


def build_trading_graph() -> Any:
    """Build and compile the LangGraph trading pipeline (v2.0).

    Flow:
        market_data → [kafka_events, l2_order_flow] (parallel data ingestion)
        → hmm_regime (regime classification)
        → [technical, fundamental, sentiment, options, macro, india_specific, geopolitical_nlp] (parallel analysis)
        → bull_bear_debate (adversarial synthesis)
        → risk_manager
        → [marl_optimizer, delta_hedger] (parallel portfolio optimization)
        → compliance_check (FEMA/LRS gate)
        → portfolio_manager
        → order_router (intelligent multi-broker execution)
        → episodic_reflection (post-trade learning)
        → END
    """
    if StateGraph is None:
        logger.error("langgraph not installed. Cannot build graph.")
        return None

    workflow = StateGraph(TradingState)

    # ── Data Ingestion Layer ─────────────────────────────
    workflow.add_node("market_data", market_data_agent)
    workflow.add_node("kafka_events", _kafka_events_node)
    workflow.add_node("l2_order_flow", _l2_order_flow_node)

    # ── Regime Classification ────────────────────────────
    workflow.add_node("hmm_regime", _hmm_regime_node)

    # ── Quantitative Screener (pre-LLM filter) ──────────
    workflow.add_node("quant_screener", quantitative_screener_node)

    # ── Analyst Agents (parallel fan-out) ────────────────
    workflow.add_node("technical", technical_agent)
    workflow.add_node("fundamental", fundamental_agent)
    workflow.add_node("sentiment", sentiment_agent)
    workflow.add_node("options", options_agent)
    workflow.add_node("macro", macro_agent)
    workflow.add_node("india_specific", india_specific_agent)
    workflow.add_node("geopolitical_nlp", _geopolitical_nlp_node)

    # ── Synthesis Layer ──────────────────────────────────
    workflow.add_node("bull_bear_debate", _bull_bear_debate_node)

    # ── Risk + Optimization ──────────────────────────────
    workflow.add_node("risk_manager", risk_manager_agent)
    workflow.add_node("marl_optimizer", _marl_optimizer_node)
    workflow.add_node("delta_hedger", _delta_hedger_node)

    # ── Compliance ───────────────────────────────────────
    workflow.add_node("compliance_check", _compliance_check_node)

    # ── Final Decision + Execution ───────────────────────
    workflow.add_node("portfolio_manager", portfolio_manager_agent)
    workflow.add_node("order_router", _order_router_node)
    workflow.add_node("episodic_reflection", _episodic_reflection_node)

    # ── Entry ────────────────────────────────────────────
    workflow.set_entry_point("market_data")

    # ── Edges ────────────────────────────────────────────

    # market_data → parallel data enrichment
    workflow.add_edge("market_data", "kafka_events")
    workflow.add_edge("market_data", "l2_order_flow")

    # Data enrichment → HMM regime
    workflow.add_edge("kafka_events", "hmm_regime")
    workflow.add_edge("l2_order_flow", "hmm_regime")

    # HMM regime → quant screener (pre-LLM filter)
    workflow.add_edge("hmm_regime", "quant_screener")

    # Quant screener → all analyst agents (parallel fan-out)
    workflow.add_edge("quant_screener", "technical")
    workflow.add_edge("quant_screener", "fundamental")
    workflow.add_edge("quant_screener", "sentiment")
    workflow.add_edge("quant_screener", "options")
    workflow.add_edge("quant_screener", "macro")
    workflow.add_edge("quant_screener", "india_specific")
    workflow.add_edge("quant_screener", "geopolitical_nlp")

    # All analysts → bull_bear_debate (fan-in)
    workflow.add_edge("technical", "bull_bear_debate")
    workflow.add_edge("fundamental", "bull_bear_debate")
    workflow.add_edge("sentiment", "bull_bear_debate")
    workflow.add_edge("options", "bull_bear_debate")
    workflow.add_edge("macro", "bull_bear_debate")
    workflow.add_edge("india_specific", "bull_bear_debate")
    workflow.add_edge("geopolitical_nlp", "bull_bear_debate")

    # Debate → risk_manager → parallel optimization
    workflow.add_edge("bull_bear_debate", "risk_manager")
    workflow.add_edge("risk_manager", "marl_optimizer")
    workflow.add_edge("risk_manager", "delta_hedger")

    # Optimization → compliance → portfolio_manager → execution → reflection → END
    workflow.add_edge("marl_optimizer", "compliance_check")
    workflow.add_edge("delta_hedger", "compliance_check")
    workflow.add_edge("compliance_check", "portfolio_manager")
    workflow.add_edge("portfolio_manager", "order_router")
    workflow.add_edge("order_router", "episodic_reflection")
    workflow.add_edge("episodic_reflection", END)

    graph = workflow.compile()
    logger.info("Trading graph v2.1 compiled successfully ({} nodes).", 21)
    return graph


async def run_trading_pipeline(watchlist: list[str] | None = None) -> TradingState:
    """Run the full trading pipeline once.

    Args:
        watchlist: Optional override of default watchlist.

    Returns:
        Final TradingState after all agents have run.
    """
    graph = build_trading_graph()
    if graph is None:
        logger.error("Cannot run pipeline — graph not built.")
        return create_initial_state(watchlist)

    initial_state = create_initial_state(watchlist)

    logger.info("=" * 60)
    logger.info("TRADING PIPELINE START — {} symbols", len(initial_state["watchlist"]))
    logger.info("=" * 60)

    try:
        final_state = await graph.ainvoke(initial_state)

        # Log summary
        decisions = final_state.get("trading_decisions", [])
        executed = final_state.get("executed_orders", [])
        errs = final_state.get("errors", [])

        logger.info("Pipeline complete: {} decisions, {} executed, {} errors", len(decisions), len(executed), len(errs))

        if errs:
            for e in errs:
                logger.warning("Pipeline error: {}", e)

        return final_state

    except Exception as exc:
        logger.exception("Trading pipeline failed: {}", exc)
        return create_initial_state(watchlist)
