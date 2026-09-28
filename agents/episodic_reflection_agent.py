"""Episodic Reflection Agent — builds a vector-searchable trade memory and performs root-cause
analysis on losses, generating actionable heuristics that prevent repeated mistakes.

Integrates with ChromaDB for vector storage and retrieval.

Weekly review cycle:
1. Embed all closed trades from the past week
2. Cluster losses by root cause
3. Generate heuristics to prevent recurrence
4. Feed heuristics as soft constraints into portfolio_manager
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from loguru import logger

from config.settings import settings

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "vector_db", "trade_memory")


class TradeEpisode:
    """Represents a single completed trade for reflection."""

    def __init__(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        exit_price: float,
        entry_date: str,
        exit_date: str,
        quantity: int,
        pnl: float,
        pnl_pct: float,
        strategy: str = "",
        agent_signals: dict[str, Any] | None = None,
        market_context: dict[str, Any] | None = None,
    ) -> None:
        self.symbol = symbol
        self.action = action
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.entry_date = entry_date
        self.exit_date = exit_date
        self.quantity = quantity
        self.pnl = pnl
        self.pnl_pct = pnl_pct
        self.strategy = strategy
        self.agent_signals = agent_signals or {}
        self.market_context = market_context or {}

    def to_text(self) -> str:
        """Convert to natural language for embedding."""
        return (
            f"Trade: {self.action} {self.quantity} shares of {self.symbol} "
            f"from {self.entry_date} to {self.exit_date}. "
            f"Entry: ₹{self.entry_price:.2f}, Exit: ₹{self.exit_price:.2f}. "
            f"P&L: ₹{self.pnl:.2f} ({self.pnl_pct:+.2f}%). "
            f"Strategy: {self.strategy}. "
            f"Signals: {json.dumps(self.agent_signals, default=str)[:500]}. "
            f"Context: {json.dumps(self.market_context, default=str)[:500]}."
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return metadata for ChromaDB storage."""
        return {
            "symbol": self.symbol,
            "action": self.action,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "strategy": self.strategy,
            "is_loss": self.pnl < 0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "strategy": self.strategy,
            "agent_signals": self.agent_signals,
            "market_context": self.market_context,
        }


class EpisodicMemory:
    """Vector-searchable trade memory backed by ChromaDB."""

    COLLECTION_NAME = "trade_episodes"

    def __init__(self) -> None:
        self.client = None
        self.collection = None
        if CHROMADB_AVAILABLE:
            try:
                os.makedirs(VECTOR_DB_DIR, exist_ok=True)
                self.client = chromadb.PersistentClient(
                    path=VECTOR_DB_DIR,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self.collection = self.client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("ChromaDB trade memory initialized at {}", VECTOR_DB_DIR)
            except Exception as e:
                logger.warning("ChromaDB init failed, using in-memory fallback: {}", e)
                self._init_fallback()
        else:
            logger.info("ChromaDB not available, using in-memory fallback")
            self._init_fallback()

    def _init_fallback(self) -> None:
        """Simple in-memory storage when ChromaDB unavailable."""
        self._episodes: list[dict[str, Any]] = []
        self.collection = None

    def store_episode(self, episode: TradeEpisode) -> None:
        """Embed and store a trade episode."""
        episode_id = f"{episode.symbol}_{episode.entry_date}_{episode.exit_date}"
        text = episode.to_text()
        metadata = episode.to_metadata()

        if self.collection is not None:
            try:
                self.collection.upsert(
                    ids=[episode_id],
                    documents=[text],
                    metadatas=[metadata],
                )
            except Exception as e:
                logger.warning("ChromaDB upsert failed: {}", e)
                self._episodes.append({"id": episode_id, "text": text, "metadata": metadata})
        else:
            self._episodes.append({"id": episode_id, "text": text, "metadata": metadata})

    def store_episodes_bulk(self, episodes: list[TradeEpisode]) -> None:
        """Store multiple episodes at once."""
        for ep in episodes:
            self.store_episode(ep)

    def query_similar(self, query_text: str, n_results: int = 10,
                      where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Find similar trade episodes. For loss analysis, use where={"is_loss": True}."""
        if self.collection is not None:
            try:
                kwargs: dict[str, Any] = {
                    "query_texts": [query_text],
                    "n_results": min(n_results, self.collection.count() or 1),
                }
                if where:
                    kwargs["where"] = where
                results = self.collection.query(**kwargs)
                episodes = []
                for i in range(len(results["ids"][0])):
                    episodes.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0,
                    })
                return episodes
            except Exception as e:
                logger.warning("ChromaDB query failed: {}", e)

        # Fallback: simple text matching
        matching = []
        query_lower = query_text.lower()
        for ep in self._episodes:
            if where and where.get("is_loss") and not ep.get("metadata", {}).get("is_loss"):
                continue
            if any(word in ep.get("text", "").lower() for word in query_lower.split()[:3]):
                matching.append(ep)
        return matching[:n_results]

    def get_recent_losses(self, days: int = 7) -> list[dict[str, Any]]:
        """Retrieve losses from the past N days."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        if self.collection is not None:
            try:
                results = self.collection.get(
                    where={"$and": [
                        {"is_loss": True},
                        {"exit_date": {"$gte": cutoff}},
                    ]},
                )
                return [
                    {"id": results["ids"][i], "text": results["documents"][i],
                     "metadata": results["metadatas"][i]}
                    for i in range(len(results["ids"]))
                ]
            except Exception:
                pass

        return [
            ep for ep in self._episodes
            if ep.get("metadata", {}).get("is_loss") and ep.get("metadata", {}).get("exit_date", "") >= cutoff
        ]

    @property
    def total_episodes(self) -> int:
        if self.collection is not None:
            try:
                return self.collection.count()
            except Exception:
                pass
        return len(self._episodes)


class EpisodicReflectionAgent:
    """Performs root-cause analysis on trading losses and generates heuristics."""

    # Categories of loss root causes
    ROOT_CAUSE_CATEGORIES = [
        "REGIME_MISMATCH",       # Trading trend strategy in mean-reverting market
        "POSITION_SIZING",       # Too large/small position
        "STOP_LOSS_TOO_TIGHT",   # Stopped out before move completed
        "STOP_LOSS_TOO_WIDE",    # Let loss run too far
        "ENTRY_TIMING",          # Entered too early/late
        "SECTOR_ROTATION",       # Missed sector rotation
        "EVENT_RISK",            # Unexpected news/event
        "LIQUIDITY_TRAP",        # Low liquidity caused slippage
        "CORRELATION_BLOW",      # Correlated positions moved together
        "OVERTRADING",           # Too many trades, death by a thousand cuts
    ]

    def __init__(self) -> None:
        self.memory = EpisodicMemory()
        self._heuristics: list[dict[str, Any]] = []

    def store_completed_trade(self, trade_data: dict[str, Any]) -> None:
        """Store a newly completed trade in the episode memory."""
        episode = TradeEpisode(
            symbol=trade_data.get("symbol", ""),
            action=trade_data.get("action", ""),
            entry_price=trade_data.get("entry_price", 0),
            exit_price=trade_data.get("exit_price", 0),
            entry_date=trade_data.get("entry_date", ""),
            exit_date=trade_data.get("exit_date", datetime.utcnow().strftime("%Y-%m-%d")),
            quantity=trade_data.get("quantity", 0),
            pnl=trade_data.get("pnl", 0),
            pnl_pct=trade_data.get("pnl_pct", 0),
            strategy=trade_data.get("strategy", ""),
            agent_signals=trade_data.get("agent_signals"),
            market_context=trade_data.get("market_context"),
        )
        self.memory.store_episode(episode)

    def run_weekly_reflection(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Weekly root-cause analysis and heuristic generation.

        Returns:
            Dict with root_causes, heuristics, and summary.
        """
        losses = self.memory.get_recent_losses(days=7)

        if not losses:
            logger.info("No losses in past 7 days — skipping reflection")
            return {
                "root_causes": [],
                "heuristics": [],
                "summary": "No losses to analyze",
                "total_episodes": self.memory.total_episodes,
            }

        # Cluster losses by root cause
        root_causes = self._classify_root_causes(losses)

        # Generate heuristics to prevent recurrence
        heuristics = self._generate_heuristics(root_causes, losses)

        # Store heuristics
        self._heuristics.extend(heuristics)

        summary = (
            f"Analyzed {len(losses)} losses. "
            f"Root causes: {', '.join(rc['category'] for rc in root_causes)}. "
            f"Generated {len(heuristics)} new heuristics."
        )

        logger.info("Reflection complete: {}", summary)

        return {
            "root_causes": root_causes,
            "heuristics": heuristics,
            "summary": summary,
            "total_episodes": self.memory.total_episodes,
        }

    def _classify_root_causes(self, losses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Classify each loss into a root cause category."""
        if not ANTHROPIC_AVAILABLE or not settings.anthropic_api_key:
            return self._rule_based_classification(losses)

        loss_summaries = "\n".join(
            f"- {l.get('text', l.get('metadata', {}).get('symbol', 'Unknown'))}"
            for l in losses[:15]
        )

        prompt = f"""Analyze these {len(losses)} trading losses and classify each into root cause categories.

Losses:
{loss_summaries}

Valid root cause categories:
{json.dumps(self.ROOT_CAUSE_CATEGORIES)}

For each loss, determine:
1. The most likely root cause category
2. A specific explanation
3. Severity (1-10)

Output ONLY JSON array:
[{{"loss_index": 0, "category": "REGIME_MISMATCH", "explanation": "...", "severity": 7}}]"""

        try:
            client = Anthropic(api_key=settings.anthropic_api_key)
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else "[]"
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            logger.warning("LLM classification failed: {}", e)

        return self._rule_based_classification(losses)

    def _rule_based_classification(self, losses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fallback rule-based loss classification."""
        results = []
        for i, loss in enumerate(losses):
            meta = loss.get("metadata", {})
            pnl_pct = meta.get("pnl_pct", 0)

            if abs(pnl_pct) > 10:
                category = "STOP_LOSS_TOO_WIDE"
            elif abs(pnl_pct) < 1:
                category = "OVERTRADING"
            else:
                category = "ENTRY_TIMING"

            results.append({
                "loss_index": i,
                "category": category,
                "explanation": f"{meta.get('symbol', '?')} lost {pnl_pct:.1f}%",
                "severity": min(10, max(1, int(abs(pnl_pct)))),
            })
        return results

    def _generate_heuristics(self, root_causes: list[dict[str, Any]],
                             losses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generate actionable heuristics from root cause analysis."""
        if not ANTHROPIC_AVAILABLE or not settings.anthropic_api_key:
            return self._rule_based_heuristics(root_causes)

        causes_text = json.dumps(root_causes[:10], default=str)

        prompt = f"""Based on this root-cause analysis of recent trading losses, generate 
actionable heuristics to prevent recurrence:

Root Causes:
{causes_text}

Generate specific, implementable rules. Each heuristic should be:
1. Specific enough to be coded as a filter
2. Have a clear trigger condition
3. Have a clear action (block, reduce size, require confirmation)

Output ONLY JSON array:
[{{"heuristic": "If VIX > 25 and strategy is momentum, reduce position size by 50%",
  "trigger": "vix > 25 AND strategy == momentum",
  "action": "reduce_size_50pct",
  "root_cause": "REGIME_MISMATCH",
  "confidence": 0.8}}]"""

        try:
            client = Anthropic(api_key=settings.anthropic_api_key)
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else "[]"
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                heuristics = json.loads(text[start:end])
                for h in heuristics:
                    h["generated_at"] = datetime.utcnow().isoformat()
                return heuristics
        except Exception as e:
            logger.warning("LLM heuristic generation failed: {}", e)

        return self._rule_based_heuristics(root_causes)

    def _rule_based_heuristics(self, root_causes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fallback heuristic generation."""
        heuristic_map = {
            "REGIME_MISMATCH": {
                "heuristic": "Check HMM regime before applying momentum strategies",
                "trigger": "regime == HIGH_VOL_MEAN_REVERT AND strategy == momentum",
                "action": "block_trade",
            },
            "STOP_LOSS_TOO_WIDE": {
                "heuristic": "Tighten stop loss to 2x ATR instead of 3x ATR",
                "trigger": "consecutive_losses > 2",
                "action": "tighten_stop_loss",
            },
            "OVERTRADING": {
                "heuristic": "Limit to 5 new trades per day maximum",
                "trigger": "daily_trade_count >= 5",
                "action": "block_new_trades",
            },
            "ENTRY_TIMING": {
                "heuristic": "Require volume confirmation before entry",
                "trigger": "volume < 1.5x average",
                "action": "delay_entry",
            },
            "CORRELATION_BLOW": {
                "heuristic": "Block new position if correlation > 0.7 with existing",
                "trigger": "max_portfolio_correlation > 0.7",
                "action": "block_trade",
            },
        }

        heuristics = []
        seen_categories = set()
        for rc in root_causes:
            cat = rc.get("category", "")
            if cat in heuristic_map and cat not in seen_categories:
                h = heuristic_map[cat].copy()
                h["root_cause"] = cat
                h["confidence"] = 0.6
                h["generated_at"] = datetime.utcnow().isoformat()
                heuristics.append(h)
                seen_categories.add(cat)

        return heuristics

    def get_active_heuristics(self) -> list[dict[str, Any]]:
        """Return heuristics that should be applied to current trading decisions."""
        return [h for h in self._heuristics if h.get("confidence", 0) >= 0.5]

    def find_similar_past_trades(self, symbol: str, context: str = "",
                                 n_results: int = 5) -> list[dict[str, Any]]:
        """Find past trades similar to a proposed trade for reference."""
        query = f"Trade in {symbol}. {context}" if context else f"Trade in {symbol}"
        return self.memory.query_similar(query, n_results=n_results)


# Singleton
episodic_reflection = EpisodicReflectionAgent()


# ─── LangGraph Node Wrapper ───────────────────────────────────────

def episodic_reflection_agent(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: Run episodic reflection after trade execution."""
    try:
        # Store any newly executed orders as episodes
        executed = state.get("executed_orders", [])
        for order in executed:
            if order.get("status") == "COMPLETE" and order.get("exit_price"):
                episodic_reflection.store_completed_trade(order)

        # Check if it's time for weekly reflection (e.g., Friday after market close)
        now = datetime.now()
        if now.weekday() == 4 and now.hour >= 16:  # Friday after 4 PM
            reflection = episodic_reflection.run_weekly_reflection(state)
        else:
            reflection = {
                "heuristics": episodic_reflection.get_active_heuristics(),
                "summary": "Applied existing heuristics",
                "total_episodes": episodic_reflection.memory.total_episodes,
            }

        # Find similar past trades for current watchlist symbols
        similar_trades: dict[str, list[dict[str, Any]]] = {}
        for symbol in state.get("watchlist", [])[:5]:
            past = episodic_reflection.find_similar_past_trades(symbol)
            if past:
                similar_trades[symbol] = past

        return {
            "agent_logs": state.get("agent_logs", []) + [
                f"[EpisodicReflection] {reflection.get('summary', 'Done')}. "
                f"Active heuristics: {len(reflection.get('heuristics', []))}. "
                f"Total episodes: {reflection.get('total_episodes', 0)}."
            ],
            "reflection_heuristics": reflection.get("heuristics", []),
            "similar_past_trades": similar_trades,
        }

    except Exception as e:
        logger.error("Episodic reflection agent error: {}", e)
        return {
            "agent_logs": state.get("agent_logs", []) + [
                f"[EpisodicReflection] Error: {e}"
            ],
        }
