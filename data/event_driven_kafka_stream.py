"""Event-Driven Kafka Stream — replaces scheduled polling with sub-second event delivery.

Routes macro events, corporate announcements, and regulatory changes to appropriate agents.
Kafka is OPTIONAL — falls back to scheduled polling if not configured.

Required: confluent-kafka, aiohttp
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime
from typing import Any, Callable

import aiohttp
from loguru import logger

from config.settings import settings

try:
    from confluent_kafka import Consumer, KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.info("confluent-kafka not installed. Event streaming will use polling fallback.")


# Event types that require immediate latency-arbitrage processing
LATENCY_ARBITRAGE_EVENTS = frozenset({
    "RBI_RATE_DECISION",
    "US_CPI_RELEASE",
    "US_FOMC_DECISION",
    "INDIA_GDP_RELEASE",
    "INDIA_CPI_RELEASE",
    "FII_DII_PROVISIONAL",
    "NIFTY_INDEX_REBALANCE",
    "SEBI_CIRCULAR",
    "CORPORATE_EARNINGS_BEAT_MISS",
})


class EventDrivenKafkaStream:
    """Persistent event stream consumer for institutional data feeds.

    Data sources (implement in order of cost/complexity):
    1. FREE: NSE announcements API
    2. FREE: RBI press releases RSS
    3. FREE: Economic Times + Moneycontrol RSS (already in v1.0)
    4. PAID: Trading Economics API (350+ global indicators)
    5. ENTERPRISE: Bloomberg B-PIPE / Dow Jones Elementized News Feed
    """

    def __init__(self) -> None:
        self._kafka_enabled = False
        self._consumer = None
        self._event_queue: deque[dict[str, Any]] = deque(maxlen=1000)
        self._critical_events: deque[dict[str, Any]] = deque(maxlen=100)
        self._running = False

    def init_kafka(self, bootstrap_servers: str | None = None,
                   group_id: str = "india-ai-trader") -> bool:
        """Initialize Kafka consumer. Returns True if successful."""
        if not KAFKA_AVAILABLE:
            logger.info("Kafka not available. Using polling fallback.")
            return False

        servers = bootstrap_servers or getattr(settings, "kafka_bootstrap_servers", "")
        if not servers:
            logger.info("Kafka not configured (no bootstrap servers). Using polling fallback.")
            return False

        try:
            self._consumer = Consumer({
                "bootstrap.servers": servers,
                "group.id": group_id,
                "auto.offset.reset": "latest",
                "enable.auto.commit": True,
                "session.timeout.ms": 30000,
            })
            self._consumer.subscribe([
                "macro-events", "corporate-events", "regulatory-events",
            ])
            self._kafka_enabled = True
            logger.info("Kafka consumer initialized: {}", servers)
            return True
        except Exception as e:
            logger.warning("Kafka init failed: {}. Using polling fallback.", e)
            return False

    async def start_consuming(self, agent_router: Callable | None = None) -> None:
        """Main event loop — consume from Kafka or poll free data sources."""
        self._running = True

        if self._kafka_enabled:
            await self._consume_kafka(agent_router)
        else:
            await self._poll_free_sources(agent_router)

    async def stop(self) -> None:
        """Stop the event stream."""
        self._running = False
        if self._consumer:
            self._consumer.close()

    async def _consume_kafka(self, router: Callable | None) -> None:
        """Consume events from Kafka topics."""
        logger.info("Starting Kafka event consumer loop...")
        while self._running:
            try:
                msg = self._consumer.poll(timeout=0.1)
                if msg is None:
                    await asyncio.sleep(0.01)
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error("Kafka error: {}", msg.error())
                    continue

                event = json.loads(msg.value().decode("utf-8"))
                await self._handle_event(event, router)

            except Exception as e:
                logger.error("Kafka consumer error: {}", e)
                await asyncio.sleep(1)

    async def _poll_free_sources(self, router: Callable | None) -> None:
        """Poll free data sources when Kafka is not configured.

        Checks NSE announcements, RBI press releases every 60 seconds.
        """
        logger.info("Starting polling fallback for event stream...")
        while self._running:
            try:
                events = await self._fetch_free_events()
                for event in events:
                    await self._handle_event(event, router)
            except Exception as e:
                logger.warning("Event polling error: {}", e)
            await asyncio.sleep(60)  # Poll every 60 seconds

    async def _fetch_free_events(self) -> list[dict[str, Any]]:
        """Fetch events from free data sources (NSE, RBI)."""
        events = []
        async with aiohttp.ClientSession() as session:
            # NSE corporate announcements
            nse_events = await self._fetch_nse_announcements(session)
            events.extend(nse_events)

            # RBI press releases
            rbi_events = await self._fetch_rbi_releases(session)
            events.extend(rbi_events)

        return events

    async def _fetch_nse_announcements(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        """Fetch NSE corporate announcements API."""
        url = "https://www.nseindia.com/api/corporate-announcements?index=equities"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [
                        {
                            "type": "CORPORATE_ANNOUNCEMENT",
                            "symbol": item.get("symbol", ""),
                            "subject": item.get("desc", ""),
                            "data": item,
                            "timestamp": datetime.utcnow().isoformat(),
                            "source": "NSE",
                        }
                        for item in (data if isinstance(data, list) else [])[:20]
                    ]
        except Exception as e:
            logger.debug("NSE announcements fetch failed: {}", e)
        return []

    async def _fetch_rbi_releases(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        """Fetch RBI press releases."""
        url = "https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    # Parse HTML for press releases (simplified)
                    text = await resp.text()
                    if "rate" in text.lower() or "policy" in text.lower():
                        return [{
                            "type": "RBI_ANNOUNCEMENT",
                            "data": {"raw_text": text[:500]},
                            "timestamp": datetime.utcnow().isoformat(),
                            "source": "RBI",
                        }]
        except Exception as e:
            logger.debug("RBI releases fetch failed: {}", e)
        return []

    async def _handle_event(self, event: dict[str, Any],
                            router: Callable | None) -> None:
        """Classify and route events to correct agent."""
        event_type = event.get("type", "UNKNOWN")

        if event_type in LATENCY_ARBITRAGE_EVENTS:
            # CRITICAL PATH — route immediately
            self._critical_events.append(event)
            logger.info(
                "CRITICAL EVENT: {} — routing for immediate processing",
                event_type,
            )
            if router:
                try:
                    await router(event, priority="critical")
                except Exception as e:
                    logger.error("Critical event routing failed: {}", e)
        else:
            # Lower priority — queue for next regular scan
            self._event_queue.append(event)

    def get_pending_events(self, max_events: int = 50) -> list[dict[str, Any]]:
        """Get pending events from queue (called by agents during regular scan)."""
        events = []
        while self._event_queue and len(events) < max_events:
            events.append(self._event_queue.popleft())
        return events

    def get_critical_events(self) -> list[dict[str, Any]]:
        """Get unprocessed critical events (latency-sensitive)."""
        events = list(self._critical_events)
        self._critical_events.clear()
        return events

    @property
    def is_kafka_enabled(self) -> bool:
        return self._kafka_enabled

    @property
    def queue_size(self) -> int:
        return len(self._event_queue)


# Singleton
event_stream = EventDrivenKafkaStream()
