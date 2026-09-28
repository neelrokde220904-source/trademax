"""News fetcher — NewsAPI + RSS feeds (Economic Times, Moneycontrol, NSE)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import feedparser
import httpx
import pytz
from loguru import logger

from config.settings import settings

IST = pytz.timezone("Asia/Kolkata")

# RSS feed URLs
RSS_FEEDS = {
    "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
}

# NSE announcements endpoint
NSE_ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corporate-announcements"
NSE_BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


async def fetch_news_api(query: str, days: int = 3) -> list[dict[str, Any]]:
    """Fetch news articles from NewsAPI.org.

    Args:
        query: Search query (e.g. 'RELIANCE NSE stock')
        days: Number of days to look back

    Returns:
        List of article dicts with title, description, source, published_at, url.
    """
    if not settings.news_api_key or settings.news_api_key.startswith("your_"):
        logger.debug("NewsAPI key not configured, skipping.")
        return []

    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 10,
        "apiKey": settings.news_api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for art in data.get("articles", []):
            articles.append({
                "title": art.get("title", ""),
                "description": art.get("description", ""),
                "source": art.get("source", {}).get("name", ""),
                "published_at": art.get("publishedAt", ""),
                "url": art.get("url", ""),
                "content": art.get("content", ""),
            })
        return articles

    except Exception as exc:
        logger.error("NewsAPI fetch failed for '{}': {}", query, exc)
        return []


async def fetch_rss_feeds() -> list[dict[str, Any]]:
    """Fetch latest articles from RSS feeds (Economic Times, Moneycontrol)."""
    articles = []

    for source, url in RSS_FEEDS.items():
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            feed = feedparser.parse(resp.text)
            for entry in feed.entries[:15]:
                articles.append({
                    "title": entry.get("title", ""),
                    "description": entry.get("summary", ""),
                    "source": source,
                    "published_at": entry.get("published", ""),
                    "url": entry.get("link", ""),
                    "content": entry.get("summary", ""),
                })

            logger.debug("Fetched {} articles from {}.", len(feed.entries[:15]), source)

        except Exception as exc:
            logger.error("RSS fetch failed for {}: {}", source, exc)

    return articles


async def fetch_nse_announcements(symbol: str = "") -> list[dict[str, Any]]:
    """Fetch corporate announcements from NSE India.

    Args:
        symbol: Optional filter by symbol.
    """
    try:
        params = {}
        if symbol:
            params["symbol"] = symbol

        async with httpx.AsyncClient(timeout=15.0, headers=NSE_BASE_HEADERS) as client:
            # First hit NSE homepage to get cookies
            await client.get("https://www.nseindia.com")
            resp = await client.get(NSE_ANNOUNCEMENTS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        announcements = []
        for item in data[:20]:
            announcements.append({
                "title": item.get("desc", ""),
                "symbol": item.get("symbol", ""),
                "source": "NSE",
                "published_at": item.get("an_dt", ""),
                "url": item.get("attchmntFile", ""),
                "content": item.get("desc", ""),
            })

        return announcements

    except Exception as exc:
        logger.error("NSE announcements fetch failed: {}", exc)
        return []


async def fetch_all_news(symbol: str = "") -> list[dict[str, Any]]:
    """Aggregate news from all sources for a symbol.

    Args:
        symbol: The stock symbol to search for.

    Returns:
        Combined list of news articles from all sources.
    """
    tasks = [
        fetch_rss_feeds(),
    ]

    if symbol:
        tasks.append(fetch_news_api(f"{symbol} NSE stock India"))
        tasks.append(fetch_nse_announcements(symbol))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_articles = []
    for result in results:
        if isinstance(result, list):
            all_articles.extend(result)
        elif isinstance(result, Exception):
            logger.error("News fetch error: {}", result)

    # Sort by published_at (newest first)
    all_articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)
    return all_articles


def format_news_for_prompt(articles: list[dict[str, Any]], max_articles: int = 10) -> str:
    """Format news articles for LLM prompt consumption."""
    if not articles:
        return "No recent news articles found."

    lines = []
    for i, art in enumerate(articles[:max_articles], 1):
        lines.append(
            f"{i}. [{art.get('source', 'Unknown')}] {art.get('title', 'No title')}\n"
            f"   {art.get('description', '')[:200]}\n"
            f"   Published: {art.get('published_at', 'Unknown')}"
        )
    return "\n\n".join(lines)
