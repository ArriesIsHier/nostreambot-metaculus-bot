"""Free news search for the fork's free mode: GDELT (keyless) plus Tavily (free key, optional).

Free mode has no web-search provider at all (Gemini 3 grounding is paid-tier only and the
OpenRouter search providers need a paid key), and most tournament questions turn on news from
the last few days. GDELT's DOC API indexes worldwide news, needs no key and permits commercial
use; Tavily's free plan (1,000 searches a month) adds a search engine with page snippets when
``TAVILY_API_KEY`` is set. Google News and Bing News RSS were rejected: both answer without a key
but restrict their feeds to personal, non-commercial use, which a prize-eligible bot is not.

The section is evidence, not analysis: dated headlines, snippets and a few article excerpts,
newest first, for the forecasters to weigh. Like ``prediction_market`` it returns ``""`` under
``is_benchmarking``, because today's news leaks a past question's outcome.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote, urlparse

import aiohttp
import trafilatura
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    FREE_NEWS_ARTICLE_CHARS,
    FREE_NEWS_ENABLED_ENV,
    FREE_NEWS_GDELT_MAX_RECORDS,
    FREE_NEWS_GDELT_TIMESPAN,
    FREE_NEWS_HTTP_TIMEOUT,
    FREE_NEWS_MAX_ARTICLES,
    FREE_NEWS_MAX_FEED_BYTES,
    FREE_NEWS_MAX_HEADLINES,
    FREE_NEWS_MAX_PAGE_BYTES,
    FREE_NEWS_MAX_QUERIES,
    FREE_NEWS_QUERY_MODEL,
    FREE_NEWS_QUERY_TIMEOUT,
    FREE_NEWS_TAVILY_DAYS,
    FREE_NEWS_TAVILY_MAX_QUERIES,
    FREE_NEWS_TAVILY_MAX_RESULTS,
    FREE_NEWS_WALL_TIMEOUT,
    TAVILY_API_KEY_ENV,
    env_flag_enabled,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.research.http_fetch import BROWSER_HEADERS, build_session, decode_text_body, read_body_capped
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.providers import ResearchCallable

logger = logging.getLogger(__name__)

PROVIDER_NAME = "free_news"
GDELT_DOC_URL = (
    "https://api.gdeltproject.org/api/v2/doc/doc?query={query}%20sourcelang:english"
    "&mode=artlist&format=json&maxrecords={limit}&timespan={timespan}&sort=datedesc"
)
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
# GDELT asks for at most one request per 5 seconds per client.
GDELT_MIN_INTERVAL_S = 5.5
GDELT_ATTEMPTS = 2
GDELT_RETRY_BACKOFF_S = 6.0

_SPACE_RE = re.compile(r"\s+")
# Question-shaped words that only dilute a news query.
_QUESTION_WORDS = re.compile(
    r"\b(will|what|which|who|when|how|many|much|the|a|an|be|is|are|by|before|after|on|in|of|to|"
    r"for|than|more|less|at|least|most|between|and|or|does|do|did|has|have)\b",
    re.IGNORECASE,
)

_QUERY_PROMPT = """You write news search queries for a forecaster.

Question: {title}

Write {n} different short search queries (2 to 6 words each) that would surface the most
recent news articles needed to forecast this question: the main entity and event, plus the
key driver or scheduled decision. No quotes, no operators, no dates, no numbering.
One query per line, nothing else."""


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: str
    published: datetime | None
    snippet: str
    url: str
    origin: str


def _squash(text: str | None) -> str:
    return _SPACE_RE.sub(" ", text or "").strip()


def _parse_gdelt_date(value: str | None) -> datetime | None:
    """GDELT's ``seendate`` is ``YYYYMMDDTHHMMSSZ``."""
    try:
        return datetime.strptime(value or "", "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def _parse_iso_or_http_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_gdelt_articles(payload: Any) -> list[NewsItem]:
    """The ``articles`` of a GDELT DOC ``artlist`` JSON payload; anything malformed yields none."""
    articles = payload.get("articles") if isinstance(payload, dict) else None
    if not isinstance(articles, list):
        return []
    items: list[NewsItem] = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        title, url = _squash(article.get("title")), str(article.get("url") or "")
        if title and url.startswith("http"):
            items.append(
                NewsItem(
                    title=title,
                    source=_squash(article.get("domain")),
                    published=_parse_gdelt_date(article.get("seendate")),
                    snippet="",
                    url=url,
                    origin="gdelt",
                )
            )
    return items


def parse_tavily_results(payload: Any) -> list[NewsItem]:
    """The ``results`` of a Tavily search payload; anything malformed yields none."""
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    items: list[NewsItem] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        title, url = _squash(result.get("title")), str(result.get("url") or "")
        if title and url.startswith("http"):
            items.append(
                NewsItem(
                    title=title,
                    source=urlparse(url).netloc.removeprefix("www."),
                    published=_parse_iso_or_http_date(result.get("published_date")),
                    snippet=_squash(result.get("content"))[:FREE_NEWS_ARTICLE_CHARS],
                    url=url,
                    origin="tavily",
                )
            )
    return items


def fallback_queries(title: str) -> list[str]:
    """One keyword query from the title, for when the query author returns nothing usable."""
    words = _QUESTION_WORDS.sub(" ", re.sub(r"[^\w\s$%.-]", " ", title))
    query = _SPACE_RE.sub(" ", words).strip()
    return [" ".join(query.split()[:8])] if query else []


def parse_queries(text: str, limit: int) -> list[str]:
    queries: list[str] = []
    for line in text.splitlines():
        query = re.sub(r"^[\s\-*\d.)]+", "", line).strip().strip('"')
        if query and len(query) <= 120 and query.lower() not in (q.lower() for q in queries):
            queries.append(query)
    return queries[:limit]


async def author_queries(title: str) -> list[str]:
    """Up to ``FREE_NEWS_MAX_QUERIES`` queries from the cheap utility model, title keywords on failure."""
    llm = build_llm_with_openrouter_fallback(
        FREE_NEWS_QUERY_MODEL,
        role="news_query_author",
        temperature=None,
        timeout=FREE_NEWS_QUERY_TIMEOUT,
        allowed_tries=1,
    )
    try:
        text = await asyncio.wait_for(
            llm.invoke(_QUERY_PROMPT.format(title=title, n=FREE_NEWS_MAX_QUERIES)), timeout=FREE_NEWS_QUERY_TIMEOUT
        )
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # the query author is additive; title keywords still search
        logger.warning("FREE_NEWS: query author failed (%s); using title keywords", type(exc).__name__)
        return fallback_queries(title)
    return parse_queries(text, FREE_NEWS_MAX_QUERIES) or fallback_queries(title)


async def _read_text(resp: aiohttp.ClientResponse, max_bytes: int) -> str | None:
    if resp.status != 200:
        logger.info("FREE_NEWS: %s answered HTTP %s", resp.url.host, resp.status)
        return None
    body = await read_body_capped(resp, max_bytes=max_bytes, label=PROVIDER_NAME)
    if body is None:
        return None
    text, _ = decode_text_body(body, resp.headers.get("Content-Type"))
    return text


async def _get_text(session: aiohttp.ClientSession, url: str, max_bytes: int) -> str | None:
    try:
        async with session.get(url) as resp:
            return await _read_text(resp, max_bytes)
    except (TimeoutError, aiohttp.ClientError, OSError, ValueError) as exc:
        logger.info("FREE_NEWS: fetch failed for %s (%s)", urlparse(url).netloc, type(exc).__name__)
        return None


class _GdeltPacer:
    """One gate for the whole process, so concurrent questions share GDELT's request budget.

    Built lazily inside the running loop: an ``asyncio.Lock`` binds to the loop that first
    waits on it, and the test suite runs several loops in one process.
    """

    def __init__(self) -> None:
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_request = 0.0

    def lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock, self._loop, self._last_request = asyncio.Lock(), loop, 0.0
        return self._lock

    async def wait_turn(self) -> None:
        loop = asyncio.get_running_loop()
        wait = GDELT_MIN_INTERVAL_S - (loop.time() - self._last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = loop.time()


_GDELT_PACER = _GdeltPacer()


async def search_gdelt(session: aiohttp.ClientSession, query: str) -> list[NewsItem]:
    """GDELT DOC artlist for ``query``, paced to GDELT's published rate limit."""
    url = GDELT_DOC_URL.format(query=quote(query), limit=FREE_NEWS_GDELT_MAX_RECORDS, timespan=FREE_NEWS_GDELT_TIMESPAN)
    for attempt in range(1, GDELT_ATTEMPTS + 1):
        async with _GDELT_PACER.lock():
            await _GDELT_PACER.wait_turn()
            text = await _get_text(session, url, FREE_NEWS_MAX_FEED_BYTES)
        if text:
            try:
                return parse_gdelt_articles(json.loads(text))
            except json.JSONDecodeError:
                # GDELT answers a rate-limit or a malformed query with plain text, not JSON.
                logger.info("FREE_NEWS: GDELT returned non-JSON for %r: %s", query, text[:120])
                if "limit requests" not in text:
                    return []
        if attempt < GDELT_ATTEMPTS:
            # A shared runner IP can trip GDELT's per-client limit; one paced retry usually clears it.
            await asyncio.sleep(GDELT_RETRY_BACKOFF_S)
    return []


async def search_tavily(session: aiohttp.ClientSession, query: str, api_key: str) -> list[NewsItem]:
    payload = {
        "query": query,
        "topic": "news",
        "days": FREE_NEWS_TAVILY_DAYS,
        "max_results": FREE_NEWS_TAVILY_MAX_RESULTS,
        "search_depth": "basic",
        "include_answer": False,
    }
    try:
        async with session.post(
            TAVILY_SEARCH_URL, json=payload, headers={"Authorization": f"Bearer {api_key}"}
        ) as resp:
            status = resp.status
            text = await _read_text(resp, FREE_NEWS_MAX_FEED_BYTES)
    except (TimeoutError, aiohttp.ClientError, OSError, ValueError) as exc:
        logger.info("FREE_NEWS: Tavily failed (%s)", type(exc).__name__)
        return []
    if not text:
        logger.info("FREE_NEWS: Tavily answered HTTP %s", status)
        return []
    try:
        return parse_tavily_results(json.loads(text))
    except json.JSONDecodeError:
        return []


def merge_items(batches: list[list[NewsItem]], limit: int) -> list[NewsItem]:
    """Dedupe by normalized headline and by URL across sources and queries, newest first."""
    seen: set[str] = set()
    merged: list[NewsItem] = []
    for item in (item for batch in batches for item in batch):
        keys = {re.sub(r"\W+", "", item.title.lower())[:80], item.url}
        if not keys & seen:
            seen |= keys
            merged.append(item)
    oldest = datetime.min.replace(tzinfo=UTC)
    merged.sort(key=lambda item: item.published or oldest, reverse=True)
    return merged[:limit]


def _article_excerpt(page_html: str) -> str:
    text = trafilatura.extract(page_html, include_comments=False, include_tables=False) or ""
    text = _SPACE_RE.sub(" ", text).strip()
    return text[:FREE_NEWS_ARTICLE_CHARS] + ("…" if len(text) > FREE_NEWS_ARTICLE_CHARS else "")


def render_section(items: list[NewsItem], excerpts: dict[str, str], queries: list[str]) -> str:
    today = datetime.now(UTC).date().isoformat()
    origins = sorted({item.origin for item in items})
    lines = [
        f"## Recent News (free search: {', '.join(origins)})",
        f"Retrieved {today} for queries: {'; '.join(queries)}. Headlines are leads, not verified facts:"
        " check each against the question's resolution criteria and dates, and weigh recency.",
        "",
        "### Headlines (newest first)",
    ]
    for item in items:
        date = item.published.date().isoformat() if item.published else "undated"
        source = f" ({item.source})" if item.source else ""
        snippet = f" — {item.snippet[:300]}" if item.snippet else ""
        lines.append(f"- {date}: {item.title}{source} <{item.url}>{snippet}")
    if excerpts:
        lines += ["", "### Article excerpts"]
        for item in items:
            excerpt = excerpts.get(item.url)
            if excerpt:
                date = item.published.date().isoformat() if item.published else "undated"
                lines += [f"#### {item.title} ({item.source or 'unknown outlet'}, {date})", excerpt, ""]
    return "\n".join(lines).rstrip()


async def search_news(title: str) -> tuple[str, dict[str, int]]:
    """The rendered section for ``title`` plus its counts; ``""`` when nothing was found."""
    queries = await author_queries(title)
    counts = {"queries": len(queries), "gdelt": 0, "tavily": 0, "headlines": 0, "articles": 0}
    if not queries:
        return "", counts
    tavily_key = os.getenv(TAVILY_API_KEY_ENV)
    async with build_session(timeout_s=FREE_NEWS_HTTP_TIMEOUT, headers=BROWSER_HEADERS) as session:
        searches = [search_gdelt(session, query) for query in queries]
        if tavily_key:
            searches += [search_tavily(session, q, tavily_key) for q in queries[:FREE_NEWS_TAVILY_MAX_QUERIES]]
        batches = await asyncio.gather(*searches)
        for batch in batches:
            for item in batch:
                counts[item.origin] += 1
        items = merge_items(list(batches), FREE_NEWS_MAX_HEADLINES)
        # Tavily already carries page text; read the newest articles that arrived without any.
        readable = [item for item in items if not item.snippet][:FREE_NEWS_MAX_ARTICLES]
        pages = await asyncio.gather(*(_get_text(session, item.url, FREE_NEWS_MAX_PAGE_BYTES) for item in readable))
    excerpts: dict[str, str] = {}
    for item, page in zip(readable, pages, strict=True):
        excerpt = await asyncio.to_thread(_article_excerpt, page) if page else ""
        if excerpt:
            excerpts[item.url] = excerpt
    counts.update(headlines=len(items), articles=len(excerpts))
    if not items:
        return "", counts
    return render_section(items, excerpts, queries), counts


def free_news_provider(is_benchmarking: bool = False) -> ResearchCallable:
    """Factory for the free news provider; soft-fails to ``""`` with its counts recorded."""

    async def _fetch(question: MetaculusQuestion) -> str:
        if is_benchmarking or not env_flag_enabled(FREE_NEWS_ENABLED_ENV):
            return ""
        qid = getattr(question, "id_of_question", None)
        title = getattr(question, "question_text", "") or ""
        try:
            section, counts = await asyncio.wait_for(search_news(title), timeout=FREE_NEWS_WALL_TIMEOUT)
        except TimeoutError:
            logger.warning("FREE_NEWS: wall timeout for qid=%s", qid)
            record_provider_detail(qid, PROVIDER_NAME, {"counts": {"wall_timeout": 1}})
            return ""
        record_provider_detail(qid, PROVIDER_NAME, {"counts": counts})
        logger.info("FREE_NEWS: qid=%s %s", qid, " ".join(f"{k}={v}" for k, v in counts.items()))
        return section

    return _fetch
