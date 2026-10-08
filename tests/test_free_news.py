"""Fork (ArriesIsHier): the free news provider's parsing, merging and gating, all offline."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from metaculus_bot.research import free_news as fn


def _question(**fields: Any) -> Any:
    """A stand-in carrying only the attributes the provider reads."""
    return SimpleNamespace(**fields)


def _item(title: str, url: str, day: int | None, origin: str = "gdelt", snippet: str = "") -> fn.NewsItem:
    published = datetime(2026, 10, day, tzinfo=UTC) if day else None
    return fn.NewsItem(title=title, source="example.com", published=published, snippet=snippet, url=url, origin=origin)


class TestParsing:
    def test_gdelt_artlist(self):
        payload = {
            "articles": [
                {
                    "url": "https://a.com/1",
                    "title": " Shutdown  looms ",
                    "seendate": "20261007T120000Z",
                    "domain": "a.com",
                },
                {"url": "ftp://bad", "title": "skipped: not http"},
                {"title": "skipped: no url"},
                "skipped: not a dict",
            ]
        }
        items = fn.parse_gdelt_articles(payload)
        assert [(i.title, i.source, i.published, i.origin) for i in items] == [
            ("Shutdown looms", "a.com", datetime(2026, 10, 7, 12, tzinfo=UTC), "gdelt")
        ]

    @pytest.mark.parametrize("payload", [None, [], {"articles": None}, {"articles": "x"}])
    def test_gdelt_malformed_payloads_yield_nothing(self, payload):
        assert fn.parse_gdelt_articles(payload) == []

    def test_tavily_results(self):
        payload = {
            "results": [
                {
                    "title": "Vote set",
                    "url": "https://www.b.org/x",
                    "content": "The vote is Tuesday.",
                    "published_date": "Tue, 06 Oct 2026 09:00:00 GMT",
                }
            ]
        }
        (item,) = fn.parse_tavily_results(payload)
        assert (item.source, item.snippet, item.published, item.origin) == (
            "b.org",
            "The vote is Tuesday.",
            datetime(2026, 10, 6, 9, tzinfo=UTC),
            "tavily",
        )

    def test_queries_are_cleaned_deduped_and_capped(self):
        text = '1. US government shutdown\n- "Senate funding vote"\nus government shutdown\n\nCR deadline\nextra'
        assert fn.parse_queries(text, 3) == ["US government shutdown", "Senate funding vote", "CR deadline"]

    def test_fallback_query_strips_question_words(self):
        assert fn.fallback_queries("Will the US federal government shut down before November 1, 2026?") == [
            "US federal government shut down November 1 2026"
        ]


class TestMergeAndRender:
    def test_dedupes_by_headline_and_url_and_sorts_newest_first(self):
        batches = [
            [_item("Old story", "https://a.com/1", 1), _item("New story", "https://a.com/2", 7)],
            [
                _item("new STORY!", "https://b.com/9", 6),
                _item("Other", "https://a.com/1", 5),
                _item("Undated", "https://c.com", None),
            ],
        ]
        merged = fn.merge_items(batches, 10)
        assert [i.title for i in merged] == ["New story", "Old story", "Undated"]

    def test_render_lists_headlines_and_excerpts(self):
        items = [_item("Vote set", "https://b.org/x", 6, origin="tavily", snippet="The vote is Tuesday.")]
        section = fn.render_section(items, {"https://b.org/x": "Full text."}, ["funding vote"])
        assert "## Recent News (free search: tavily)" in section
        assert "- 2026-10-06: Vote set (example.com) <https://b.org/x> — The vote is Tuesday." in section
        assert "Full text." in section


class TestProviderGating:
    @pytest.mark.asyncio
    async def test_benchmarking_never_searches(self, monkeypatch):
        monkeypatch.setenv("FREE_NEWS_ENABLED", "true")
        with mock.patch.object(fn, "search_news") as search:
            assert await fn.free_news_provider(is_benchmarking=True)(_question(id_of_question=1)) == ""
        search.assert_not_called()

    @pytest.mark.asyncio
    async def test_flag_off_never_searches(self, monkeypatch):
        monkeypatch.delenv("FREE_NEWS_ENABLED", raising=False)
        with mock.patch.object(fn, "search_news") as search:
            assert await fn.free_news_provider()(_question(id_of_question=1)) == ""
        search.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_the_section_and_records_counts(self, monkeypatch):
        monkeypatch.setenv("FREE_NEWS_ENABLED", "true")

        async def fake_search(title):
            return "## Recent News", {"queries": 2, "headlines": 5}

        recorded = {}
        monkeypatch.setattr(fn, "search_news", fake_search)
        monkeypatch.setattr(fn, "record_provider_detail", lambda qid, name, detail: recorded.update({name: detail}))
        question = _question(id_of_question=7, question_text="Will X happen?")
        assert await fn.free_news_provider()(question) == "## Recent News"
        assert recorded == {"free_news": {"counts": {"queries": 2, "headlines": 5}}}

    @pytest.mark.asyncio
    async def test_query_author_failure_falls_back_to_title_keywords(self, monkeypatch):
        class Boom:
            async def invoke(self, prompt):
                raise RuntimeError("down")

        monkeypatch.setattr(fn, "build_llm_with_openrouter_fallback", lambda *a, **k: Boom())
        assert await fn.author_queries("Will Brazil hold a runoff?") == ["Brazil hold runoff"]
