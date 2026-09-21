"""The Sentiment Analyst reads the feeds that cover the market it is analysing.

StockTwits 404s on a ``.T`` symbol and the English subreddits do not discuss
Tokyo codes, so a Japanese run used to reach the analyst with both of its social
sources reporting "unavailable" while the prompt still explained how to read a
Bullish/Bearish ratio it had never been given. Sources are resolved per market
now; these cover the resolution, the X vendor that covers Tokyo, and that an
absent feed is still reported as an absence rather than as quiet agreement.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from tradingagents.agents.analysts import sentiment_analyst as sentiment
from tradingagents.dataflows import social, x_sentiment
from tradingagents.dataflows.config import set_config


@pytest.mark.unit
class TestVendorResolution:
    def test_a_tokyo_listing_gets_the_feed_that_covers_it(self):
        assert social.resolve_vendor_names("7203.T") == ("x",)

    def test_a_us_listing_keeps_stocktwits_and_reddit(self):
        assert social.resolve_vendor_names("AAPL") == ("stocktwits", "reddit")

    def test_resolution_runs_on_the_canonical_symbol(self):
        # 7203.JP names the same listing, so it must resolve the same way.
        assert social.resolve_vendor_names("7203.JP") == ("x",)

    def test_an_explicit_chain_overrides_the_market_default(self):
        set_config({"social_vendors": "stocktwits,x"})
        assert social.resolve_vendor_names("7203.T") == ("stocktwits", "x")

    def test_an_unknown_vendor_costs_one_source_not_the_run(self):
        set_config({"social_vendors": "stocktwits,nope"})
        assert social.resolve_vendor_names("AAPL") == ("stocktwits",)

    def test_a_fetcher_that_raises_is_reported_as_unavailable(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("upstream is down")

        monkeypatch.setitem(
            social._VENDORS, "stocktwits",
            social._Vendor(label="StockTwits", guidance="g", fetch=_boom),
        )
        set_config({"social_vendors": "stocktwits"})
        sources = social.resolve_social_sources("AAPL", "2026-09-14", "2026-09-21")
        assert "unavailable" in sources[0].text
        assert "RuntimeError" in sources[0].text


def _xai_response(payload: dict):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(payload).encode()

    return _Response()


@pytest.mark.unit
class TestXSentiment:
    @pytest.fixture(autouse=True)
    def _key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "test-key")

    def test_the_window_is_pushed_into_the_search_tool(self, monkeypatch):
        """The property that makes a JP backtest possible: a pinned date range.

        StockTwits and Reddit only serve their latest items, so a historical run
        gets a coverage placeholder from them. x_search takes the window.
        """
        captured = {}

        def _urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data)
            captured["auth"] = request.get_header("Authorization")
            return _xai_response({"output_text": "found 12 posts"})

        monkeypatch.setattr(x_sentiment, "urlopen", _urlopen)
        x_sentiment.fetch_x_posts("7203.T", "2026-05-01", "2026-05-08")

        tool = captured["body"]["tools"][0]
        assert tool["type"] == "x_search"
        assert tool["from_date"] == "2026-05-01"
        assert tool["to_date"] == "2026-05-08"
        assert captured["auth"] == "Bearer test-key"

    def test_a_tokyo_listing_is_searched_in_japanese_by_its_bare_code(self, monkeypatch):
        captured = {}

        def _urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data)
            return _xai_response({"output_text": "ok"})

        monkeypatch.setattr(x_sentiment, "urlopen", _urlopen)
        x_sentiment.fetch_x_posts("7203.T", "2026-09-14", "2026-09-21")
        prompt = captured["body"]["input"][0]["content"]
        assert "Search in Japanese" in prompt
        assert "7203" in prompt

    def test_a_us_listing_is_searched_in_english(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            x_sentiment, "urlopen",
            lambda request, timeout=None: (
                captured.__setitem__("body", json.loads(request.data))
                or _xai_response({"output_text": "ok"})
            ),
        )
        x_sentiment.fetch_x_posts("AAPL", "2026-09-14", "2026-09-21")
        assert "Search in English" in captured["body"]["input"][0]["content"]

    def test_the_block_says_it_is_a_digest_and_carries_its_sources(self, monkeypatch):
        monkeypatch.setattr(x_sentiment, "urlopen", lambda *a, **k: _xai_response({
            "output": [{"content": [{
                "type": "output_text",
                "text": "Read 12 posts: 7 bullish, 3 bearish, 2 neither.",
                "annotations": [{"url": "https://x.com/someone/status/1"}],
            }]}],
            "citations": ["https://x.com/another/status/2"],
        }))
        block = x_sentiment.fetch_x_posts("7203.T", "2026-09-14", "2026-09-21")
        # The analyst must be able to tell this from a raw message stream.
        assert "Digest produced by xAI's X search agent" in block
        assert "7 bullish" in block
        assert "https://x.com/another/status/2" in block
        assert "https://x.com/someone/status/1" in block

    def test_a_missing_key_is_a_placeholder_not_an_exception(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        block = x_sentiment.fetch_x_posts("7203.T", "2026-09-14", "2026-09-21")
        assert block.startswith("<X sentiment unavailable")
        assert "XAI_API_KEY" in block

    def test_an_http_error_is_a_placeholder_not_an_exception(self, monkeypatch):
        from urllib.error import HTTPError

        def _boom(*a, **k):
            raise HTTPError("https://api.x.ai/v1/responses", 429, "Too Many", {}, None)

        monkeypatch.setattr(x_sentiment, "urlopen", _boom)
        block = x_sentiment.fetch_x_posts("7203.T", "2026-09-14", "2026-09-21")
        assert "unavailable" in block and "429" in block

    def test_an_empty_answer_is_reported_rather_than_returned_as_silence(self, monkeypatch):
        monkeypatch.setattr(x_sentiment, "urlopen",
                            lambda *a, **k: _xai_response({"output": []}))
        block = x_sentiment.fetch_x_posts("7203.T", "2026-09-14", "2026-09-21")
        assert "unavailable" in block

    def test_the_post_budget_is_configurable_and_stated(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            x_sentiment, "urlopen",
            lambda request, timeout=None: (
                captured.__setitem__("body", json.loads(request.data))
                or _xai_response({"output_text": "ok"})
            ),
        )
        set_config({"x_sentiment_max_posts": 15, "x_sentiment_model": "grok-4.5"})
        block = x_sentiment.fetch_x_posts("7203.T", "2026-09-14", "2026-09-21")
        assert "15" in captured["body"]["input"][0]["content"]
        assert captured["body"]["model"] == "grok-4.5"
        assert "Up to 15 posts were read" in block


@pytest.mark.unit
class TestSentimentPrompt:
    def test_each_feed_brings_its_own_reading_guidance(self):
        message = sentiment._build_system_message(
            ticker="7203.T", start_date="2026-09-14", end_date="2026-09-21",
            news_block="NEWS",
            social_sources=[social.SocialSource("X posts — via xAI", "weigh it as a digest", "POSTS")],
        )
        assert "weigh it as a digest" in message
        assert "<start_of_xposts>" in message and "POSTS" in message
        # The StockTwits ratio rule must not be given to a run without StockTwits.
        assert "90/10" not in message

    def test_a_stocktwits_run_still_gets_the_ratio_rule(self):
        message = sentiment._build_system_message(
            ticker="AAPL", start_date="a", end_date="b", news_block="NEWS",
            social_sources=[
                social.SocialSource(*(
                    social._VENDORS["stocktwits"].label,
                    social._VENDORS["stocktwits"].guidance,
                    "st",
                ))
            ],
        )
        assert "90/10" in message

    def test_no_social_feed_is_stated_as_an_absence_of_data(self):
        message = sentiment._build_system_message(
            ticker="7203.T", start_date="a", end_date="b",
            news_block="NEWS", social_sources=[],
        )
        assert "No social feed is configured" in message
        assert "absence of data, not an absence of chatter" in message

    def test_the_analyst_is_told_a_placeholder_is_not_a_finding(self):
        message = sentiment._build_system_message(
            ticker="AAPL", start_date="a", end_date="b", news_block="NEWS",
            social_sources=[social.SocialSource("X posts", "g", "<X sentiment unavailable: ...>")],
        )
        assert "could not be read" in message
        assert "never fill the gap from general knowledge" in message


@pytest.mark.unit
def test_the_analyst_fetches_through_the_resolver(monkeypatch):
    """The analyst must not reach past the resolver to a hardcoded feed."""
    calls = []
    monkeypatch.setattr(
        sentiment, "resolve_social_sources",
        lambda ticker, start, end: calls.append((ticker, start, end)) or [],
    )
    monkeypatch.setattr(sentiment.get_news, "func", lambda *a, **k: "news", raising=False)

    llm = mock.Mock()
    llm.invoke.return_value = mock.Mock(content="report", tool_calls=[])
    with mock.patch.object(sentiment, "bind_structured", return_value=None), \
         mock.patch.object(sentiment, "invoke_structured_or_freetext", return_value="report"):
        sentiment.create_sentiment_analyst(llm)({
            "company_of_interest": "7203.T", "trade_date": "2026-09-21",
            "asset_type": "stock", "messages": [],
        })

    assert calls == [("7203.T", "2026-09-14", "2026-09-21")]
