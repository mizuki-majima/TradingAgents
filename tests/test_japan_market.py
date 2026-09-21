"""Tokyo-listing support: symbol forms, local-language news, JP macro, conventions.

Yahoo prices ``.T`` symbols as well as it prices US ones, so what a Japanese run
was missing sat around the price path: the symbol forms other vendors spell a
Tokyo listing with, an English-only news feed that returns little or nothing for
one, US-only macro aliases, a coverage check that read every same-day window from
Tokyo as reaching into the future, and proposals quoted in dollars and single
shares. These cover each of those.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest import mock

import pytest

from tradingagents.agents.analysts.news_analyst import _macro_hint
from tradingagents.agents.schemas import _coerce_optional_float
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.dataflows import google_news, interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.date_window import coverage_gap
from tradingagents.dataflows.fred import _resolve_series_id
from tradingagents.dataflows.symbol_utils import normalize_symbol
from tradingagents.portfolio import PortfolioContext


@pytest.mark.unit
class TokyoSymbolFormsTests(unittest.TestCase):
    def test_vendor_suffixes_resolve_to_yahoos_dot_t(self):
        for raw in ("7203.JP", "7203.TYO", "7203.TSE", "7203.TKS", "7203.jpx"):
            self.assertEqual(normalize_symbol(raw), "7203.T", raw)

    def test_yahoo_native_tokyo_symbol_is_unchanged(self):
        self.assertEqual(normalize_symbol("7203.T"), "7203.T")
        self.assertEqual(normalize_symbol("6758.t"), "6758.T")

    def test_jpx_alphanumeric_code_resolves(self):
        # Codes issued since 2024 end in a letter (e.g. 130A).
        self.assertEqual(normalize_symbol("130A.JP"), "130A.T")

    def test_bare_code_is_left_alone_by_default(self):
        # Guessing a market from digits alone would price a different listing.
        self.assertEqual(normalize_symbol("7203"), "7203")

    def test_bare_code_takes_the_configured_suffix(self):
        set_config({"default_exchange_suffix": ".T"})
        self.assertEqual(normalize_symbol("7203"), "7203.T")
        self.assertEqual(normalize_symbol("130A"), "130A.T")

    def test_configured_suffix_accepts_a_leading_dot_or_not(self):
        set_config({"default_exchange_suffix": "T"})
        self.assertEqual(normalize_symbol("7203"), "7203.T")

    def test_configured_suffix_composes_with_the_existing_rules(self):
        # 700 -> 700.HK -> Yahoo's 4-digit padding, in one pass.
        set_config({"default_exchange_suffix": ".HK"})
        self.assertEqual(normalize_symbol("700"), "0700.HK")

    def test_configured_suffix_does_not_touch_symbols_that_have_one(self):
        set_config({"default_exchange_suffix": ".T"})
        self.assertEqual(normalize_symbol("AAPL"), "AAPL")
        self.assertEqual(normalize_symbol("0700.HK"), "0700.HK")
        self.assertEqual(normalize_symbol("BTCUSD"), "BTC-USD")


def _rss(items) -> bytes:
    """A Google News RSS payload from ``(title, published, source)`` triples."""
    body = "".join(
        f"<item><title>{title}</title>"
        f"<link>https://example.invalid/{i}</link>"
        f"<pubDate>{format_datetime(published)}</pubDate>"
        f"<source url='https://example.invalid'>{source}</source></item>"
        for i, (title, published, source) in enumerate(items)
    )
    return f"<rss><channel>{body}</channel></rss>".encode()


def _serving(payload: bytes):
    """Patch the vendor's urlopen to serve ``payload``, recording the URLs asked for."""
    asked: list[str] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return payload

    def _urlopen(request, timeout=None):
        asked.append(request.full_url)
        return _Response()

    return mock.patch.object(google_news, "urlopen", _urlopen), asked


@pytest.mark.unit
class GoogleNewsVendorTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.end = self.now.strftime("%Y-%m-%d")
        self.start = (self.now - timedelta(days=7)).strftime("%Y-%m-%d")

    def test_a_tokyo_ticker_is_searched_in_japanese_by_its_bare_code(self):
        patch, asked = _serving(_rss([("トヨタ、今期最終を8％上方修正", self.now, "株探")]))
        with patch:
            report = google_news.get_news("7203.T", self.start, self.end)
        self.assertIn("ceid=JP%3Aja", asked[0])
        self.assertIn("7203", asked[0])
        # The suffix is dropped: Japanese sources title a story by the bare code.
        self.assertNotIn("7203.T", asked[0].split("&")[0])
        self.assertIn("上方修正", report)
        self.assertIn("株探", report)

    def test_a_us_ticker_keeps_the_english_edition_and_its_whole_symbol(self):
        patch, asked = _serving(_rss([("Apple hits a record", self.now, "Reuters")]))
        with patch:
            report = google_news.get_news("AAPL", self.start, self.end)
        self.assertIn("ceid=US%3Aen", asked[0])
        self.assertIn("AAPL", asked[0])
        self.assertIn("Apple hits a record", report)

    def test_news_locale_pins_the_edition(self):
        set_config({"news_locale": "ja"})
        patch, asked = _serving(_rss([("日経平均が反発", self.now, "ロイター")]))
        with patch:
            google_news.get_news("AAPL", self.start, self.end)
        self.assertIn("ceid=JP%3Aja", asked[0])

    def test_evergreen_quote_pages_are_dropped(self):
        # Yahoo!ファイナンス re-stamps its per-listing quote page with today's
        # date, so unfiltered it outranks every real story in the window.
        patch, _ = _serving(_rss([
            ("トヨタ自動車(株)【7203】：株価・株式情報（夜間PTS含む）", self.now, "Yahoo!ファイナンス"),
            ("トヨタ自動車(株)【7203】：掲示板", self.now, "Yahoo!ファイナンス"),
            ("トヨタ自動車[7203]：2026年3月期 決算短信(適時開示)", self.now, "日本経済新聞"),
        ]))
        with patch:
            report = google_news.get_news("7203.T", self.start, self.end)
        self.assertNotIn("掲示板", report)
        self.assertNotIn("株価・株式情報", report)
        # The 日経 disclosure item uses half-width brackets and must survive.
        self.assertIn("決算短信", report)

    def test_articles_outside_the_window_are_dropped(self):
        old = self.now - timedelta(days=120)
        patch, _ = _serving(_rss([("古い記事", old, "株探")]))
        with patch:
            report = google_news.get_news("7203.T", self.start, self.end)
        self.assertNotIn("古い記事", report)

    def test_a_fetch_failure_is_reported_as_unavailable_not_as_silence(self):
        def _boom(request, timeout=None):
            raise OSError("connection reset")

        with mock.patch.object(google_news, "urlopen", _boom):
            from tradingagents.dataflows.errors import NoMarketDataError

            with self.assertRaises(NoMarketDataError):
                google_news.get_news("7203.T", self.start, self.end)

    def test_global_news_uses_the_locales_own_queries(self):
        set_config({"news_locale": "ja"})
        patch, asked = _serving(_rss([("日銀が政策金利を引き上げ", self.now, "ロイター")]))
        with patch:
            report = google_news.get_global_news(self.end, look_back_days=7, limit=5)
        self.assertTrue(any("%E6%97%A5%E9%8A%80" in url for url in asked))  # 日銀
        self.assertIn("政策金利", report)

    def test_global_news_falls_back_to_the_shared_queries_for_english(self):
        set_config({"news_locale": "en"})
        patch, asked = _serving(_rss([("Fed holds rates", self.now, "Reuters")]))
        with patch:
            google_news.get_global_news(self.end, look_back_days=7, limit=5)
        self.assertTrue(any("Federal+Reserve" in url or "Federal%20Reserve" in url for url in asked))

    def test_the_vendor_is_routable_for_both_news_methods(self):
        for method in ("get_news", "get_global_news"):
            self.assertIn("google_news", interface.VENDOR_METHODS[method])
        self.assertIn("google_news", interface.VENDOR_LIST)


@pytest.mark.unit
class CoverageWindowTests(unittest.TestCase):
    def test_a_window_ending_on_the_local_date_is_not_in_the_future(self):
        """08:00 in Tokyo is still yesterday in UTC (#1364).

        The run's own date is what every other point-in-time guard uses, so a
        same-day window must not be reported as reaching past today just because
        UTC has not caught up.
        """
        tokyo_today = (datetime.now(timezone.utc) + timedelta(hours=9)).strftime("%Y-%m-%d")
        with mock.patch(
            "tradingagents.dataflows.date_window.get_current_date", return_value=tokyo_today
        ):
            gap = coverage_gap((), tokyo_today, tokyo_today, "Google ニュース", "news")
        self.assertIsNone(gap)

    def test_a_genuinely_future_window_is_still_flagged(self):
        future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        gap = coverage_gap((), future, future, "Google ニュース", "news")
        self.assertIsNotNone(gap)
        self.assertIn("extends past today", gap)


@pytest.mark.unit
class JapanContextTests(unittest.TestCase):
    def test_a_tokyo_listing_carries_its_currency_and_lot_size(self):
        context = build_instrument_context("7203.T")
        self.assertIn("JPY", context)
        self.assertIn("100-share", context)

    def test_a_us_listing_carries_no_such_note(self):
        context = build_instrument_context("AAPL")
        self.assertNotIn("100-share", context)

    def test_the_news_analyst_names_japanese_macro_series_for_a_tokyo_ticker(self):
        hint = _macro_hint("7203.T")
        self.assertIn("boj_policy_rate", hint)
        self.assertIn("usdjpy", hint)

    def test_the_news_analyst_keeps_the_us_series_elsewhere(self):
        self.assertIn("core_pce", _macro_hint("AAPL"))
        self.assertNotIn("boj_policy_rate", _macro_hint("AAPL"))

    def test_japanese_macro_aliases_resolve_to_fred_series(self):
        self.assertEqual(_resolve_series_id("boj_policy_rate"), "IRSTCI01JPM156N")
        self.assertEqual(_resolve_series_id("jp_10y"), "IRLTLT01JPM156N")
        self.assertEqual(_resolve_series_id("usdjpy"), "DEXJPUS")
        self.assertEqual(_resolve_series_id("Nikkei 225"), "NIKKEI225")

    def test_a_price_written_with_a_trailing_unit_keeps_its_level(self):
        # A Japanese-language run writes "3,025円" where a US one writes "$3025";
        # dropping the level over its unit would lose a stop the model did state.
        self.assertEqual(_coerce_optional_float("3,025円"), 3025.0)
        self.assertEqual(_coerce_optional_float("2980 JPY"), 2980.0)
        self.assertEqual(_coerce_optional_float("¥3,025"), 3025.0)

    def test_a_percentage_is_still_refused(self):
        self.assertIsNone(_coerce_optional_float("15%"))


@pytest.mark.unit
class PortfolioSymbolMatchingTests(unittest.TestCase):
    """A book and a run may spell one instrument differently.

    Found in a live run: with ``default_exchange_suffix=".T"`` set, analysing
    ``7203`` against a book holding ``7203.T`` rendered "No current position in
    7203" and then listed the same 500 units under "Other positions". The
    Portfolio Manager read that as an inconsistency and told the desk to verify
    the balance before executing — correct of it, and a fault of ours.
    """

    def setUp(self):
        self.book = PortfolioContext.model_validate({
            "cash": 5_000_000.0, "currency": "JPY",
            "positions": [
                {"ticker": "7203.T", "quantity": 500, "average_price": 2850.0},
                {"ticker": "6758.T", "quantity": 100},
            ],
        })

    def test_a_bare_code_finds_the_suffixed_holding(self):
        set_config({"default_exchange_suffix": ".T"})
        held = self.book.position_in("7203")
        self.assertIsNotNone(held)
        self.assertEqual(held.quantity, 500)

    def test_a_vendor_spelling_finds_the_same_holding(self):
        self.assertEqual(self.book.position_in("7203.TYO").quantity, 500)

    def test_the_matched_position_is_not_also_listed_as_another(self):
        set_config({"default_exchange_suffix": ".T"})
        rendered = self.book.render("7203")
        self.assertIn("Current position in 7203.T: 500", rendered)
        self.assertNotIn("No current position", rendered)
        self.assertNotIn("7203.T 500", rendered.split("Other positions:")[-1])

    def test_a_genuinely_absent_name_still_reads_as_absent(self):
        rendered = self.book.render("9432.T")
        self.assertIn("No current position in 9432.T", rendered)

    def test_the_rule_is_not_japan_specific(self):
        crypto = PortfolioContext.model_validate(
            {"positions": [{"ticker": "BTC-USD", "quantity": 2}]})
        self.assertEqual(crypto.position_in("BTCUSD").quantity, 2)
