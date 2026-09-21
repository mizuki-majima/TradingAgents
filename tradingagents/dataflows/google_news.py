"""Google News RSS vendor: local-language news for markets Yahoo covers thinly.

Yahoo Finance's news feed is English and US-centric, which is fine for a US
listing and thin to empty everywhere else. Measured over one week on Tokyo
listings, ``get_news`` returned 16 items for ``7203.T`` (about half of them
sector noise), 8 for ``8306.T`` (none about the company), 1 for ``9432.T`` (a
different company entirely), and none at all for ``4063.T``, ``6098.T``,
``7741.T`` and ``2413.T``. The domestic flow those runs never see — 日本経済新聞,
株探, 四季報オンライン, and the 適時開示 disclosure summaries — is indexed by
Google News, which serves a keyless, dated RSS search per locale.

The locale follows the ticker's exchange suffix, so a ``.T`` symbol is searched
in Japanese against Japanese sources while a US symbol keeps its English feed.
Configure it as the news vendor (ahead of yfinance, which then covers whatever
Google returns nothing for)::

    config["data_vendors"]["news_data"] = "google_news,yfinance"

Like the Yahoo and Reddit feeds this one serves what is indexed *now*, so a
historical window it cannot reach is reported through the shared
``coverage_gap`` placeholder rather than as an absence of news.
"""

from __future__ import annotations

import contextlib
import http.client
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dateutil.relativedelta import relativedelta

from .config import get_config
from .date_window import coverage_gap, in_window
from .errors import NoMarketDataError
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

_SEARCH = "https://news.google.com/rss/search?{qs}"
_UA = "tradingagents/0.5 (+https://github.com/TauricResearch/TradingAgents)"
_TIMEOUT = 15.0

# Finance portals publish one evergreen page per listing — the quote page, the
# message board, the chart, the 株主優待 table — and Google re-stamps each with
# today's date, so unfiltered they arrive newest-first and fill the window ahead
# of every actual story.
#
# A portal page is recognised by BOTH halves: the listing's code in brackets and
# one of the fixed page labels. Requiring both is what keeps 日本経済新聞's
# disclosure items ("トヨタ自動車[7203]：2026年3月期 決算短信") — which carry the
# code the same way but no page label — out of the filter.
_CODE_IN_BRACKETS = re.compile(r"[【\[(（][0-9]{3}[0-9A-Z][】\])）]")
_PAGE_LABELS = (
    "株価・株式", "株価チャート", "株価・チャート", "掲示板", "適時開示情報",
    "現物信用売買内訳", "株つぶやき", "株主優待", "企業情報", "売買予想",
    "業績・財務", "個人投資家の予想", "時系列", "株価診断", "配当情報",
    "板気配", "今の株価の理由", "リアルタイムチャート",
    "Stock Price", "Share Price", "Stock Quote",
)
# Titles that are a page label end to end, with or without a bracketed code.
_ALWAYS_EVERGREEN = (
    re.compile(r"株価・株式情報"),
    re.compile(r"株価・チャート・企業概要"),
    re.compile(r"Stock Price & Latest News"),
)


@dataclass(frozen=True)
class _Locale:
    """One Google News edition, plus how to phrase a ticker query in it."""

    hl: str
    gl: str
    ceid: str
    ticker_query: str
    label: str
    strip_suffix: bool


_LOCALES = {
    "ja": _Locale(
        hl="ja", gl="JP", ceid="JP:ja",
        # The bare 4-character code is how Japanese sources title a story about a
        # listing, and quoting it keeps the match off same-numbered noise. The
        # company name is deliberately not used: the vendors reachable here
        # report it in English ("Toyota Motor Corporation"), which retrieves the
        # company's own English press pages rather than the market coverage.
        # Measured across nine Tokyo listings, this template returned about half
        # the in-window articles a bare code does but almost none of its
        # off-domain matches (a local events page, a baseball report) — the code
        # alone is a weak query term. What remains beside the company's own news
        # is index and policy coverage, which a Tokyo decision wants anyway.
        # ``news_query_template`` overrides it.
        ticker_query='"{term}" 株価',
        label="Google ニュース",
        strip_suffix=True,
    ),
    "en": _Locale(
        hl="en-US", gl="US", ceid="US:en",
        ticker_query='"{term}" stock',
        label="Google News",
        strip_suffix=False,
    ),
}

# Exchange suffix -> locale. Anything unlisted reads in English, which is what
# the international wires publish in.
_SUFFIX_LOCALES = {".T": "ja"}

_DEFAULT_LOCALE = "en"


def _resolve_locale(symbol: str | None) -> tuple[str, _Locale]:
    """The locale for ``symbol``: the configured one, else its exchange's.

    ``news_locale`` pins every search to one edition; the default ``"auto"``
    derives it per symbol, and falls back to English for the global-news call,
    which names no instrument.
    """
    configured = str(get_config().get("news_locale") or "auto").strip().lower()
    if configured in _LOCALES:
        return configured, _LOCALES[configured]
    if symbol:
        upper = symbol.upper()
        for suffix, name in _SUFFIX_LOCALES.items():
            if upper.endswith(suffix):
                return name, _LOCALES[name]
    return _DEFAULT_LOCALE, _LOCALES[_DEFAULT_LOCALE]


def _published(item: ET.Element) -> datetime | None:
    """An item's ``pubDate`` (RFC 822) as a datetime, or None when unusable."""
    raw = item.findtext("pubDate")
    if not raw:
        return None
    with contextlib.suppress(ValueError, TypeError):
        return parsedate_to_datetime(raw)
    return None


def _is_evergreen(title: str) -> bool:
    """Whether a title names a portal's per-listing page rather than a story."""
    if any(pattern.search(title) for pattern in _ALWAYS_EVERGREEN):
        return True
    return bool(_CODE_IN_BRACKETS.search(title)) and any(
        label in title for label in _PAGE_LABELS
    )


def _recency_operator(start_date: str) -> str:
    """Google News' ``when:Nd``, covering ``start_date`` through today.

    Without it the search is ranked by relevance alone and answers a one-week
    question with a year of coverage, most of which the window then discards —
    measured on Tokyo listings, it roughly triples the in-window article count.
    A window that ended long ago widens the fetch instead of narrowing it; the
    window filter still does the trimming, and the usual coverage placeholder
    still says the feed could not observe it.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    days = (datetime.now(timezone.utc).date() - start).days + 1
    return f" when:{min(max(days, 1), 365)}d"


def _search(query: str, locale: _Locale) -> list[dict]:
    """Run one RSS search and return its items, newest first.

    Google ranks by relevance, not recency, so the caller cannot read the tail
    of the list as the edge of coverage — see ``_gap`` below.
    """
    qs = urlencode({"q": query, "hl": locale.hl, "gl": locale.gl, "ceid": locale.ceid})
    req = Request(_SEARCH.format(qs=qs), headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=_TIMEOUT) as resp:
            root = ET.fromstring(resp.read())
    except (OSError, http.client.HTTPException, ET.ParseError) as exc:
        raise NoMarketDataError(
            query, query, f"Google News search failed: {type(exc).__name__}: {exc}"
        ) from exc

    items = []
    for element in root.findall("./channel/item"):
        title = (element.findtext("title") or "").strip()
        if not title or _is_evergreen(title):
            continue
        items.append({
            "title": title,
            "publisher": (element.findtext("source") or "Unknown").strip(),
            "link": (element.findtext("link") or "").strip(),
            "pub_date": _published(element),
        })
    items.sort(key=lambda a: (a["pub_date"] is not None, a["pub_date"]), reverse=True)
    return items


def _render(heading: str, articles: list[dict]) -> str:
    body = ""
    for article in articles:
        body += f"### {article['title']} (source: {article['publisher']})\n"
        if article["pub_date"]:
            body += f"Published: {article['pub_date']:%Y-%m-%d}\n"
        if article["link"]:
            body += f"Link: {article['link']}\n"
        body += "\n"
    return f"{heading}\n\n{body}"


def _gap(start_date: str, end_date: str, source: str, subject: str) -> str | None:
    """Coverage placeholder for a window this feed cannot vouch for.

    No dates are passed: results are relevance-ranked and merged across queries,
    so the oldest item returned says nothing about how far back coverage reaches.
    Only the present bounds the window, exactly as for the merged Yahoo global
    search.
    """
    return coverage_gap((), start_date, end_date, source, subject)


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Ticker news from Google News, searched in the listing's own language."""
    canonical = normalize_symbol(ticker)
    name, locale = _resolve_locale(canonical)
    term = canonical.rsplit(".", 1)[0] if locale.strip_suffix and "." in canonical else canonical

    template = get_config().get("news_query_template") or locale.ticker_query
    query = template.format(term=term) + _recency_operator(start_date)
    articles = _search(query, locale)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    limit = get_config()["news_article_limit"]
    kept = [a for a in articles if in_window(a["pub_date"], start_dt, end_dt)][:limit]

    resolved = "" if canonical == ticker.upper() else f" (resolved to {canonical})"
    if not kept:
        gap = _gap(start_date, end_date, f"{locale.label} [{name}]", f"news for {ticker}{resolved}")
        return gap or (
            f"No {locale.label} results for {ticker}{resolved} between "
            f"{start_date} and {end_date}"
        )
    return _render(
        f"## {ticker}{resolved} News — {locale.label} [{name}], "
        f"from {start_date} to {end_date}:",
        kept,
    )


def get_global_news(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Macro/market headlines from Google News, in the configured locale.

    The search terms come from ``global_news_queries_by_locale[locale]`` when
    that locale has its own set, else from ``global_news_queries`` — so pinning
    ``news_locale`` to ``"ja"`` also moves the macro read onto 日銀 / 日経平均
    coverage instead of translating a Fed-shaped query.
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]

    name, locale = _resolve_locale(None)
    queries = config.get("global_news_queries_by_locale", {}).get(
        name, config["global_news_queries"]
    )

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    kept: list[dict] = []
    seen: set[str] = set()
    recency = _recency_operator(start_date)
    for query in queries:
        try:
            found = _search(query + recency, locale)
        except NoMarketDataError as exc:
            # One dead query must not sink a macro read the other four can serve.
            logger.warning("Google News query %r failed: %s", query, exc)
            continue
        for article in found:
            if not in_window(article["pub_date"], start_dt, curr_dt):
                continue
            if article["title"] in seen:
                continue
            seen.add(article["title"])
            kept.append(article)
        if len(kept) >= limit:
            break

    if not kept:
        gap = _gap(start_date, curr_date, f"{locale.label} [{name}]", "market news")
        return gap or f"No {locale.label} results between {start_date} and {curr_date}"

    kept.sort(key=lambda a: (a["pub_date"] is not None, a["pub_date"]), reverse=True)
    return _render(
        f"## Global Market News — {locale.label} [{name}], "
        f"from {start_date} to {curr_date}:",
        kept[:limit],
    )
