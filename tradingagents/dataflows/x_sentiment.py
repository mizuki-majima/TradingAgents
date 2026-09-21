"""Retail sentiment from X, via xAI's server-side ``x_search`` tool.

StockTwits and Reddit carry no Japanese listings — a ``.T`` request to the
StockTwits stream returns 404, and the English subreddits do not discuss Tokyo
codes — so a Japanese run reaches the Sentiment Analyst with two of its three
sources reporting "unavailable". X is where Japanese retail actually posts, and
xAI exposes it as a first-party search tool rather than as scraping.

Two properties matter here beyond coverage:

* ``from_date`` / ``to_date`` pin the search to the analysis window. The other
  social feeds serve only their latest items, so a historical run gets a
  coverage placeholder instead of data; this one can answer for a past window,
  which is what lets a backtest carry a sentiment read at all.
* The result is a search *agent's* digest, not a raw message stream. That is a
  real difference in kind from the StockTwits and Reddit blocks, so the block
  says so in its own header and carries the source URLs the model cited, and
  the Sentiment Analyst is told to weigh it as a digest.

Needs ``XAI_API_KEY``. X Search is billed per post fetched, so the post budget
is capped and configurable (``x_sentiment_max_posts``).
"""

from __future__ import annotations

import json
import logging
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .config import get_config
from .errors import VendorNotConfiguredError
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.x.ai/v1/responses"

# x_search is agentic: the model issues several searches and reads the results
# before answering, so it is far slower than a plain completion. Measured at
# 113s for one week of a Tokyo listing against 3s for a no-tool call, so the
# default leaves real headroom — a timeout here costs the run its only social
# source. ``x_sentiment_timeout`` overrides it.
_DEFAULT_TIMEOUT = 300.0

# How many cited post URLs to print. They carry no titles, so the list is for
# spot-checking the digest, not for reading.
_CITATIONS_SHOWN = 8

# The exchange suffix decides which language the posts are in, and therefore how
# the instrument is named on X: Japanese retail writes the bare 4-digit code,
# often with a cashtag, and almost never the Yahoo symbol.
_MARKET_HINTS = {
    ".T": (
        "Japanese retail traders on X refer to this listing by its bare code "
        "({term}) and by the company's Japanese name, usually with a cashtag "
        "($ {term}) and hashtags such as #日本株 or #{term}. Search in Japanese. "
        "Report the posts in Japanese as written."
    ),
}
_DEFAULT_HINT = (
    "Traders on X refer to this instrument by its ticker, usually as a cashtag "
    "(${term}). Search in English."
)


def _api_key() -> str:
    key = os.getenv("XAI_API_KEY")
    if not key:
        raise VendorNotConfiguredError(
            "XAI_API_KEY is not set. X sentiment needs an xAI API key "
            "(https://console.x.ai); without it this source reports unavailable."
        )
    return key


def _market_hint(symbol: str, term: str) -> str:
    for suffix, hint in _MARKET_HINTS.items():
        if symbol.upper().endswith(suffix):
            return hint.format(term=term)
    return _DEFAULT_HINT.format(term=term)


def _prompt(symbol: str, term: str, start_date: str, end_date: str, max_posts: int) -> str:
    """What the search agent is asked for: observations, not a trade view."""
    return (
        f"Search X for what retail traders and investors posted about the listed "
        f"company with ticker {symbol} between {start_date} and {end_date}.\n\n"
        f"{_market_hint(symbol, term)}\n\n"
        f"Read up to about {max_posts} posts.\n\n"
        f"The posts themselves are what is wanted — another agent reads them and "
        f"judges the sentiment, so a summary without them is not useful. Report:\n"
        f"1. Up to 15 of the most substantive posts, and never fewer than every "
        f"post you found if you found fewer than 15. For each, give the date, the "
        f"handle, whether it reads bullish / bearish / neither, and the post's "
        f"content in its own language.\n"
        f"2. A one-line tally: how many posts you read, and how many were "
        f"bullish / bearish / neither about this company's stock.\n"
        f"3. The recurring topics, and any event the posts react to.\n\n"
        f"Rules: report only posts you actually retrieved, and quote them as "
        f"written. If you found few posts or none, say exactly that and give the "
        f"count — do not pad the answer with general knowledge about the company, "
        f"and do not substitute posts about a different company. Do not give a "
        f"trading recommendation or a price call; another agent weighs this."
    )


def _extract_text(payload: dict) -> str:
    """The model's answer text from a /v1/responses payload."""
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts: list[str] = []
    for item in payload.get("output") or []:
        for block in (item or {}).get("content") or []:
            if block.get("type") == "output_text" and block.get("text"):
                parts.append(block["text"])
    return "\n".join(parts).strip()


def _extract_citations(payload: dict) -> list[str]:
    """Source URLs the search touched, from either place the API reports them.

    These are what makes the digest checkable, so they are carried into the
    block rather than summarised away.
    """
    urls: list[str] = []
    seen: set[str] = set()

    def _add(url) -> None:
        if isinstance(url, str) and url and url not in seen:
            seen.add(url)
            urls.append(url)

    for citation in payload.get("citations") or []:
        _add(citation if isinstance(citation, str) else (citation or {}).get("url"))
    for item in payload.get("output") or []:
        for block in (item or {}).get("content") or []:
            for annotation in block.get("annotations") or []:
                _add((annotation or {}).get("url"))
    return urls


def fetch_x_posts(
    ticker: str,
    start_date: str,
    end_date: str,
    limit: int | None = None,
) -> str:
    """Retail posts about ``ticker`` on X over ``[start_date, end_date]``.

    Returns a prompt-ready block, or a ``<...unavailable...>`` placeholder —
    never raises, so the Sentiment Analyst always sees either data or an
    explicit statement that there is none. A missing key, a throttled API and
    an empty result are three different placeholders, because reporting any of
    them as silence would hand the analyst a signal nobody observed.
    """
    config = get_config()
    max_posts = int(limit or config.get("x_sentiment_max_posts") or 60)
    model = config.get("x_sentiment_model") or "grok-4.6"
    timeout = float(config.get("x_sentiment_timeout") or _DEFAULT_TIMEOUT)

    symbol = normalize_symbol(ticker)
    term = symbol.rsplit(".", 1)[0] if "." in symbol else symbol

    try:
        key = _api_key()
    except VendorNotConfiguredError as exc:
        logger.warning("X sentiment not configured: %s", exc)
        return f"<X sentiment unavailable: {exc}>"

    # Both dates are inclusive on our side; the tool's to_date is the last day
    # it searches, so the window the analyst asked for is the window searched.
    body = {
        "model": model,
        "input": [{"role": "user", "content": _prompt(symbol, term, start_date, end_date, max_posts)}],
        "tools": [{"type": "x_search", "from_date": start_date, "to_date": end_date}],
    }
    request = Request(
        _ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except HTTPError as exc:
        # The key is in a header, not the URL, so the status is safe to report.
        logger.warning("X search failed for %s: HTTP %s", symbol, exc.code)
        return f"<X sentiment unavailable: xAI returned HTTP {exc.code}>"
    except TimeoutError as exc:
        # Distinct from a failure: the search was running and we stopped waiting,
        # which is a knob to turn rather than a fact about the instrument.
        logger.warning("X search timed out for %s after %.0fs: %s", symbol, timeout, exc)
        return (
            f"<X sentiment unavailable: the search did not finish within "
            f"{timeout:.0f}s; raise x_sentiment_timeout or lower x_sentiment_max_posts>"
        )
    except Exception as exc:  # noqa: BLE001 — a social source must not end a run
        logger.warning("X search failed for %s: %s", symbol, exc)
        return f"<X sentiment unavailable: {type(exc).__name__}>"

    text = _extract_text(payload)
    if not text:
        return f"<X sentiment unavailable: xAI returned no content for {symbol}>"

    citations = _extract_citations(payload)
    block = (
        f"[Digest produced by xAI's X search agent over {start_date}..{end_date}, "
        f"not a raw message stream. Up to {max_posts} posts were read.]\n\n{text}"
    )
    if citations:
        # These are bare status URLs with no titles, so a long list crowds out
        # the posts without adding a readable signal. A handful is enough to
        # spot-check the digest against; the count carries the rest.
        shown = citations[:_CITATIONS_SHOWN]
        listed = "\n".join(f"- {url}" for url in shown)
        more = len(citations) - len(shown)
        block += (
            f"\n\nSources the search cited ({len(citations)} total"
            + (f", {more} not listed" if more else "")
            + f"):\n{listed}"
        )
    return block
