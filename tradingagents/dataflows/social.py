"""Which social feeds the Sentiment Analyst reads, and how to read each one.

The analyst used to fetch StockTwits and Reddit by name. Neither covers a Tokyo
listing — the StockTwits stream 404s on a ``.T`` symbol and the English
subreddits do not discuss Japanese codes — so a Japanese run reached the analyst
with both sources reporting "unavailable" and a prompt still telling it how to
read a Bullish/Bearish ratio that was never there.

Sources are resolved per run instead, and each one carries its own reading
guidance into the prompt, so the analyst is only ever told how to read feeds it
actually has. ``social_vendors`` picks them; ``"auto"`` follows the exchange.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from .config import get_config
from .reddit import fetch_reddit_posts
from .stocktwits import fetch_stocktwits_messages
from .symbol_utils import normalize_symbol
from .x_sentiment import fetch_x_posts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SocialSource:
    """One feed: what it is, how to read it, and the text it returned."""

    name: str
    guidance: str
    text: str


@dataclass(frozen=True)
class _Vendor:
    label: str
    guidance: str
    fetch: Callable[[str, str, str], str]


def _stocktwits(ticker: str, start_date: str, end_date: str) -> str:
    return fetch_stocktwits_messages(
        ticker, limit=30, start_date=start_date, end_date=end_date
    )


def _reddit(ticker: str, start_date: str, end_date: str) -> str:
    return fetch_reddit_posts(ticker, start_date=start_date, end_date=end_date)


_VENDORS: dict[str, _Vendor] = {
    "stocktwits": _Vendor(
        label="StockTwits messages — retail-trader platform indexed by cashtag",
        guidance=(
            "Fast-moving retail signal. Each message carries a user-labeled "
            "sentiment tag (Bullish / Bearish / no-label) plus the body. Read "
            "the Bullish/Bearish ratio as a leading retail-sentiment signal: a "
            "70/30 split is moderately bullish; >=90/10 may indicate "
            "over-extension and contrarian risk; 50/50 is uncertainty. Sample "
            "size matters — base rates on the actual message count, not "
            "percentages alone."
        ),
        fetch=_stocktwits,
    ),
    "reddit": _Vendor(
        label="Reddit posts — r/wallstreetbets, r/stocks, r/investing",
        guidance=(
            "Community discussion, without vote or comment counts. Subreddit "
            "character matters (r/wallstreetbets is often contrarian and "
            "exuberant; r/stocks more measured; r/investing longer-term). Judge "
            "a post by its body excerpt, not its title alone, and do not infer "
            "engagement."
        ),
        fetch=_reddit,
    ),
    "x": _Vendor(
        label="X posts — retrieved and summarised by xAI's X search agent",
        guidance=(
            "This block is a search agent's digest of the window, not a raw "
            "message stream, so it is weaker evidence than a feed you can count "
            "yourself: lean on the quoted posts and the cited source links "
            "rather than on the digest's own characterisation, and treat its "
            "bullish/bearish tally as the agent's reading rather than as "
            "user-applied labels. If it reports few posts or none, that is the "
            "finding — say so in `confidence` rather than filling the gap."
        ),
        fetch=fetch_x_posts,
    ),
}

# Exchange suffix -> the feeds that actually cover it. Anything unlisted keeps
# the US-market default.
_AUTO_BY_SUFFIX = {".T": ("x",)}
_AUTO_DEFAULT = ("stocktwits", "reddit")


def resolve_vendor_names(ticker: str) -> tuple[str, ...]:
    """The configured feed names for ``ticker``.

    ``social_vendors`` is either an explicit comma-separated chain or ``"auto"``,
    which picks by exchange. Unknown names are dropped with a warning rather
    than failing the run, so a typo costs one source, not the analysis.
    """
    configured = get_config().get("social_vendors") or "auto"
    if isinstance(configured, str):
        requested = tuple(v.strip().lower() for v in configured.split(",") if v.strip())
    else:
        requested = tuple(str(v).strip().lower() for v in configured)

    if requested in ((), ("auto",)):
        symbol = normalize_symbol(ticker).upper()
        for suffix, names in _AUTO_BY_SUFFIX.items():
            if symbol.endswith(suffix):
                return names
        return _AUTO_DEFAULT

    known = tuple(name for name in requested if name in _VENDORS)
    for name in requested:
        if name not in _VENDORS:
            logger.warning(
                "Unknown social vendor %r; available: %s", name, sorted(_VENDORS)
            )
    return known


def resolve_social_sources(
    ticker: str, start_date: str, end_date: str
) -> list[SocialSource]:
    """Fetch every configured feed for ``ticker`` over the window.

    Every fetcher degrades to a placeholder string rather than raising, and a
    fetcher that raises anyway is reported as unavailable — an exception here
    must not end a run over one of several sentiment inputs.
    """
    sources: list[SocialSource] = []
    for name in resolve_vendor_names(ticker):
        vendor = _VENDORS[name]
        try:
            text = vendor.fetch(ticker, start_date, end_date)
        except Exception as exc:  # noqa: BLE001 — one feed must not sink the analyst
            logger.warning("Social vendor %r failed for %s: %s", name, ticker, exc)
            text = f"<{name} unavailable: {type(exc).__name__}>"
        sources.append(SocialSource(vendor.label, vendor.guidance, text))
    return sources
