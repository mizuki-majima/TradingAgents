"""Japanese statements with their filing dates, from EDINET DB.

The ``edinet`` vendor reads 金融庁's own API, which is authoritative and free but
is indexed by file date with no company filter, so one company's filings are
found by walking dates backwards and parsing XBRL-to-CSV. EDINET DB is a
third-party service that has already done that work: one request returns a
company's annual history with the figures normalised across JP-GAAP, IFRS and US
GAAP, and — the part that matters here — each year carries the ``submit_date`` of
the filing it came from.

That date is what makes a Japanese backtest honest. A filer publishes its
有価証券報告書 roughly three months after the fiscal year ends, so a run dated
inside that gap must not see the year it has not read yet. Rows are filtered on
``submit_date <= curr_date``, and the rendered table shows the date beside each
column so the agents can see the vintage they are reasoning about.

One limit to be precise about: the service serves the *current* value of each
year with ``is_restated_*`` flags, not the vintage that was current on the
analysis date. So a figure restated later reads as restated, which SEC EDGAR
would not do. The large leak — reading a year before it was published — is
closed; the smaller one, restatement vintage, is not.

Annual only. A quarterly request raises so the chain falls through to a vendor
that has interim figures.

Needs ``EDINETDB_API_KEY`` (https://edinetdb.jp). A key in ``EDINET_API_KEY``
that carries this service's ``edb_`` prefix is accepted too, since that is where
it most often lands.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path

from .config import get_config
from .errors import NoMarketDataError, VendorNotConfiguredError, VendorRateLimitError
from .symbol_utils import normalize_symbol
from .utils import get_scrubbed, safe_ticker_component

logger = logging.getLogger(__name__)

_BASE = "https://edinetdb.jp/v1"
_TIMEOUT = 30
_CACHE_DIR = "edinetdb"

# This service's keys carry a distinguishing prefix, which is what lets a key
# sitting in the official vendor's variable be recognised rather than sent to an
# API that will reject it.
KEY_PREFIX = "edb_"

# Line items, as the service names them. Values are already normalised across
# accounting standards, so unlike the XBRL path there is no per-taxonomy
# alternative list; a field a filer does not report simply comes back null.
_STATEMENTS: dict[str, list[tuple[str, str]]] = {
    "balance_sheet": [
        ("Total Assets", "total_assets"),
        ("Current Assets", "current_assets"),
        ("Cash and Equivalents", "cash"),
        ("Inventories", "inventories"),
        ("Property, Plant and Equipment", "ppe"),
        ("Total Liabilities", "total_liabilities"),
        ("Current Liabilities", "current_liabilities"),
        ("Interest-Bearing Debt (current)", "ibd_current"),
        ("Interest-Bearing Debt (non-current)", "ibd_noncurrent"),
        ("Net Assets", "net_assets"),
        ("Shareholders Equity", "shareholders_equity"),
    ],
    "income_statement": [
        ("Revenue", "revenue"),
        ("Cost of Sales", "cost_of_sales"),
        ("Gross Profit", "gross_profit"),
        ("Operating Income", "operating_income"),
        ("Ordinary Income", "ordinary_income"),
        ("Profit Before Tax", "profit_before_tax"),
        ("Income Taxes", "income_taxes"),
        ("Net Income", "net_income"),
        ("EPS", "eps"),
        ("Diluted EPS", "diluted_eps"),
    ],
    "cashflow": [
        ("Operating Cash Flow", "cf_operating"),
        ("Investing Cash Flow", "cf_investing"),
        ("Financing Cash Flow", "cf_financing"),
        ("Capital Expenditure", "capex"),
        ("Depreciation", "depreciation"),
        ("Cash Dividends Paid", "cash_dividends_paid"),
    ],
}

# Reported per share rather than in yen, so they must not be scaled to millions.
_PER_SHARE = {"eps", "diluted_eps", "bps", "dividend_per_share"}


def get_api_key() -> str:
    key = os.getenv("EDINETDB_API_KEY")
    if not key:
        fallback = os.getenv("EDINET_API_KEY") or ""
        if fallback.strip().startswith(KEY_PREFIX):
            key = fallback
    if not key:
        raise VendorNotConfiguredError(
            "EDINETDB_API_KEY is not set. Keys are issued at https://edinetdb.jp "
            f"and begin with '{KEY_PREFIX}'."
        )
    return key.strip()


def _cache_dir() -> Path:
    path = Path(get_config()["data_cache_dir"]) / _CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get(path: str, params: dict | None = None) -> dict:
    key = get_api_key()
    response = get_scrubbed(
        f"{_BASE}{path}",
        params=params or {},
        timeout=_TIMEOUT,
        secret=key,
        headers={"X-API-Key": key},
        passthrough=(404, 429),
    )
    if response.status_code == 404:
        raise NoMarketDataError(path, path, "not covered by EDINET DB")
    if response.status_code == 429:
        # The free tier is a daily request budget; say so rather than letting it
        # read as an absence of filings.
        raise VendorRateLimitError("EDINET DB daily request quota exhausted")
    return response.json()


def _cached(name: str, path: str, params: dict | None = None) -> dict:
    """``_get`` with a per-day disk cache.

    The free tier is a daily request budget, and a filing history changes only
    when something new is filed, so a run re-reading the same company must not
    spend another request on it.
    """
    stamp = date.today().isoformat()
    file = _cache_dir() / f"{safe_ticker_component(name)}-{stamp}.json"
    if file.exists():
        try:
            return json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Discarding unreadable EDINET DB cache %s", file)

    payload = _get(path, params)
    try:
        file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not cache EDINET DB response for %s: %s", name, exc)
    return payload


def company_for(ticker: str) -> dict:
    """The filer record for a Tokyo listing, or raise so the chain falls through."""
    symbol = normalize_symbol(ticker).upper()
    if not symbol.endswith(".T"):
        raise NoMarketDataError(ticker, symbol, "not a Tokyo (.T) listing")
    code = symbol[:-2]

    payload = _cached(f"company-{code}", "/companies", {"sec_code": code})
    rows = payload.get("data") or []
    if not rows:
        raise NoMarketDataError(ticker, symbol, f"no EDINET filer for code {code}")
    return rows[0]


def _as_of(rows: list[dict], curr_date: str) -> list[dict]:
    """Years whose filing was submitted on or before ``curr_date``, newest first.

    A row with no ``submit_date`` cannot be shown to have been public, so it is
    dropped rather than assumed readable — the whole point of this vendor is
    that a backtest never sees a year before it was filed.
    """
    kept = []
    for row in rows:
        submitted = (row.get("submit_date") or "")[:10]
        if submitted and submitted <= curr_date:
            kept.append(row)
    kept.sort(key=lambda r: (r.get("submit_date") or "", r.get("fiscal_year") or 0), reverse=True)
    return kept


def _format(field: str, value) -> str:
    if value is None:
        return "n/a"
    if not isinstance(value, (int, float)):
        return str(value)
    if field in _PER_SHARE:
        return f"{value:,.2f}"
    return f"{value / 1_000_000:,.1f}"


def _statement(kind: str, ticker: str, freq: str, curr_date: str | None, title: str) -> str:
    if freq.lower() != "annual":
        # Only annual figures are served; an interim request belongs to a vendor
        # that has one, so this must not answer with a year and call it a quarter.
        raise NoMarketDataError(
            ticker, ticker,
            "EDINET DB serves annual figures only; interim periods come from another vendor",
        )

    company = company_for(ticker)
    curr_date = (curr_date or date.today().isoformat())[:10]
    edinet_code = company.get("edinet_code")

    payload = _cached(f"financials-{edinet_code}", f"/companies/{edinet_code}/financials")
    rows = _as_of(payload.get("data") or [], curr_date)
    if not rows:
        raise NoMarketDataError(
            ticker, ticker,
            f"no annual report filed by {curr_date} for {edinet_code}",
        )
    columns = rows[: int(get_config().get("edinetdb_periods") or 3)]

    name = company.get("name_ja") or company.get("name") or edinet_code
    lines = [
        f"# {title} for {ticker.upper()} ({name}) — EDINET DB, filing-dated",
        f"# Point-in-time as of: {curr_date}. Only fiscal years whose filing was "
        f"submitted on or before this date are shown.",
        "# Accounting standard: "
        + (columns[0].get("accounting_standard") or "unknown")
        + f"; basis: {columns[0].get('basis') or 'unknown'}.",
        "# Yen amounts are in millions; per-share figures are in yen.",
        "# Each year carries the value as currently reported, with restatement "
        "flags — not the vintage that was current on the analysis date.",
        "",
    ]
    header = "| Line item | " + " | ".join(
        f"FY{row.get('fiscal_year')} (filed {(row.get('submit_date') or '?')[:10]})"
        for row in columns
    ) + " |"
    lines += [header, "|---|" + "---:|" * len(columns)]
    for label, field in _STATEMENTS[kind]:
        cells = [_format(field, row.get(field)) for row in columns]
        if any(cell != "n/a" for cell in cells):
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

    restated = [
        f"FY{row.get('fiscal_year')}" for row in columns
        if any(row.get(flag) for flag in ("is_restated_eps", "is_restated_bps", "is_restated_diluted_eps"))
    ]
    if restated:
        lines += ["", f"Restated since first reported: {', '.join(restated)}."]
    source = columns[0].get("edinet_filing_url")
    if source:
        lines += ["", f"Newest filing: {source}"]
    return "\n".join(lines)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Balance sheet for a Tokyo listing, filtered to what was filed by ``curr_date``."""
    return _statement("balance_sheet", ticker, freq, curr_date, "Balance Sheet")


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Income statement for a Tokyo listing, filtered to what was filed by ``curr_date``."""
    return _statement("income_statement", ticker, freq, curr_date, "Income Statement")


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Cash flow statement for a Tokyo listing, filtered to what was filed by ``curr_date``."""
    return _statement("cashflow", ticker, freq, curr_date, "Cash Flow Statement")
