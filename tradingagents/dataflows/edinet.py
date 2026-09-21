"""Japanese company statements as they were filed, from EDINET (金融庁).

This is the Tokyo counterpart to ``sec_edgar``, and it exists for the same
reason. Every other fundamentals vendor available for a ``.T`` listing serves a
period's current value and dates the statement by the fiscal period it covers.
Japanese filers publish a 有価証券報告書 about three months after the fiscal
year ends, so a backtest dated inside that gap is handed figures nobody could
have read yet — which is exactly the leak that makes a backtest's results look
better than the strategy was.

EDINET's 書類一覧API is indexed by *file date*, so a run dated ``curr_date``
reads only filings submitted on or before that date, and a figure restated by a
訂正有価証券報告書 still reads as first reported until the correction's own
filing date. That is the point-in-time guarantee; it is bounded by one day,
since a document filed after the close on ``curr_date`` was not public during
that session.

Two consequences of how EDINET is shaped are worth knowing:

* **There is no company filter.** ``documents.json`` takes a single date and
  returns everything filed that day, so finding one company's filings means
  walking back day by day. The per-day index is therefore cached and shared
  across every ticker and every run, which is what makes a backtest grid over
  Japanese names affordable: the first run pays for the walk, the rest do not.
* **Quarterly reports ended in 2024.** 四半期報告書 (docTypeCode 140) was
  abolished for periods from April 2024; filers now publish a 半期報告書 (160)
  instead, and quarterly detail moved to the 決算短信, which is filed with TDnet
  and is not in this API. So ``freq="quarterly"`` serves the half-year report
  for recent periods and says so, rather than implying a quarter.

Needs a free subscription key from https://api.edinet-fsa.go.jp (``EDINET_API_KEY``).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from .config import get_config
from .errors import NoMarketDataError, VendorNotConfiguredError
from .symbol_utils import normalize_symbol
from .utils import get_scrubbed, safe_ticker_component

logger = logging.getLogger(__name__)

_DOCUMENTS_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
_DOCUMENT_URL = "https://api.edinet-fsa.go.jp/api/v2/documents/{doc_id}"
_TIMEOUT = 60

# 書類種別コード, from the API spec's 参考資料. The correction codes sit beside
# their originals: a 訂正 filing carries the restated figures, and reading it
# only from its own file date onward is what keeps the vintage honest.
_ANNUAL = ("120", "130")     # 有価証券報告書 / 訂正有価証券報告書
_SEMI = ("160", "170")       # 半期報告書 / 訂正半期報告書
_QUARTERLY = ("140", "150")  # 四半期報告書 / 訂正四半期報告書 (periods before 2024-04)

_KINDS = {
    "annual": _ANNUAL,
    # A quarterly request takes whichever interim report the period had: the
    # half-year report that replaced it, or the quarterly report for older dates.
    "quarterly": _SEMI + _QUARTERLY,
}

# A filing's own index entry is immutable once its file date has passed, so the
# reduced day index can be cached indefinitely. Today's is not cached at all.
_INDEX_DIR = "edinet"

# Line items, each with the XBRL element names and the Japanese labels that
# carry it. Element names are matched on their local part, so the taxonomy
# prefix (``jppfs_cor:`` for Japanese GAAP, ``jpigp_cor:`` for IFRS) does not
# have to be enumerated. The labels are matched too, because the same line is
# named differently across the two taxonomies and across taxonomy years, and a
# label match is what keeps an IFRS filer from reading as an empty statement.
# First match wins and values are never summed across names.
_STATEMENTS: dict[str, list[tuple[str, tuple[str, ...], tuple[str, ...]]]] = {
    "balance_sheet": [
        ("Total Assets", ("Assets", "AssetsIFRS"), ("資産合計",)),
        ("Current Assets", ("CurrentAssets", "CurrentAssetsIFRS"), ("流動資産合計",)),
        ("Cash and Deposits",
         ("CashAndDeposits", "CashAndCashEquivalentsIFRS", "CashAndDepositsIFRS"),
         ("現金及び預金", "現金及び現金同等物")),
        ("Total Liabilities", ("Liabilities", "LiabilitiesIFRS"), ("負債合計",)),
        ("Current Liabilities",
         ("CurrentLiabilities", "CurrentLiabilitiesIFRS"), ("流動負債合計",)),
        ("Net Assets / Equity",
         ("NetAssets", "EquityIFRS", "EquityAttributableToOwnersOfParentIFRS"),
         ("純資産合計", "資本合計")),
    ],
    "income_statement": [
        ("Revenue",
         ("NetSales", "RevenueIFRS", "NetSalesIFRS", "SalesRevenuesIFRS",
          "RevenueFromContractsWithCustomersIFRS"),
         ("売上高", "売上収益", "営業収益")),
        ("Cost of Sales", ("CostOfSales", "CostOfSalesIFRS"), ("売上原価",)),
        ("Gross Profit", ("GrossProfit", "GrossProfitIFRS"), ("売上総利益",)),
        ("Operating Income",
         ("OperatingIncome", "OperatingProfitLossIFRS", "OperatingIncomeIFRS"),
         ("営業利益", "営業利益（損失）")),
        ("Ordinary Income", ("OrdinaryIncome",), ("経常利益", "経常利益（損失）")),
        ("Net Income",
         ("ProfitLossAttributableToOwnersOfParent",
          "ProfitLossAttributableToOwnersOfParentIFRS"),
         ("親会社株主に帰属する当期純利益", "親会社の所有者に帰属する当期利益")),
    ],
    "cashflow": [
        ("Operating Cash Flow",
         ("NetCashProvidedByUsedInOperatingActivities",
          "NetCashProvidedByUsedInOperatingActivitiesIFRS"),
         ("営業活動によるキャッシュ・フロー",)),
        ("Investing Cash Flow",
         ("NetCashProvidedByUsedInInvestmentActivities",
          "NetCashProvidedByUsedInInvestingActivitiesIFRS"),
         ("投資活動によるキャッシュ・フロー",)),
        ("Financing Cash Flow",
         ("NetCashProvidedByUsedInFinancingActivities",
          "NetCashProvidedByUsedInFinancingActivitiesIFRS"),
         ("財務活動によるキャッシュ・フロー",)),
    ],
}


def get_api_key() -> str:
    key = os.getenv("EDINET_API_KEY")
    if not key:
        raise VendorNotConfiguredError(
            "EDINET_API_KEY is not set. Get a free subscription key at "
            "https://api.edinet-fsa.go.jp (アカウント作成 → APIキーの発行)."
        )
    return key


def tokyo_code(ticker: str) -> str | None:
    """The 4-character TSE code for a Tokyo listing, or None for anything else.

    EDINET reports a filer's 提出者証券コード as five characters — the listing's
    code with a trailing ``0`` — so the comparison happens on the 4-character
    form both sides can produce.
    """
    symbol = normalize_symbol(ticker).upper()
    if not symbol.endswith(".T"):
        return None
    code = symbol[:-2]
    return code if len(code) == 4 and code[:3].isdigit() else None


def _cache_dir() -> Path:
    path = Path(get_config()["data_cache_dir"]) / _INDEX_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch_day(day: str) -> list[dict]:
    """Every listed-company filing on one file date, reduced to what we use."""
    key = get_api_key()
    response = get_scrubbed(
        _DOCUMENTS_URL,
        params={"date": day, "type": 2, "Subscription-Key": key},
        timeout=_TIMEOUT,
        secret=key,
        passthrough=(404,),
    )
    if response.status_code == 404:
        # Outside the 10-year retention window, or a date EDINET has no file for.
        return []
    payload = response.json()

    rows = []
    for entry in payload.get("results") or []:
        sec_code = entry.get("secCode")
        # Rows with no securities code are funds and unlisted filers; rows whose
        # CSV flag is off (withdrawn, non-disclosed, expired) cannot be fetched
        # at all, so indexing them would only produce a later dead end.
        if not sec_code or entry.get("csvFlag") != "1":
            continue
        if entry.get("withdrawalStatus") != "0" or entry.get("disclosureStatus") != "0":
            continue
        rows.append({
            "code": str(sec_code)[:4],
            "docID": entry.get("docID"),
            "docTypeCode": entry.get("docTypeCode"),
            "periodStart": entry.get("periodStart"),
            "periodEnd": entry.get("periodEnd"),
            "submitDateTime": entry.get("submitDateTime"),
            "filerName": entry.get("filerName"),
            "docDescription": entry.get("docDescription"),
        })
    return rows


def _day_index(day: str) -> list[dict]:
    """``_fetch_day`` with an on-disk cache.

    A past file date's list cannot change in a way that matters here — a later
    correction is its own filing on its own date — so it is cached without a
    TTL. Today's is never cached, since filings are still arriving.
    """
    if day >= date.today().isoformat():
        return _fetch_day(day)

    path = _cache_dir() / f"{safe_ticker_component(day)}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Discarding unreadable EDINET index cache %s", path)

    rows = _fetch_day(day)
    try:
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not cache EDINET index for %s: %s", day, exc)
    return rows


def find_filings(code: str, curr_date: str, doc_types, need: int = 2) -> list[dict]:
    """Filings of ``doc_types`` for ``code``, newest first, filed by ``curr_date``.

    Walks back one file date at a time because the API offers no company filter.
    Weekends are skipped (nothing is filed then) and the walk stops as soon as
    ``need`` filings are found, so the common case — the latest annual report,
    filed within the last few months — costs far less than the bound.
    """
    scan_days = int(get_config().get("edinet_scan_days") or 450)
    day = datetime.strptime(curr_date, "%Y-%m-%d").date()
    found: list[dict] = []
    requests_made = 0

    for _ in range(scan_days):
        if day.weekday() < 5:  # Mon-Fri; EDINET accepts no filings at the weekend
            requests_made += 1
            if requests_made == 20:
                logger.info(
                    "EDINET index is cold for %s; walking back up to %d days. "
                    "The per-day index is shared, so later runs reuse it.",
                    code, scan_days,
                )
            for row in _day_index(day.isoformat()):
                if row["code"] == code and row["docTypeCode"] in doc_types:
                    found.append(row)
            if len(found) >= need:
                break
        day -= timedelta(days=1)

    return found[:need]


def _document_csv(doc_id: str) -> list[dict]:
    """The filing's XBRL-to-CSV rows.

    EDINET serves a ZIP whose ``XBRL_TO_CSV`` folder holds the report itself
    (``jpcrp*``) beside the audit report (``jpaud*``); only the former carries
    the statements. The CSV is UTF-16 and tab-separated, and its values contain
    raw tabs and newlines inside quotes, so it is read with a real CSV reader
    rather than split on the delimiter.
    """
    key = get_api_key()
    response = get_scrubbed(
        _DOCUMENT_URL.format(doc_id=doc_id),
        params={"type": 5, "Subscription-Key": key},
        timeout=_TIMEOUT,
        secret=key,
    )

    try:
        archive = zipfile.ZipFile(io.BytesIO(response.content))
    except zipfile.BadZipFile as exc:
        raise NoMarketDataError(doc_id, doc_id, f"EDINET returned no CSV archive: {exc}") from exc

    names = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and "jpaud" not in name.lower()
    ]
    if not names:
        raise NoMarketDataError(doc_id, doc_id, "filing has no statement CSV")

    rows: list[dict] = []
    for name in names:
        raw = archive.read(name)
        text = raw.decode("utf-16", errors="replace").lstrip("﻿")
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        rows.extend(reader)
    return rows


def _element_local(element_id: str) -> str:
    """``jppfs_cor:Assets`` -> ``Assets``; the prefix varies by taxonomy."""
    return element_id.rsplit(":", 1)[-1].strip()


def _context_rank(context: str) -> int | None:
    """How much a context is worth, or None to reject it.

    An undimensioned current-period context is the consolidated figure for the
    period the filing reports; a context carrying a member is a segment, a prior
    period, or the parent-only column. Parent-only is kept as a last resort,
    since a filer with no subsidiaries reports nothing else.
    """
    if not context.startswith("Current"):
        return None
    if "_" not in context:
        return 2
    if context.endswith("_NonConsolidatedMember"):
        return 1
    return None


def _pick(rows: list[dict], names: tuple[str, ...], labels: tuple[str, ...]) -> tuple[str, str] | None:
    """The best (value, unit) for one line item, or None when it is not reported."""
    best_rank = 0
    best: tuple[str, str] | None = None
    for row in rows:
        element = _element_local(row.get("要素ID") or "")
        label = (row.get("項目名") or "").strip()
        if element not in names and label not in labels:
            continue
        rank = _context_rank((row.get("コンテキストID") or "").strip())
        if rank is None or rank <= best_rank:
            continue
        value = (row.get("値") or "").strip()
        if not value or value == "－":
            continue
        best_rank, best = rank, (value, (row.get("単位") or "").strip())
    return best


def _format_value(value: str, unit: str) -> str:
    """Yen figures in millions, like the other statement vendors report them."""
    try:
        number = float(value)
    except ValueError:
        return value
    if unit in ("JPY", "円", ""):
        return f"{number / 1_000_000:,.1f}"
    return f"{number:,.4g}"


def _statement(kind: str, ticker: str, freq: str, curr_date: str, title: str) -> str:
    code = tokyo_code(ticker)
    if code is None:
        # Not a Tokyo listing: fall through to the next vendor in the chain.
        raise NoMarketDataError(ticker, ticker, "not a Tokyo (.T) listing")

    curr_date = curr_date or date.today().isoformat()
    doc_types = _KINDS["annual" if freq.lower() == "annual" else "quarterly"]
    filings = find_filings(code, curr_date, doc_types)
    if not filings:
        raise NoMarketDataError(
            ticker, ticker,
            f"no {freq} report filed with EDINET by {curr_date} for code {code}",
        )

    columns: list[tuple[dict, dict[str, tuple[str, str]]]] = []
    for filing in filings:
        try:
            rows = _document_csv(filing["docID"])
        except (NoMarketDataError, requests.RequestException) as exc:
            logger.warning("EDINET document %s unreadable: %s", filing["docID"], exc)
            continue
        values = {}
        for label, names, labels in _STATEMENTS[kind]:
            picked = _pick(rows, names, labels)
            if picked:
                values[label] = picked
        if values:
            columns.append((filing, values))
        time.sleep(0.2)  # EDINET publishes no rate limit; do not hammer it either

    if not columns:
        raise NoMarketDataError(
            ticker, ticker,
            f"EDINET filings found for {code} but none reported a usable {title.lower()}",
        )

    filer = columns[0][0].get("filerName") or code
    lines = [
        f"# {title} for {code} ({filer}) — EDINET, as filed",
        f"# Point-in-time as of: {curr_date}. Only documents submitted on or before "
        f"this date are read; a figure restated later still reads as first reported.",
        "# Yen amounts are in millions.",
        "",
    ]
    header = "| Line item | " + " | ".join(
        f"{f.get('periodEnd') or f.get('periodStart') or '?'}"
        f" (filed {(f.get('submitDateTime') or '?')[:10]})"
        for f, _ in columns
    ) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(columns))
    for label, _, _ in _STATEMENTS[kind]:
        cells = []
        for _, values in columns:
            picked = values.get(label)
            cells.append(_format_value(*picked) if picked else "n/a")
        if any(cell != "n/a" for cell in cells):
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    for filing, _ in columns:
        lines.append(
            f"- {filing.get('docDescription') or filing.get('docTypeCode')}"
            f" (docID {filing['docID']}, filed {filing.get('submitDateTime')})"
        )
    if freq.lower() != "annual" and any(
        f.get("docTypeCode") in _SEMI for f, _ in columns
    ):
        lines.append("")
        lines.append(
            "Note: 四半期報告書 was abolished for periods from April 2024, so this "
            "interim statement is a 半期報告書 covering half a year, not a quarter. "
            "Quarterly detail is published as a 決算短信 through TDnet, which this "
            "API does not serve."
        )
    return "\n".join(lines)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Balance sheet for a Tokyo listing, as filed with EDINET by ``curr_date``."""
    return _statement("balance_sheet", ticker, freq, curr_date, "Balance Sheet")


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Income statement for a Tokyo listing, as filed with EDINET by ``curr_date``."""
    return _statement("income_statement", ticker, freq, curr_date, "Income Statement")


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    """Cash flow statement for a Tokyo listing, as filed with EDINET by ``curr_date``."""
    return _statement("cashflow", ticker, freq, curr_date, "Cash Flow Statement")
