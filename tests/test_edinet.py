"""EDINET serves Japanese statements as filed, which is what a JP backtest needs.

The leak this vendor closes: every other fundamentals source for a ``.T`` listing
dates a statement by the period it covers, and a Japanese filer publishes its
有価証券報告書 about three months after the year ends. A backtest dated inside
that gap reads figures that were not public. EDINET is indexed by file date, so
these cover that the walk never looks past the analysis date, that the parser
survives the format's quirks (UTF-16, tabs inside quoted values, two taxonomies),
and that a non-Tokyo ticker falls through to the next vendor.
"""

from __future__ import annotations

import io
import json
import zipfile
from unittest import mock

import pytest

from tradingagents.dataflows import edinet, interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorNotConfiguredError

_HEADER = ["要素ID", "項目名", "コンテキストID", "相対年度", "連結・個別", "期間・時点", "ユニットID", "単位", "値"]


def _csv_zip(rows, name="XBRL_TO_CSV/jpcrp030000-asr-001_E02144-000.csv", extra=None):
    """A ZIP shaped like a type=5 download: UTF-16, tab-separated, quoted values."""
    def line(cells):
        out = []
        for cell in cells:
            if any(ch in cell for ch in "\t\n\""):
                out.append('"' + cell.replace('"', '""') + '"')
            else:
                out.append(cell)
        return "\t".join(out)

    body = "\r\n".join([line(_HEADER)] + [line(r) for r in rows]) + "\r\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, body.encode("utf-16"))
        # The audit report ships in the same archive and carries none of the
        # statements; reading it would only add noise.
        archive.writestr("XBRL_TO_CSV/jpaud000000-aai-001_E02144-000.csv",
                         (line(_HEADER) + "\r\n").encode("utf-16"))
        for path, content in (extra or {}).items():
            archive.writestr(path, content.encode("utf-16"))
    return buffer.getvalue()


def _row(element, label, context, value, unit="JPY", consolidated="連結", relative="当期"):
    period = "時点" if "Instant" in context else "期間"
    return [element, label, context, relative, consolidated, period, "JPY", unit, value]


def _parsed(element, label, context, value, unit="JPY"):
    """A row as ``_document_csv`` hands it on: the CSV header zipped to values."""
    return dict(zip(_HEADER, _row(element, label, context, value, unit), strict=True))


def _response(content=b"", status=200, payload=None):
    response = mock.Mock()
    response.status_code = status
    response.content = content
    response.json.return_value = payload or {}
    return response


@pytest.fixture(autouse=True)
def _api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("EDINET_API_KEY", "test-key")
    set_config({"data_cache_dir": str(tmp_path)})


@pytest.mark.unit
class TestTokyoCode:
    def test_a_tokyo_listing_yields_its_four_character_code(self):
        assert edinet.tokyo_code("7203.T") == "7203"
        assert edinet.tokyo_code("7203.JP") == "7203"  # normalized first

    def test_a_jpx_alphanumeric_code_is_accepted(self):
        assert edinet.tokyo_code("130A.T") == "130A"

    def test_anything_else_is_not_a_tokyo_listing(self):
        for ticker in ("AAPL", "0700.HK", "600519.SS", "BTC-USD"):
            assert edinet.tokyo_code(ticker) is None


@pytest.mark.unit
class TestDayIndex:
    def _index(self, monkeypatch, results):
        monkeypatch.setattr(
            edinet, "get_scrubbed",
            lambda *a, **k: _response(payload={"results": results}),
        )

    def test_only_listed_filings_with_a_fetchable_csv_are_indexed(self, monkeypatch):
        self._index(monkeypatch, [
            {"secCode": "72030", "docID": "S1", "docTypeCode": "120", "csvFlag": "1",
             "withdrawalStatus": "0", "disclosureStatus": "0"},
            # A fund: no securities code, so it can never match a ticker.
            {"secCode": None, "docID": "S2", "docTypeCode": "120", "csvFlag": "1",
             "withdrawalStatus": "0", "disclosureStatus": "0"},
            # No CSV available — indexing it would only produce a dead end later.
            {"secCode": "67580", "docID": "S3", "docTypeCode": "120", "csvFlag": "0",
             "withdrawalStatus": "0", "disclosureStatus": "0"},
            # Withdrawn and non-disclosed filings cannot be fetched at all.
            {"secCode": "94320", "docID": "S4", "docTypeCode": "120", "csvFlag": "1",
             "withdrawalStatus": "2", "disclosureStatus": "0"},
            {"secCode": "40630", "docID": "S5", "docTypeCode": "120", "csvFlag": "1",
             "withdrawalStatus": "0", "disclosureStatus": "1"},
        ])
        rows = edinet._day_index("2026-06-25")
        assert [r["docID"] for r in rows] == ["S1"]
        # The 5-character 提出者証券コード is reduced to the listing's own code.
        assert rows[0]["code"] == "7203"

    def test_a_past_day_is_cached_and_not_refetched(self, monkeypatch, tmp_path):
        calls = []

        def _fetch(*a, **k):
            calls.append(1)
            return _response(payload={"results": [
                {"secCode": "72030", "docID": "S1", "docTypeCode": "120", "csvFlag": "1",
                 "withdrawalStatus": "0", "disclosureStatus": "0"},
            ]})

        monkeypatch.setattr(edinet, "get_scrubbed", _fetch)
        assert edinet._day_index("2020-06-25")
        assert edinet._day_index("2020-06-25")
        assert len(calls) == 1
        assert json.loads((tmp_path / "edinet" / "2020-06-25.json").read_text("utf-8"))

    def test_today_is_never_cached_because_filings_are_still_arriving(self, monkeypatch):
        from datetime import date

        calls = []

        def _fetch(*a, **k):
            calls.append(1)
            return _response(payload={"results": []})

        monkeypatch.setattr(edinet, "get_scrubbed", _fetch)
        today = date.today().isoformat()
        edinet._day_index(today)
        edinet._day_index(today)
        assert len(calls) == 2

    def test_a_date_edinet_has_no_file_for_is_an_empty_day_not_an_error(self, monkeypatch):
        monkeypatch.setattr(edinet, "get_scrubbed", lambda *a, **k: _response(status=404))
        assert edinet._day_index("2004-01-05") == []


@pytest.mark.unit
class TestFindFilings:
    def test_the_walk_never_looks_past_the_analysis_date(self, monkeypatch):
        """The point-in-time guarantee: only file dates <= curr_date are requested."""
        asked = []

        def _index(day):
            asked.append(day)
            return []

        monkeypatch.setattr(edinet, "_day_index", _index)
        set_config({"edinet_scan_days": 10})
        edinet.find_filings("7203", "2026-06-25", edinet._ANNUAL)
        assert asked, "the walk must request at least one day"
        assert max(asked) <= "2026-06-25"

    def test_weekends_are_skipped(self, monkeypatch):
        asked = []
        monkeypatch.setattr(edinet, "_day_index", lambda day: asked.append(day) or [])
        set_config({"edinet_scan_days": 14})
        edinet.find_filings("7203", "2026-06-25", edinet._ANNUAL)
        # 2026-06-20/21 are a Saturday and Sunday; EDINET takes no filings then.
        assert "2026-06-20" not in asked
        assert "2026-06-21" not in asked

    def test_the_walk_stops_as_soon_as_enough_filings_are_found(self, monkeypatch):
        asked = []

        def _index(day):
            asked.append(day)
            if day == "2026-06-25":
                return [
                    {"code": "7203", "docID": "S1", "docTypeCode": "120"},
                    {"code": "7203", "docID": "S2", "docTypeCode": "120"},
                ]
            return []

        monkeypatch.setattr(edinet, "_day_index", _index)
        set_config({"edinet_scan_days": 400})
        found = edinet.find_filings("7203", "2026-06-25", edinet._ANNUAL, need=2)
        assert [f["docID"] for f in found] == ["S1", "S2"]
        assert asked == ["2026-06-25"]

    def test_other_companies_and_other_document_types_are_ignored(self, monkeypatch):
        monkeypatch.setattr(edinet, "_day_index", lambda day: [
            {"code": "6758", "docID": "OTHER", "docTypeCode": "120"},
            {"code": "7203", "docID": "LARGE_HOLDING", "docTypeCode": "350"},
            {"code": "7203", "docID": "WANTED", "docTypeCode": "120"},
        ] if day == "2026-06-25" else [])
        set_config({"edinet_scan_days": 3})
        found = edinet.find_filings("7203", "2026-06-25", edinet._ANNUAL, need=1)
        assert [f["docID"] for f in found] == ["WANTED"]


@pytest.mark.unit
class TestCsvParsing:
    def test_values_carrying_tabs_and_newlines_survive(self, monkeypatch):
        monkeypatch.setattr(edinet, "get_scrubbed", lambda *a, **k: _response(
            content=_csv_zip([
                _row("jppfs_cor:Assets", "資産合計", "CurrentYearInstant", "1000000"),
                # A note field with a raw tab and newline inside quotes: the reason
                # this file cannot be parsed by splitting on the delimiter.
                _row("jpcrp_cor:Note", "注記", "CurrentYearInstant", "a\tb\nc", unit=""),
            ])))
        rows = edinet._document_csv("S1")
        assert any(r["項目名"] == "注記" and r["値"] == "a\tb\nc" for r in rows)

    def test_the_audit_report_in_the_same_archive_is_not_read(self, monkeypatch):
        monkeypatch.setattr(edinet, "get_scrubbed", lambda *a, **k: _response(
            content=_csv_zip([_row("jppfs_cor:Assets", "資産合計", "CurrentYearInstant", "5")])))
        rows = edinet._document_csv("S1")
        assert len(rows) == 1

    def test_an_archive_without_a_statement_csv_is_no_data(self, monkeypatch):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("PublicDoc/0000000_header.pdf", b"%PDF")
        monkeypatch.setattr(edinet, "get_scrubbed",
                            lambda *a, **k: _response(content=buffer.getvalue()))
        with pytest.raises(NoMarketDataError):
            edinet._document_csv("S1")


@pytest.mark.unit
class TestContextSelection:
    def test_the_consolidated_current_period_figure_wins(self):
        rows = [
            {"要素ID": "jppfs_cor:Assets", "項目名": "資産合計",
             "コンテキストID": "CurrentYearInstant_NonConsolidatedMember", "値": "1", "単位": "JPY"},
            {"要素ID": "jppfs_cor:Assets", "項目名": "資産合計",
             "コンテキストID": "CurrentYearInstant", "値": "2", "単位": "JPY"},
        ]
        assert edinet._pick(rows, ("Assets",), ("資産合計",)) == ("2", "JPY")

    def test_a_parent_only_filer_still_reports(self):
        rows = [{"要素ID": "jppfs_cor:Assets", "項目名": "資産合計",
                 "コンテキストID": "CurrentYearInstant_NonConsolidatedMember",
                 "値": "7", "単位": "JPY"}]
        assert edinet._pick(rows, ("Assets",), ("資産合計",)) == ("7", "JPY")

    def test_prior_periods_and_segments_are_rejected(self):
        rows = [
            {"要素ID": "jppfs_cor:Assets", "項目名": "資産合計",
             "コンテキストID": "Prior1YearInstant", "値": "1", "単位": "JPY"},
            {"要素ID": "jppfs_cor:Assets", "項目名": "資産合計",
             "コンテキストID": "CurrentYearInstant_AutomotiveReportableSegmentsMember",
             "値": "2", "単位": "JPY"},
        ]
        assert edinet._pick(rows, ("Assets",), ("資産合計",)) is None

    def test_an_ifrs_filer_matches_on_its_japanese_label(self):
        # The IFRS taxonomy names the element differently; the label is what
        # keeps an IFRS filer from reading as an empty statement.
        rows = [{"要素ID": "jpigp_cor:SomeElementWeDoNotEnumerate", "項目名": "売上収益",
                 "コンテキストID": "CurrentYearDuration", "値": "48000000", "単位": "JPY"}]
        picked = edinet._pick(rows, ("NetSales",), ("売上高", "売上収益"))
        assert picked == ("48000000", "JPY")

    def test_a_dash_placeholder_is_not_a_value(self):
        rows = [{"要素ID": "jppfs_cor:GrossProfit", "項目名": "売上総利益",
                 "コンテキストID": "CurrentYearDuration", "値": "－", "単位": "JPY"}]
        assert edinet._pick(rows, ("GrossProfit",), ("売上総利益",)) is None


@pytest.mark.unit
class TestStatements:
    def _wire(self, monkeypatch, filings, rows):
        monkeypatch.setattr(edinet, "find_filings", lambda *a, **k: filings)
        monkeypatch.setattr(edinet, "_document_csv", lambda doc_id: rows)
        monkeypatch.setattr(edinet.time, "sleep", lambda *_: None)

    def test_a_balance_sheet_reports_its_filing_date_not_just_its_period(self, monkeypatch):
        self._wire(
            monkeypatch,
            [{"docID": "S1", "docTypeCode": "120", "periodEnd": "2026-03-31",
              "submitDateTime": "2026-06-25 09:30", "filerName": "トヨタ自動車株式会社",
              "docDescription": "有価証券報告書"}],
            [_parsed("jppfs_cor:Assets", "資産合計", "CurrentYearInstant", "95000000000000"),
             _parsed("jppfs_cor:Liabilities", "負債合計", "CurrentYearInstant", "55000000000000")],
        )
        report = edinet.get_balance_sheet("7203.T", "annual", "2026-09-01")
        assert "as filed" in report
        assert "Point-in-time as of: 2026-09-01" in report
        assert "filed 2026-06-25" in report
        assert "トヨタ自動車株式会社" in report
        # Yen reported in millions, like the other statement vendors.
        assert "95,000,000.0" in report

    def test_a_half_year_report_is_not_presented_as_a_quarter(self, monkeypatch):
        self._wire(
            monkeypatch,
            [{"docID": "S1", "docTypeCode": "160", "periodEnd": "2026-09-30",
              "submitDateTime": "2026-11-14 15:00", "filerName": "X", "docDescription": "半期報告書"}],
            [_parsed("jppfs_cor:NetSales", "売上高", "CurrentYearDuration", "1000000")],
        )
        report = edinet.get_income_statement("7203.T", "quarterly", "2026-12-01")
        assert "半期報告書" in report
        assert "not a quarter" in report

    def test_a_non_tokyo_ticker_falls_through_to_the_next_vendor(self):
        with pytest.raises(NoMarketDataError):
            edinet.get_balance_sheet("AAPL", "annual", "2026-09-01")

    def test_no_filing_by_the_analysis_date_is_no_data(self, monkeypatch):
        monkeypatch.setattr(edinet, "find_filings", lambda *a, **k: [])
        with pytest.raises(NoMarketDataError) as caught:
            edinet.get_balance_sheet("7203.T", "annual", "2026-05-01")
        assert "by 2026-05-01" in str(caught.value)

    def test_a_missing_key_is_reported_as_unconfigured(self, monkeypatch):
        monkeypatch.delenv("EDINET_API_KEY", raising=False)
        with pytest.raises(VendorNotConfiguredError):
            edinet.get_api_key()


@pytest.mark.unit
def test_the_vendor_is_routable_for_every_statement():
    for method in ("get_balance_sheet", "get_cashflow", "get_income_statement"):
        assert "edinet" in interface.VENDOR_METHODS[method]
    assert "edinet" in interface.VENDOR_LIST
