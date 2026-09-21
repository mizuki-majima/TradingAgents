"""EDINET DB serves Japanese annual figures with the filing date attached.

The leak this closes is the same one ``edinet`` closes, by a cheaper route: a
Japanese filer publishes its 有価証券報告書 about three months after the year
ends, so a backtest dated inside that gap must not see the year it has not read
yet. Verified against the live API while these were written — for 7203.T an
analysis dated 2026-09-18 sees FY2026 (filed 2026-06-10) and one dated
2026-05-01 does not, stopping at FY2025.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from tradingagents.dataflows import edinet, edinetdb, interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)

_COMPANY = {
    "edinet_code": "E02144", "sec_code": "72030", "name_ja": "トヨタ自動車株式会社",
    "name_en": "TOYOTA MOTOR CORPORATION", "accounting_standard": "IFRS",
}

_YEARS = [
    {"fiscal_year": 2024, "submit_date": "2024-06-25 15:00", "accounting_standard": "IFRS",
     "basis": "consolidated", "revenue": 45095325000000.0, "net_income": 4944933000000.0,
     "eps": 365.94, "total_assets": 90114296000000.0},
    {"fiscal_year": 2025, "submit_date": "2025-06-18 15:00", "accounting_standard": "IFRS",
     "basis": "consolidated", "revenue": 48036704000000.0, "net_income": 4765086000000.0,
     "eps": 359.56, "total_assets": 93601350000000.0, "is_restated_eps": True},
    {"fiscal_year": 2026, "submit_date": "2026-06-10 15:33", "accounting_standard": "IFRS",
     "basis": "consolidated", "revenue": 50684952000000.0, "net_income": 3848098000000.0,
     "eps": 295.25, "total_assets": 105522331000000.0,
     "edinet_filing_url": "https://disclosure2.edinet-fsa.go.jp/WZEK0040.aspx?S100Y8NY"},
]


def _response(payload=None, status=200):
    response = mock.Mock()
    response.status_code = status
    response.json.return_value = payload or {}
    response.content = json.dumps(payload or {}).encode()
    return response


@pytest.fixture(autouse=True)
def _wire(monkeypatch, tmp_path):
    monkeypatch.setenv("EDINETDB_API_KEY", "edb_test")
    monkeypatch.delenv("EDINET_API_KEY", raising=False)
    set_config({"data_cache_dir": str(tmp_path)})

    def _fake(url, **kwargs):
        if "/companies/" in url and url.endswith("/financials"):
            return _response({"data": _YEARS})
        if url.endswith("/companies"):
            code = kwargs["params"].get("sec_code")
            return _response({"data": [_COMPANY] if code == "7203" else []})
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(edinetdb, "get_scrubbed", _fake)


@pytest.mark.unit
class TestPointInTime:
    def test_a_year_is_invisible_until_its_filing_date(self):
        """The whole reason this vendor exists."""
        before = edinetdb.get_income_statement("7203.T", "annual", "2026-05-01")
        after = edinetdb.get_income_statement("7203.T", "annual", "2026-09-18")
        assert "FY2026" not in before
        assert "FY2025 (filed 2025-06-18)" in before
        assert "FY2026 (filed 2026-06-10)" in after

    def test_the_filing_date_is_shown_beside_every_column(self):
        report = edinetdb.get_balance_sheet("7203.T", "annual", "2026-09-18")
        for stamp in ("filed 2026-06-10", "filed 2025-06-18", "filed 2024-06-25"):
            assert stamp in report

    def test_a_row_with_no_filing_date_is_dropped_not_assumed_public(self):
        rows = [{"fiscal_year": 2027, "revenue": 1.0}, *_YEARS]
        kept = edinetdb._as_of(rows, "2030-01-01")
        assert all(r.get("fiscal_year") != 2027 for r in kept)

    def test_nothing_filed_by_the_analysis_date_is_no_data(self):
        with pytest.raises(NoMarketDataError) as caught:
            edinetdb.get_income_statement("7203.T", "annual", "2000-01-01")
        assert "by 2000-01-01" in str(caught.value)

    def test_the_restatement_caveat_is_stated_where_it_applies(self):
        report = edinetdb.get_income_statement("7203.T", "annual", "2026-09-18")
        # The figures are as currently reported, which SEC EDGAR would not do;
        # the header says so and the affected years are named.
        assert "not the vintage that was current on the analysis date" in report
        assert "Restated since first reported: FY2025" in report


@pytest.mark.unit
class TestRendering:
    def test_yen_is_reported_in_millions_but_per_share_figures_are_not(self):
        report = edinetdb.get_income_statement("7203.T", "annual", "2026-09-18")
        assert "50,684,952.0" in report   # revenue, yen millions
        assert "295.25" in report         # EPS, yen per share

    def test_the_filer_is_named_in_japanese_and_the_standard_is_stated(self):
        report = edinetdb.get_balance_sheet("7203.T", "annual", "2026-09-18")
        assert "トヨタ自動車株式会社" in report
        assert "Accounting standard: IFRS" in report

    def test_a_line_nobody_reported_is_left_out_entirely(self):
        # These IFRS rows carry no ordinary_income; an all-"n/a" row would only
        # invite the agent to reason about a blank.
        assert "Ordinary Income" not in edinetdb.get_income_statement(
            "7203.T", "annual", "2026-09-18")

    def test_the_period_count_is_configurable(self):
        set_config({"edinetdb_periods": 1})
        report = edinetdb.get_income_statement("7203.T", "annual", "2026-09-18")
        assert "FY2026" in report and "FY2025" not in report


@pytest.mark.unit
class TestChainBehaviour:
    def test_an_interim_request_falls_through_to_a_vendor_that_has_one(self):
        with pytest.raises(NoMarketDataError) as caught:
            edinetdb.get_income_statement("7203.T", "quarterly", "2026-09-18")
        assert "annual figures only" in str(caught.value)

    def test_a_non_tokyo_ticker_falls_through(self):
        for ticker in ("AAPL", "0700.HK"):
            with pytest.raises(NoMarketDataError):
                edinetdb.get_balance_sheet(ticker, "annual", "2026-09-18")

    def test_an_unlisted_code_is_no_data(self, monkeypatch):
        monkeypatch.setattr(edinetdb, "get_scrubbed",
                            lambda url, **k: _response({"data": []}))
        with pytest.raises(NoMarketDataError):
            edinetdb.get_balance_sheet("9999.T", "annual", "2026-09-18")

    def test_an_exhausted_quota_is_not_reported_as_an_absence_of_filings(self, monkeypatch):
        monkeypatch.setattr(edinetdb, "get_scrubbed",
                            lambda url, **k: _response(status=429))
        with pytest.raises(VendorRateLimitError):
            edinetdb.get_balance_sheet("7203.T", "annual", "2026-09-18")

    def test_a_repeat_read_spends_no_further_request(self, monkeypatch):
        calls = []
        original = edinetdb.get_scrubbed
        monkeypatch.setattr(edinetdb, "get_scrubbed",
                            lambda url, **k: calls.append(url) or original(url, **k))
        edinetdb.get_balance_sheet("7203.T", "annual", "2026-09-18")
        spent = len(calls)
        edinetdb.get_income_statement("7203.T", "annual", "2026-09-18")
        assert len(calls) == spent, "the free tier is a daily budget; cache it"

    def test_the_vendor_is_routable_for_every_statement(self):
        for method in ("get_balance_sheet", "get_cashflow", "get_income_statement"):
            assert "edinetdb" in interface.VENDOR_METHODS[method]
        assert "edinetdb" in interface.VENDOR_LIST


@pytest.mark.unit
class TestKeyRouting:
    def test_an_edinetdb_key_left_in_the_official_variable_is_still_used(self, monkeypatch):
        monkeypatch.delenv("EDINETDB_API_KEY", raising=False)
        monkeypatch.setenv("EDINET_API_KEY", "edb_abcdef")
        assert edinetdb.get_api_key() == "edb_abcdef"

    def test_an_official_key_is_not_mistaken_for_an_edinetdb_one(self, monkeypatch):
        monkeypatch.delenv("EDINETDB_API_KEY", raising=False)
        monkeypatch.setenv("EDINET_API_KEY", "0123456789abcdef0123456789abcdef")
        with pytest.raises(VendorNotConfiguredError):
            edinetdb.get_api_key()

    def test_the_official_vendor_explains_an_edinetdb_key_instead_of_401ing(self, monkeypatch):
        """Verified live: 金融庁's API answers a bad key with 200 + StatusCode 401.

        Sent blindly, that reads as "this company filed nothing" on every date,
        so the key is recognised before the request is made.
        """
        monkeypatch.setenv("EDINET_API_KEY", "edb_abcdef")
        with pytest.raises(VendorNotConfiguredError) as caught:
            edinet.get_api_key()
        assert "edinetdb" in str(caught.value)
