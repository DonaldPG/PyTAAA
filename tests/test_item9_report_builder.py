"""Tests for functions/report_builders.py (Item 9 — HTML report builder).

These tests verify that build_holdings_html_report() renders the expected
HTML structure from structured input data without touching the filesystem
or making any network calls.
"""

import pytest


##############################################################################
# Fixtures
##############################################################################

@pytest.fixture
def sample_row():
    """Return a minimal pre-formatted holding-row dict."""
    return {
        "symbol": "AAPL ",
        "shares": "   100",
        "buy_price": "150.00",
        "purchase_cost": "15000.00",
        "cumu_purchase": "15000.00",
        "current_price": "175.00",
        "profit_pct": " 16.67%",
        "value": "17500.00",
        "cumu_value": "17500.00",
        "rank": "  5",
        "cluster": "2",
        "sector": "Technology",
        "industry": "Consumer Electronics",
    }


@pytest.fixture
def sample_params():
    return {
        "stockList": "Naz100",
        "uptrendSignalMethod": "hma",
    }


def _call(rows, trade_message="<br>", removed=None, added=None, **kwargs):
    """Helper that invokes build_holdings_html_report with sensible defaults."""
    from functions.report_builders import build_holdings_html_report

    return build_holdings_html_report(
        holdings_rows=rows,
        trade_message=trade_message,
        lifetime_profit=kwargs.get("lifetime_profit", 7500.0),
        cumulative_cash_in=kwargs.get("cumulative_cash_in", 10000.0),
        lifetime_profit_annualized=kwargs.get(
            "lifetime_profit_annualized", 7500.0
        ),
        removed_tickers=removed or [],
        added_tickers=added or [],
        edition=kwargs.get("edition", "1.0"),
        ip=kwargs.get("ip", "127.0.0.1"),
        params=kwargs.get(
            "params", {"stockList": "Naz100", "uptrendSignalMethod": "hma"}
        ),
    )


##############################################################################
# Tests
##############################################################################

class TestBuildHoldingsHtmlReport:

    def test_returns_string(self, sample_row):
        """build_holdings_html_report returns a non-empty string."""
        html = _call([sample_row])
        assert isinstance(html, str)
        assert len(html) > 0

    def test_table_header_present(self, sample_row):
        """Output contains the holdings table header."""
        html = _call([sample_row])
        assert "<h3>Current stocks and weights are :</h3>" in html
        assert "<table" in html

    def test_row_symbol_present(self, sample_row):
        """Symbol from the row dict appears in the rendered HTML."""
        html = _call([sample_row])
        assert "AAPL" in html

    def test_multiple_rows_rendered(self, sample_row):
        """Multiple holding rows are all rendered."""
        other = dict(sample_row)
        other["symbol"] = "MSFT "
        other["sector"] = "Software"
        html = _call([sample_row, other])
        assert "AAPL" in html
        assert "MSFT" in html

    def test_empty_rows_allowed(self):
        """An empty holdings list produces valid (empty-table) HTML."""
        html = _call([])
        assert "<table" in html

    def test_trade_message_included(self, sample_row):
        """Trade message HTML is embedded in the output."""
        html = _call([sample_row], trade_message="<br>TRADE: Buy XYZ")
        assert "TRADE: Buy XYZ" in html

    def test_lifetime_profit_formatted(self, sample_row):
        """Lifetime profit percentage appears in the output."""
        html = _call([sample_row], lifetime_profit=2500.0, cumulative_cash_in=10000.0)
        # 2500 / 10000 = 25.0%
        assert "25.0%" in html

    def test_removed_tickers_listed(self, sample_row):
        """Removed tickers are mentioned when present."""
        html = _call([sample_row], removed=["OLD1", "OLD2"])
        assert "OLD1" in html
        assert "OLD2" in html
        assert "removed" in html

    def test_added_tickers_listed(self, sample_row):
        """Added tickers are mentioned when present."""
        html = _call([sample_row], added=["NEW1"])
        assert "NEW1" in html
        assert "added" in html

    def test_no_change_block_when_empty(self, sample_row):
        """No stock-list-change block when removed/added lists are empty."""
        html = _call([sample_row], removed=[], added=[])
        assert "There are changes in the stock list" not in html

    def test_edition_and_ip_in_footer(self, sample_row):
        """Edition and IP appear in the footer."""
        html = _call([sample_row], edition="2.5", ip="192.168.1.1")
        assert "2.5" in html
        assert "192.168.1.1" in html

    def test_stock_list_in_footer(self, sample_row):
        """Stock list parameter appears in the footer."""
        html = _call(
            [sample_row],
            params={"stockList": "SP500", "uptrendSignalMethod": "sma"},
        )
        assert "SP500" in html

    def test_uptrend_method_in_footer(self, sample_row):
        """Uptrend signal method parameter appears in the footer."""
        html = _call(
            [sample_row],
            params={"stockList": "Naz100", "uptrendSignalMethod": "pine"},
        )
        assert "pine" in html

    def test_sector_and_industry_in_row(self, sample_row):
        """Sector and industry values appear in the table row."""
        html = _call([sample_row])
        assert "Technology" in html
        assert "Consumer Electronics" in html
